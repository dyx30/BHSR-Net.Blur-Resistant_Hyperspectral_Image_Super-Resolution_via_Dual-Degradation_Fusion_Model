import torch
from torch import nn
import torch.nn.functional as F
from modules import *
from torch.nn import functional
from thop import profile, clever_format

class UM(nn.Module):  
    def __init__(self, stage_num, C, c, sigma1, sigma2):
        super(UM, self).__init__()
        self.stage_num = stage_num
        
        
        # 创建多个阶段的模块列表
        self.gx_modules = nn.ModuleList([GX(C, c, sigma1) for _ in range(stage_num)])
        self.x_solvers = nn.ModuleList([XSolver(C) for _ in range(stage_num)])
        self.l1_mus = nn.ModuleList([L1Updater(C, sigma1) for _ in range(stage_num)])

        self.y_solvers = nn.ModuleList([YSolver(c, C, sigma2) for _ in range(stage_num)])
        self.l2_mus = nn.ModuleList([L2Updater(c, sigma2) for _ in range(stage_num)])

        # 添加可学习的参数，用于控制步长
        self.gx_update_param = nn.Parameter(torch.tensor(1.0))  # 初始值为1
        self.l1_update_param = nn.Parameter(torch.tensor(1.0))  # 初始值为1
        self.y_update_param = nn.Parameter(torch.tensor(1.0))  # 初始值为1
        self.l2_update_param = nn.Parameter(torch.tensor(1.0)) # 初始值为1

        # self.iniX= ZSLcnn(C, c)
    def forward(self, z, y):
        # z[B, C, h, w] y[B, c, H, W]
      
        # 获取 z 和 y 的通道数和尺寸
        H, W = y.shape[2], y.shape[3]
        z_ = F.interpolate(z, size=(H, W), mode='bilinear', align_corners=False)
        y_ = y
        x = z_
        # x=self.iniX(z, y)  # 初始重构
        L1 = torch.zeros_like(x)  # 初始化为全1
        L2 = torch.zeros_like(y)  # 初始化为全1

        Xs = []


        # 多阶段处理
        for i in range(self.stage_num):
            # 使用对应阶段的模块

            gx = self.gx_modules[i](x, L1, y_, z_)
            x = x + self.x_solvers[i](gx) * self.gx_update_param

            L1 = L1 + self.l1_mus[i](L1, x)* self.l1_update_param  

            y_ = y_ + self.y_solvers[i](y, y_, x, L2) * self.y_update_param
            L2 = L2 + self.l2_mus[i](L2, y_) * self.l2_update_param
    
            Xs.append(x)

        return Xs,y_





class ZSLcnn(nn.Module): # 纯卷积层，只对x空间上采样，y每次加入到z中卷积
    def __init__(self, C, c):
        super(ZSLcnn, self).__init__()
        self.conv1 = nn.Sequential(        
            nn.Conv2d(C + c, 64 - c, 3, 1, 1),     
            nn.LeakyReLU(negative_slope=0.2, inplace=False), 
        )

        self.conv4 = nn.Sequential(        
            nn.Conv2d(64, C, 3, 1, 1),     
        )
        self.c = c
        self.C = C

    def forward(self, x, y):
        # 获取 x 和 y 的通道数和尺寸
        W = y.shape[3]
        w = x.shape[3]
        scale_factor = W / w
        x1 = functional.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        

        z1 = torch.cat((x1, y), dim=1)
        z2 = torch.cat((self.conv1(z1), y), dim=1)

        # 最后一层卷积
        z3 = self.conv4(z2)
        return z3
    
class fullZSLcnn(nn.Module): # 纯卷积层，只对x空间上采样，y每次加入到z中卷积
    def __init__(self, C, c):
        super(ZSLcnn, self).__init__()
        self.conv1 = nn.Sequential(        
            nn.Conv2d(C + c, 128 - c, 3, 1, 1),     
            nn.LeakyReLU(negative_slope=0.2, inplace=False), 
        )
        self.conv2 = nn.Sequential(        
            nn.Conv2d(128, 128 - c, 3, 1, 1),     
            nn.LeakyReLU(negative_slope=0.2, inplace=False), 
        )
        self.conv3 = nn.Sequential(        
            nn.Conv2d(128, 128 - c, 3, 1, 1),     
            nn.LeakyReLU(negative_slope=0.2, inplace=False), 
        )
        self.conv4 = nn.Sequential(        
            nn.Conv2d(128, C, 3, 1, 1),     
        )
        self.c = c
        self.C = C

    def forward(self, x, y):
        # 获取 x 和 y 的通道数和尺寸
        W = y.shape[3]
        w = x.shape[3]
        scale_factor = W / w
        x1 = functional.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        

        z1 = torch.cat((x1, y), dim=1)
        z2 = torch.cat((self.conv1(z1), y), dim=1)
        z2 = torch.cat((self.conv2(z2), y), dim=1)
        z2 = torch.cat((self.conv3(z2), y), dim=1)


        # 最后一层卷积
        z3 = self.conv4(z2)
        return z3
    


def calculate_model_flops_params(model, input_shapes, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    计算模型的FLOPs和参数数量，使用float32输入
    
    参数:
        model: 要评估的PyTorch模型
        input_shapes: 输入形状的元组，例如((1, 46, 64, 64), (1, 3, 256, 256))
        device: 计算设备
    
    返回:
        flops: 浮点运算量
        params: 参数数量
    """
    # 创建随机输入张量，明确指定为float32类型
    inputs = [torch.randn(*shape, dtype=torch.float32).to(device) for shape in input_shapes]
    
    # 确保模型处于评估模式并转移到指定设备
    model.eval()
    model.to(device)
    
    # 使用thop计算FLOPs和参数
    flops, params = profile(model, inputs=inputs)
    
    # 格式化输出（自动转换为合适的单位）
    flops, params = clever_format([flops, params], "%.3f")
    
    return flops, params


# 示例用法中也确保输入形状对应的张量为float32
if __name__ == "__main__":
    # 模型参数

    C = 31  # 高光谱图像通道数
    c = 3   # 多光谱图像通道数
    sigma1 = 1.0
    sigma2 = 1.0
    

    
    # 定义输入形状 - 可以根据需要修改
    input_shapes = (
        (1, C, 16, 16),    # LrHSI 输入形状
        (1, c, 512, 512)   # HrMSI 输入形状
    )
    for i in range(1,2):
        print(f"————————————{i}stage————————————")
        stage_num = 12
        # 创建模型
        model = UM(stage_num, C, c, sigma1, sigma2)
        # 计算FLOPs和参数
        flops, params = calculate_model_flops_params(model, input_shapes)
        
        # 输出结果
        print(f"模型参数: {params}")
        print(f"FLOPs: {flops}")
        print(f"输入形状1 (LrHSI): {input_shapes[0]}, 数据类型: float32")
        print(f"输入形状2 (HrMSI): {input_shapes[1]}, 数据类型: float32")
    