import torch.nn as nn
import torch

def to_gray(tensor):
    # 假设输入形状为 [batch, channels, height, width]
    # 计算所有通道的平均值作为灰度
    gray_tensor = torch.mean(tensor, dim=1, keepdim=True)
    return gray_tensor

def tv_loss(image, reduction='mean'):
    """计算图像的总变差（TV）损失（L1-TV）"""
    # 图像 shape: (batch_size, channels, height, width)
    batch_size = image.shape[0]
    # 计算水平方向相邻像素差异（h方向）
    h_diff = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])  # (B, C, H-1, W)
    # 计算垂直方向相邻像素差异（w方向）
    w_diff = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1])  # (B, C, H, W-1)
    # 求和（或平均）
    if reduction == 'mean':
        loss = (h_diff.mean() + w_diff.mean()) / batch_size
    elif reduction == 'sum':
        loss = (h_diff.sum() + w_diff.sum()) / batch_size
    return loss

class Losses(nn.Module):
    """
    自定义损失函数类
    包括以下损失项：
    1. Z_pred 和 Z 之间的l1损失
    """
    def __init__(self, scale, model_name='zsl', blur=0):  # 添加通道数参数
        super(Losses, self).__init__()
        self.mse_loss = nn.MSELoss()  # 均方误差损失
        # L1 范数损失
        self.l1_loss = nn.L1Loss()
        self.model_name = model_name  # 模型名称，用于选择不同的损失函数
        self.scale = scale  
        if scale==8:
            self.weight = 0.05 if blur>0  else 0
        else:
            self.weight = 0.35 if blur>0 else 0.05




    def forward(self, Z_pred, Z, y_, y_blur,epoch):
        # print(f"Z_pred shape: {Z_pred.shape}, Z shape: {Z.shape}")

        l1_loss = None
        weight=0.005*10/(len(Z_pred))
        for i in range(len(Z_pred)):
            if l1_loss is None:
                l1_loss = self.l1_loss(Z_pred[i], Z)*weight
            else:
                l1_loss += self.l1_loss(Z_pred[i], Z)*weight
                weight = weight * 2  # 权重逐渐增大

        total_loss = l1_loss
        # 传入通道数参数
        GL = GradientLoss(device=Z.device, in_channels=1)  # 确保损失函数在正确的设备上
        if epoch<40:
            total_loss = l1_loss + 0.01 * GL(to_gray(y_), to_gray(Z))
        else:
            total_loss = l1_loss + self.weight * GL(to_gray(y_), to_gray(Z))
 
        return total_loss

class GradientLoss(nn.Module):
    def __init__(self, device, in_channels=3):  # 接收设备参数和通道数参数
        super().__init__()
        # Sobel算子（水平和垂直方向）
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # 根据输入通道数调整Sobel算子
        self.sobel_x = self.sobel_x.repeat(1, in_channels, 1, 1).to(device)  # 转移到指定设备
        self.sobel_y = self.sobel_y.repeat(1, in_channels, 1, 1).to(device)  # 转移到指定设备
        
        # 使用输入通道数参数配置卷积层
        self.conv_x = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, bias=False).to(device)  # 转移到指定设备
        self.conv_y = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, bias=False).to(device)  # 转移到指定设备
        
        self.conv_x.weight = nn.Parameter(self.sobel_x)
        self.conv_y.weight = nn.Parameter(self.sobel_y)
        
    def forward(self, pred, target):
        # 确保输入的通道数与初始化时设置的一致
        assert pred.shape[1] == self.conv_x.in_channels, \
            f"输入通道数 {pred.shape[1]} 与损失函数配置的通道数 {self.conv_x.in_channels} 不匹配"
        
        # 计算梯度
        pred_grad_x = self.conv_x(pred)
        pred_grad_y = self.conv_y(pred)
        target_grad_x = self.conv_x(target)
        target_grad_y = self.conv_y(target)
        
        # L1损失
        loss = torch.mean(torch.abs(pred_grad_x - target_grad_x) + torch.abs(pred_grad_y - target_grad_y))
        return loss

