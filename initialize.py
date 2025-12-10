# 预处理阶段的退化
import numpy as np
import scipy.io as sio
import os
import torch
import torch.nn as nn
from scipy.ndimage import zoom
import torch.utils.data as data
from PIL import Image
from tqdm import tqdm  # 导入tqdm库

def crop_Harvard(input_dir, output_dir, save_data_key='hsi'):
    """
    裁剪指定目录下的MAT文件，居中裁剪为1024×1280×31
    
    参数:
        input_dir: 原始MAT文件所在目录
        output_dir: 裁剪后文件的保存目录
        save_data_key: MAT文件中存储图像数据的键（默认'hsi'，可根据实际修改）
    """
    # 创建输出目录（若不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有MAT文件
    mat_files = [f for f in os.listdir(input_dir) if f.endswith('.mat')]
    if not mat_files:
        print(f"未在{input_dir}中找到MAT文件")
        return
    
    # 裁剪参数（目标尺寸）
    target_h, target_w, target_c = 1024, 1280, 31
    
    # 遍历处理每个文件
    for mat_file in tqdm(mat_files, desc="裁剪进度"):
        input_path = os.path.join(input_dir, mat_file)
        output_path = os.path.join(output_dir, mat_file)
        
        try:
            # 读取MAT文件
            data = sio.loadmat(input_path)
            hsi = data['ref']  # 获取图像数据
            
            # 检查原始尺寸是否符合预期
            if hsi.shape != (1040, 1392, 31):
                print(f"警告：{mat_file}的尺寸为{hsi.shape}，不符合1040×1392×31，已跳过")
                continue
            
            # 计算居中裁剪的起始坐标
            start_h = (hsi.shape[0] - target_h) // 2  # (1040-1024)/2 = 8
            start_w = (hsi.shape[1] - target_w) // 2  # (1392-1280)/2 = 56
            
            # 执行居中裁剪
            cropped_hsi = hsi[
                start_h : start_h + target_h,  # 高度方向裁剪
                start_w : start_w + target_w,  # 宽度方向裁剪
                :  # 保留所有31个通道
            ]
            
            # 验证裁剪后尺寸
            if cropped_hsi.shape != (target_h, target_w, target_c):
                print(f"警告：{mat_file}裁剪后尺寸异常{cropped_hsi.shape}，已跳过")
                continue
            
            # 保存裁剪后的文件（保留'size'和数据键）
            sio.savemat(
                output_path,
                {
                    'size': cropped_hsi.shape,  # 存储裁剪后的尺寸
                    save_data_key: cropped_hsi       # 存储裁剪后的图像数据
                }
            )
            
        except Exception as e:
            print(f"处理{mat_file}时出错：{str(e)}")
            continue
    
    print(f"所有文件处理完成，裁剪后文件保存至：{output_dir}")

# watercolors_ms的格式是rgba
def rgba_to_uint16_grayscale(img):
    assert img.mode == "RGBA"

    # 转为 numpy 数组
    arr = np.array(img).astype(np.float32)  # shape: (H, W, 4)

    # 取 RGB 部分 (忽略 Alpha)，转灰度
    rgb = arr[:, :, :3]  # shape: (H, W, 3)
    gray = np.dot(rgb, [0.2989, 0.5870, 0.1140])  # shape: (H, W)，float32

    # 归一化后转为 uint16
    gray_norm = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
    gray_uint16 = (gray_norm * 65535).round().astype(np.uint16)

    return gray_uint16

def CAVE_PNG_to_HSI(raw_path='rawdata/CAVE',data_path='/mnt/disk3/dyx/data/CAVE'): #将CAVE数据集的PNG图像拼接成高光谱格式
    # raw_path路径下有32个文件夹，名称以_ms结尾
    # 对每一个文件夹，下有31个PNG文件，均以_xx结尾，xx为数字编号，从01到31
    # 按照编号顺序读取每个文件夹下的PNG文件，将每个文件夹下的PNG文件拼接成512*512*31的高光谱图像，使用scipy将其保存为mat格式，有两个键，一个是"size"，值为(512,512,31)，一个是"hsi"，值为高光谱图像
    # 命名为文件夹名称去掉结尾的_ms.mat，保存到data_path路径下
    
    # 遍历raw_path下的文件夹
    # 获取所有符合条件的文件夹
    folder_names = [folder for folder in os.listdir(raw_path) if folder.endswith('_ms')]

    # 创建一个进度条来展示文件夹处理的进度
    with tqdm(total=len(folder_names), desc="Processing folders", unit="folder") as pbar:
        # 只打印一次 "Generating HSI from rawdata:"，然后在后续更新文件夹名称
        tqdm.write('Generating HSI from rawdata:')  # 打印一次固定的部分
        
        for folder_name in folder_names:
            folder_path = os.path.join(raw_path, folder_name)
            
            # 初始化一个空的列表，用于存放该文件夹下的PNG图像
            images = []
            
            # 使用tqdm.set_postfix()来显示当前处理的文件夹
            pbar.set_postfix({'Folder': folder_name})
            print(f"Processing folder: {folder_name}")  # 打印当前处理的文件夹
            if folder_name == 'watercolors_ms':
                print("catch you! watercolor:rgba type !")
            
            # 遍历该文件夹下的PNG文件，文件名以_xx结尾
            for i in range(1, 32):  # 文件编号从01到31
                file_name = f"{folder_name}_{i:02d}.png"  # 拼接文件名
                file_path = os.path.join(folder_path, file_name)
                img = Image.open(file_path)
                img_array = None
                # 读取PNG图像
                if folder_name == 'watercolors_ms':
                    img_array = rgba_to_uint16_grayscale(img)
                else:
                    img_array = np.array(img, dtype=np.uint16)
                images.append(img_array)  # 将读取的图像添加到列表中
            
            # 将所有PNG图像堆叠成一个512x512x31的高光谱图像
            hsi_image = np.stack(images, axis=-1)
            
            # 创建MAT文件保存路径
            save_path = os.path.join(data_path, f"{folder_name[:-3]}.mat")  # 去掉_ms后缀
            
            # 将高光谱图像保存为MAT文件
            sio.savemat(save_path, {
                'size': hsi_image.shape,  # 高光谱图像的尺寸
                'hsi': hsi_image          # 高光谱图像数据
            })
            
            # 每处理完一个文件夹，更新进度条
            pbar.update(1)
    
    
    tqdm.write(f"All images in {raw_path} have been converted to HSI format and saved to {data_path}.")
    

def getAndSaveCAVE(path='/mnt/disk3/dyx/data/CAVE/gt',dataset='CAVE'): # 读取CAVE高光谱数据rawdata，进行空间和光谱降采样
    # 检查data中有无CAVE的32个mat文件，否则使用CAVE_PNG_to_HSI()函数读取rawData
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        files = [f for f in os.listdir(path) if f.endswith('.mat')]
        if len(files) != 32:
            print(f"Loading CAVE")
            CAVE_PNG_to_HSI(raw_path='rawdata/CAVE', data_path=path)
            return
        else:
            return


def getPSF(kernel_size=7, sigma=1.0, scale=32, anisotropy=False, Gaussian=False, angle=0, predefined_kernel=None):
    # anisotropy: 是否为各向异性高斯核
    # kernel_size: 卷积核大小, 默认为7
    # sigma: 高斯核标准差, 默认为1.0
    # nonGaussian: 是否为非高斯核, 默认为False

    kernel = np.zeros((kernel_size, kernel_size))  # 初始化卷积核为零矩阵
    kernel_name = ''

    if predefined_kernel is not None:  
        # 从predefined_kernel中读取卷积核(scipy打开mat文件)
        kernel = sio.loadmat(predefined_kernel)['kernel']
        kernel = np.array(kernel)
        # 如果 kernel 的大小小于等于 kernel_size ，将其插值上采样至尺寸 = kernel_size
        if kernel.shape[0] <= kernel_size or kernel.shape[0] > kernel_size:
            # 计算缩放因子
            zoom_factor = kernel_size / kernel.shape[0]
            # 使用双线性插值
            kernel = zoom(kernel, zoom=(zoom_factor, zoom_factor), order=1)  # order=1 表示双线性插值
            # 确保最终尺寸为 (kernelsize + 1, kernelsize + 1)
            kernel = kernel[:kernel_size, :kernel_size]

    else:
        if anisotropy:
            # 生成各向异性高斯核
            theta = np.deg2rad(angle)  # 将角度转换为弧度
            R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])  # 旋转矩阵
            D = np.diag([4 * sigma, sigma])  # 标准差矩阵
            Lambda = R @ D @ R.T  # 协方差矩阵

            # 生成网格
            ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
            xx, yy = np.meshgrid(ax, ax)
            grid = np.stack([xx, yy], axis=-1)  # 形状为 (kernel_size, kernel_size, 2)

            # 计算高斯核
            inv_Lambda = np.linalg.inv(Lambda)
            det_Lambda = np.linalg.det(Lambda)
            kernel = (1 / (2 * np.pi * np.sqrt(det_Lambda))) * \
                    np.exp(-0.5 * np.einsum('...i,ij,...j->...', grid, inv_Lambda, grid))

        else:
            # 生成各向同性高斯核
            ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
            xx, yy = np.meshgrid(ax, ax)
            kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))

    # 确保 kernel 是 float32 类型
    kernel = kernel.astype(np.float32)

    # 归一化处理，使得卷积核的和为 1
    kernel /= np.sum(kernel)

    # 返回的命名,命名规则为：
    # 如果是各向异性高斯核则Gaussian_aniso_开头+size+sigma+angle
    # 如果是各向同性高斯核则Gaussian_iso_开头+size+sigma
    # 如果是非高斯核则NonGaussian_开头+size+predefined_kernel的路径(路径倒数第二个目录+去掉后缀的文件名称)
    if not Gaussian:
        # 非高斯核命名规则
        kernel_name = f"NonGaussian_{kernel_size}_{os.path.basename(os.path.dirname(os.path.dirname(predefined_kernel)))}_{os.path.splitext(os.path.basename(predefined_kernel))[0]}"
    elif anisotropy:
        # 各向异性高斯核命名规则
        kernel_name = f"Gaussian_aniso_size{kernel_size}_sigma{sigma}_angle{angle}"
    else:
        # 各向同性高斯核命名规则
        kernel_name = f"Gaussian_iso_size{kernel_size}_sigma{sigma}"


    return kernel, kernel_name



def getSRF(predefined_SRF):

    SRF = sio.loadmat(predefined_SRF)
    keys = SRF.keys()
    srf= None
    if 'data' in keys:
        srf = SRF['data']
    elif 'srf' in keys:
        srf = SRF['srf']
    elif 'D' in keys:
        srf = SRF['D']
    else:
        raise KeyError("Neither 'data' nor 'srf' key found in the provided SRF file.")

    # 检查srf是否为float32的numpy数组，否则转换
    if not isinstance(srf, np.ndarray) or srf.dtype != np.float32:
        srf = np.array(srf, dtype=np.float32) 
    # 获取路径倒数第二个目录的名称作为srf_name
    srf_name = os.path.basename(os.path.dirname(predefined_SRF))
    
    return srf, srf_name


def spatial_blur(psf,Z,scale=8):
    # Z: 输入的高光谱图像[N,C,H,W]
    # psf: 卷积核
    # scale: 下采样倍数
    if isinstance(Z, torch.Tensor):
        Z = Z.detach().cpu().numpy()  # 如果是 PyTorch 张量，转换为 NumPy 数组

    # 对每个样本和每个通道进行卷积操作
    N, C, H, W = Z.shape
    X = np.zeros((N, C, H // scale, W // scale))

    # 创建卷积层（只创建一次，避免在循环内创建）
    padding=(psf.shape[0] +1 - scale) // 2
    if padding < 0:
        padding = 0
    conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=psf.shape[0], stride=scale, padding=padding, bias=False)
    conv.weight.data = torch.from_numpy(psf).unsqueeze(0).unsqueeze(0)  # 设置卷积核
    conv.weight.requires_grad = False  # 不需要训练卷积核

    # 将卷积层转移到设备（如 GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conv.to(device)

    # 对每个样本进行卷积操作
    for i in range(N):
        for c in range(C):
            Z_tensor = torch.from_numpy(Z[i, c]).unsqueeze(0).unsqueeze(0).float().to(device)  # 转换为4D张量并转移到设备
            X_tensor = conv(Z_tensor)  # 执行卷积操作
            X[i, c] = X_tensor.squeeze(0).squeeze(0).cpu().numpy()  # 将结果转换回 numpy 并移回 CPU

    return X

def spectral_blur(srf,Z):

    if isinstance(Z, torch.Tensor):
       Z = Z.detach().cpu().numpy()  # 如果是 PyTorch 张量，转换为 NumPy 数组

    Y= np.zeros((Z.shape[0], 3, Z.shape[2], Z.shape[3]))    #这里通道数设置
    #对Z进行光谱降采样
    for i in range(Z.shape[0]):  # 对每一张图像
        for j in range(Z.shape[2]):
            for k in range(Z.shape[3]): # 对每个像素的光谱
                Y[i, :, j, k] = np.dot(srf.T, Z[i, :, j, k])  #这里srf是否转置
    
    return Y



if __name__ == "__main__":

    psf, psf_name = getPSF(kernel_size=11, scale=8, sigma=2.0, Gaussian = True)  # 示例参数
    srf, srf_name = getSRF('data/predefined_SRF/Nikon/Nikon_srf.mat')  # 示例参数
    # k1 = '/mnt/disk3/dyx/data/predefined_blur_kernels/Levins/k1/k1.mat'
    # 'data/predefined_SRF/landsat/landsat_srf.mat'
    # 'data/predefined_SRF/worldview2/srf_houston18_worldview2.mat'
    # 'data/predefined_SRF/Nikon/Nikon_srf.mat'

    dataset = 'Harvard'
    scale = 8
    key = 'hsi'
    

    # path = os.path.join('/mnt/disk3/dyx/data', dataset, 'gt')
    path = 'data/Harvard/gt'

    # if dataset == 'CAVE':
    #     getAndSaveCAVE(path=path, dataset='CAVE')

    # elif dataset == 'Harvard':
    #     crop_Harvard(input_dir='/mnt/disk3/dyx/data/Harvard/oriHSI', output_dir=path, save_data_key=key)


    degradation_name = f"scale{scale}+{psf_name}+{srf_name}"
    degradation_dir = os.path.join('data', dataset, degradation_name)
    # 检查degradation_dir是否存在，如果不存在则创建
    if not os.path.exists(degradation_dir):
        os.makedirs(degradation_dir)

    # 获取所有gt .mat文件
    mat_files = [f for f in os.listdir(path) if f.endswith('.mat')]
    for mat_file in mat_files:
        mat_path = os.path.join(path, mat_file)
        # 读取.mat文件
        data = sio.loadmat(mat_path)
        hsi = data[key]  
        hsi = np.transpose(hsi, (2, 0, 1))  # 转换为 (C, H, W) 形状

        # 对高光谱图像进行空间降采样
        hsi = np.array(hsi, dtype=np.float32)
        Z = spatial_blur(psf, hsi[np.newaxis, ...], scale=scale)  # 添加一个批次维度

        Z = torch.from_numpy(Z).float()  # 转换为 PyTorch 张量并保持 float32 类型
        Z = Z.squeeze(0)
        # 转置为HWC格式
        Z = Z.permute(1, 2, 0)

        # 对高光谱图像进行光谱降采样
        Y = spectral_blur(srf, hsi[np.newaxis, ...])  # 转换为 NumPy 数组
        Y = torch.from_numpy(Y).float()  # 转换为 PyTorch 张量并保持 float32 类型
        Y = Y.squeeze(0)
        # 转置为HWC格式
        Y = Y.permute(1, 2, 0)

        # 在degradation_dir下分别新建LrHSI和HrMSI子目录，保存降采样后的mat图像Z和Y，键为data，名称与原图像名称一致
        lr_hsi_path = os.path.join(degradation_dir, 'LrHSI', mat_file)
        hr_msi_path = os.path.join(degradation_dir, 'HrMSI', mat_file)
        os.makedirs(os.path.dirname(lr_hsi_path), exist_ok=True)
        os.makedirs(os.path.dirname(hr_msi_path), exist_ok=True)
        # 保存降采样后的图像
        sio.savemat(lr_hsi_path, {'data': Z.numpy()})
        sio.savemat(hr_msi_path, {'data': Y.numpy()})
    print(f"降采样后的图像已保存到 {degradation_dir} 目录下。")


