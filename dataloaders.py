from torch.utils.data import DataLoader
from utils.datasets import HSI_MSI_Data, real_Data

def get_dataloaders_dataparallel(split, kernel_path=None, batch_size=32, num_workers=4, data_path='./data', datarange=None, psf=None, srf=None, scale_factor=8, dataset='CAVE', noise=0, normalize=False):
    """
    返回适用于DataParallel多卡训练的DataLoader非分布式
    """
    if dataset != 'liao':
            
        # 1. 加载训练集（无需分布式采样器）
        train_dataset = HSI_MSI_Data(
            mode='train', 
            split=split,
            noise=noise,
            data_path=data_path, 
            psf=psf, 
            srf=srf, 
            dataset=dataset, 
            datarange=datarange,
            aug=False,  # 可按需开启数据增强
            scale_factor=scale_factor,
            normalize=normalize,
            kernel_path=kernel_path
        )

        # 2. 加载测试集
        test_dataset = HSI_MSI_Data(
            mode='test', 
            split=split,
            noise=noise,
            data_path=data_path, 
            psf=psf, 
            srf=srf, 
            dataset=dataset, 
            datarange=datarange,
            aug=False,  # 测试集不增强
            scale_factor=scale_factor,
            normalize=normalize,
            kernel_path=kernel_path
        )
    else:
        # 1. 加载训练集（无需分布式采样器）
        train_dataset = real_Data(
            mode='train', 
            split=split,
            data_path=data_path, 
            dataset=dataset, 
            aug=False,  # 可按需开启数据增强
            normalize=normalize
        )

        # 2. 加载测试集
        test_dataset = real_Data(
            mode='test', 
            split=split,
            data_path=data_path, 
            dataset=dataset, 
            aug=False,  # 测试集不增强
            normalize=normalize
        )

    # 3. 创建DataLoader（训练集启用shuffle）
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,  # DataParallel用普通shuffle即可
        num_workers=num_workers, 
        pin_memory=True  # 加速GPU数据传输
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False,  # 测试集不shuffle
        num_workers=num_workers, 
        pin_memory=True
    )

    return train_loader, test_loader
