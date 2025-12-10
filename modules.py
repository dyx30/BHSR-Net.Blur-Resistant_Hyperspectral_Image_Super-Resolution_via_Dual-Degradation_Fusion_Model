import torch
import torch.nn as nn
import torch.nn.functional as F


class S1_Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(S1_Generator, self).__init__()     
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)  
        )
 
    def forward(self, L, A):
        # 输入相加
        s = L + A
        s = self.conv1(s)   
        return s

class S2_Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(S2_Generator, self).__init__()    
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)  
        )
 
    def forward(self, L, Y_):
        # 输入相加
        s = L + Y_
        s = self.conv1(s)
        return s
    
class CB_R(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CB_R, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),
                                   nn.LeakyReLU(negative_slope=0.1, inplace=True))        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))


    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)  
        # if x2.size == x.size:
        #     x2=x+x2
        return x2

class CB_B(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CB_B, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))


    def forward(self, x):
        x1 = self.conv1(x)  # 包含BN层
        # if x1.size == x.size:
        #     x1=x+x1
        return x1
    
class GX(nn.Module):
    def __init__(self, C, c, sigma1):
        super(GX, self).__init__()
        self.CB_R_down = CB_R(C, c)
        self.CB_R_up = CB_R(c, C)
        self.CB_B1 = CB_B(C, C)
        #self.CB_B2 = CB_B(C, C)

        self.CB_B_inv = nn.Sequential(
            nn.ConvTranspose2d(C, C, kernel_size=3, stride=1, padding=1, output_padding=0, bias=False),
            nn.LeakyReLU(negative_slope=0.3, inplace=True)
        )
        self.S1generator = S1_Generator(C, C)
        self.sigma1 = sigma1

    def forward(self, x, L1, Y, Z):
        """
        L1:[B,r,H,W]
        Y:[B,c,H,W]
        Z:[B,C,H,W]
        """

        # 中间卷积计算
        x2 = self.CB_R_down(x)
        x2 = x2 - Y
        x2 = self.CB_R_up(x2) 

        x3 = self.CB_B1(x)
        x3 = x3 - Z
        # x3 = self.CB_B2(x3)

        # x3 = self.CB_B1(x)
        # x3 = x3 - Z 
        x3 = self.CB_B_inv(x3)   #这里发现使用转置卷积效果更好

        x4 = x2 + x3    # [B, C, H, W]

        S = self.S1generator(L1/self.sigma1, x)  

        # 最终外部加
        output = x4 + L1 + self.sigma1 * x - self.sigma1 * S

        return output


class L1Updater(nn.Module):
    def __init__(self, C, sigma1):
        super(L1Updater, self).__init__()
        self.S1generator = S1_Generator(C, C)
        self.sigma1 = sigma1

    def forward(self, L1, X):
        """
        X:[B,C,H,W]
        L1:[B,C,H,W]
        """
        S = self.S1generator(L1/self.sigma1, X)  
        delta_L1 = self.sigma1 * (X - S)

        return delta_L1

class L2Updater(nn.Module):
    def __init__(self, c, sigma2):
        super(L2Updater, self).__init__()
        self.S2generator = S2_Generator(c, c)
        self.sigma2 = sigma2

    def forward(self, L2, Y_):
        """
        A:[B,r,H,W]
        L1:[B,r,H,W]
        """
        S = self.S2generator(L2/self.sigma2, Y_)  
        delta_L2 = self.sigma2 * (Y_ - S)

        return delta_L2


class XsolverUnit(nn.Module):
    def __init__(self, in_channel,out_channel):
        super(XsolverUnit, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel,12,3,1,1),
            nn.LeakyReLU(0.1, inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv2d(12,out_channel,3,1,1),
            nn.LeakyReLU(0.1, inplace=True))
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channel, 12, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        """输入: [B, r, H, W]"""
        x1 = self.conv1(x)
        # x_temp = self.conv3(x)
        x2 = self.conv2(x1)

        return x2 + x
        
        # return self.conv(x) + x 
        # return self.conv(x)  


class XSolver(nn.Module):
    def __init__(self, C):
        """
            C: X输入的通道数
        """
        super(XSolver, self).__init__()
        
        # 处理GX的Unit (B,C,H,W) -> (B,C,H,W)
        self.solver = XsolverUnit(
            in_channel=C,
            out_channel=C
        )

    def forward(self, GX):
        # 基础特征处理
        delta_X = self.solver(GX) 

        return delta_X

class YSolver(nn.Module):
    def __init__(self, c, C, sigma2):
        """
            c: Y输入的通道数
        """
        super(YSolver, self).__init__()
        self.convInv = nn.Sequential(
            nn.Conv2d(c,c,3,1,1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.convR = nn.Sequential(nn.Conv2d(C, c, kernel_size=1, bias=False),
                                   nn.LeakyReLU(negative_slope=0.1, inplace=True)) 
        self.convA = nn.Sequential(nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
                                   nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.S2generator = S2_Generator(c, c)
        self.sigma2 = sigma2

    def forward(self, Y, Y_, X, L2):
        y1 = self.convR(X) + self.convA(Y) + self.sigma2*self.S2generator(L2/self.sigma2, Y_) - L2
        delta_Y = self.convInv(y1)
        return delta_Y

