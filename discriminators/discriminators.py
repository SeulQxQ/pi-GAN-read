"""Discrimators used in pi-GAN"""

import math
import torch
import torch.nn as nn
import curriculums
import torch.nn.functional as F
import logging

from discriminators.sgdiscriminators import *

class GlobalAveragePooling(nn.Module):  # 全局平均池化
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.mean([2, 3])
 
class AdapterBlock(nn.Module):  # 用于迁移学习和参数效率学习的特殊网络结构
    def __init__(self, output_channels):
        super().__init__()
        """
        AdapterBlock:

            output_channels: fromRGB planes(channels)
        """
        logging.info(f"AdapterBlock output_channels: {output_channels}")
        self.model = nn.Sequential(
            nn.Conv2d(3, output_channels, 1, padding=0),
            nn.LeakyReLU(0.2)
        )
    def forward(self, input):
        # AdapterBlock input.shape: [batch_size, 3, img_size, img_size]
        return self.model(input)


def kaiming_leaky_init(m):  # 激活函数使用 leaky_relu 初始化
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

# 负责为给定的特征图添加坐标通道
class AddCoords(nn.Module):    
    """
    Source: https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py
    """

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):  # <-- CoordConv
        """
        Args:
            input_tensor --> x shape(batch, channel, img_size, img_size)
        """
        logging.info(f"AddCoords input_tensor.shape: {input_tensor.shape}")
        batch_size, _, x_dim, y_dim = input_tensor.size()

        # 生成x, y 坐标网格
        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)   # [1, img_size, img_size]  
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)   # [1, img_size, img_size]  

        # 归一化 [0, 1]
        xx_channel = xx_channel.float() / (x_dim - 1) 
        yy_channel = yy_channel.float() / (y_dim - 1)

        # xx_channel: shape(1, img_size, img_size)
        # 映射到 [-1, 1]
        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        # shape: (batch_size, 1, img_size, img_size)
        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)
        
        # ret.shape: [batch_size, channel + (x, y), img_size, img_size] channel --> 256
        return ret
    
# 实际的卷积层， 首先添加坐标通道，然后进行标准卷积
class CoordConv(nn.Module): # 卷积层 将坐标信息加入到输入中
    """
    Source: https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py
    """
    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__() #  inplanes --> in_channels, planes --> out_channels
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels+2 # inplanes + (x, y)  坐标信息 (x, y)
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs) # 卷积层

    def forward(self, x): # x.shape(9, 256, 32, 32) [batch_size, 256, img_size, img_size]
        logging.info(f"CoordConv x.shape: {x.shape}")
        ret = self.addcoords(x) # --> AddCoords 添加坐标信息 ret [batch_size, channels + (x, y), img_size, img_size] 
        ret = self.conv(ret)  # 渐进式增长策略 ret[batch_size, channels -> new_channels, img_size, img_size] 256+2 -> 400
        return ret

class ResidualCoordConvBlock(nn.Module): # 渐进式增长策略卷积层 inplanes planes 来自 ProgressiveDiscriminator
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, downsample=False, groups=1):
        super().__init__()
        p = kernel_size//2
        self.network = nn.Sequential(
            CoordConv(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=p), # --> CoordConv()
            nn.LeakyReLU(0.2, inplace=True),
            CoordConv(planes, planes, kernel_size=kernel_size, padding=p),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.network.apply(kaiming_leaky_init)

        self.proj = nn.Conv2d(inplanes, planes, 1) if inplanes != planes else None
        self.downsample = downsample # 下采样

    def forward(self, identity):
        # ProgressiveEncoderDiscriminator --> forward() indentity --> input [batch_size, 256, img_size, img_size]
        logging.info(f"ResidualCoordConvBlock identity.shape: {identity.shape}")
        y = self.network(identity) # --> CoordConv --> forward # y.shape [batch_size, channels(400), img_size, img_size]
        logging.info(f"ResidualCoordConvBlock y.shape: {y.shape}")
        if self.downsample: y = nn.functional.avg_pool2d(y, 2) # 下采样 y.shape [batch_size, channels(400), img_size/2, img_size/2]
        if self.downsample: identity = nn.functional.avg_pool2d(identity, 2) # x --> identity.shape [batch_size, channels(256), img_size/2, img_size/2]
        identity = identity if self.proj is None else self.proj(identity) # channels(256) -> channels(400) 保证与 y 的输出的通道数一致

        y = (y + identity)/math.sqrt(2)
        return y # y.shape [batch_size, channels(400), img_size/2, img_size/2]

# 渐进式增长判别器
class ProgressiveDiscriminator(nn.Module):
    """Implement of a progressive growing discriminator with ResidualCoordConv Blocks"""
    """ 每次增加新的分辨率级别时，会增加新的 ResidualCoordConvBlock """
    def __init__(self, **kwargs):
        super().__init__()
        self.epoch = 0
        self.step = 0
        self.layers = nn.ModuleList(
        [
            # inplane plane downsample
            ResidualCoordConvBlock(16, 32, downsample=True),   # 512x512 -> 256x256 # 每层有两层的CoordConv
            ResidualCoordConvBlock(32, 64, downsample=True),   # 256x256 -> 128x128
            ResidualCoordConvBlock(64, 128, downsample=True),  # 128x128 -> 64x64
            ResidualCoordConvBlock(128, 256, downsample=True), # 64x64   -> 32x32
            ResidualCoordConvBlock(256, 400, downsample=True), # 32x32   -> 16x16
            ResidualCoordConvBlock(400, 400, downsample=True), # 16x16   -> 8x8
            ResidualCoordConvBlock(400, 400, downsample=True), # 8x8     -> 4x4
            ResidualCoordConvBlock(400, 400, downsample=True), # 4x4     -> 2x2
        ])

        self.fromRGB = nn.ModuleList(
        [
            # output_channels
            AdapterBlock(16),
            AdapterBlock(32),
            AdapterBlock(64),
            AdapterBlock(128),
            AdapterBlock(256),
            AdapterBlock(400),
            AdapterBlock(400),
            AdapterBlock(400),
            AdapterBlock(400)
        ])
        self.final_layer = nn.Conv2d(400, 1, 2)
        self.img_size_to_layer = {2:8, 4:7, 8:6, 16:5, 32:4, 64:3, 128:2, 256:1, 512:0}


    def forward(self, input, alpha, instance_noise=0, **kwargs):
        start = self.img_size_to_layer[input.shape[-1]]
        logging.info(f"ProgressiveDiscriminator input.shape: {input.shape}")
        x = self.fromRGB[start](input)
        logging.info(f"ProgressiveDiscriminator x.shape: {x.shape}")
        for i, layer in enumerate(self.layers[start:]):
            if i == 1:
                x = alpha * x + (1 - alpha) * self.fromRGB[start+1](F.interpolate(input, scale_factor=0.5, mode='nearest')) # 改变输入数据的尺寸
            x = layer(x)

        x = self.final_layer(x).reshape(x.shape[0], 1)
        logging.info(f"ProgressiveDiscriminator x_output.shape: {x.shape}")    

        return x

class ProgressiveEncoderDiscriminator(nn.Module):  # 还有预测相机角度和潜在代码
    """
    Implement of a progressive growing discriminator with ResidualCoordConv Blocks.
    Identical to ProgressiveDiscriminator except it also predicts camera angles and latent codes.
    使用 ResidualCoordConv Blocks 实现渐进增长判别器。
    与 ProgressiveDiscriminator 相同，只是它还预测相机角度和潜在代码。
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.epoch = 0
        self.step = 0
        self.layers = nn.ModuleList(
        [
            ResidualCoordConvBlock(16, 32, downsample=True),   # 512x512 -> 256x256
            ResidualCoordConvBlock(32, 64, downsample=True),   # 256x256 -> 128x128
            ResidualCoordConvBlock(64, 128, downsample=True),  # 128x128 -> 64x64
            ResidualCoordConvBlock(128, 256, downsample=True), # 64x64   -> 32x32
            ResidualCoordConvBlock(256, 400, downsample=True), # 32x32   -> 16x16
            ResidualCoordConvBlock(400, 400, downsample=True), # 16x16   -> 8x8
            ResidualCoordConvBlock(400, 400, downsample=True), # 8x8     -> 4x4
            ResidualCoordConvBlock(400, 400, downsample=True), # 4x4     -> 2x2
        ])

        self.fromRGB = nn.ModuleList(
        [
            AdapterBlock(16),
            AdapterBlock(32),
            AdapterBlock(64),
            AdapterBlock(128),
            AdapterBlock(256),
            AdapterBlock(400),
            AdapterBlock(400),
            AdapterBlock(400),
            AdapterBlock(400)
        ])
        self.final_layer = nn.Conv2d(400, 1 + 256 + 2, 2) # kernel_size: 2
        self.img_size_to_layer = {2:8, 4:7, 8:6, 16:5, 32:4, 64:3, 128:2, 256:1, 512:0}


    def forward(self, input, alpha, instance_noise=0, **kwargs):
        if instance_noise > 0:
            input = input + torch.randn_like(input) * instance_noise
        # input.shape [batch_size, 3(rgb), img_size, img_size]
        
        start = self.img_size_to_layer[input.shape[-1]] # input.shape[-1] --> img_size
        x = self.fromRGB[start](input) # x.shape [batch_size, 256, img_size, img_size]
        
        for i, layer in enumerate(self.layers[start:]):
            if i == 1:
                x = alpha * x + (1 - alpha) * self.fromRGB[start+1](F.interpolate(input, scale_factor=0.5, mode='nearest', recompute_scale_factor=True))
            x = layer(x)

        x = self.final_layer(x).reshape(x.shape[0], -1) # x.shape [batch_size, 259] final_layer --> 1 + 256 + 2

        prediction = x[..., 0:1] # prediction.shape [batch_size, 1]
        latent = x[..., 1:257]  # latent.shape [batch_size, 256]
        position = x[..., 257:259]  # position.shape [batch_size, 2]
        logging.info(f"ProgressiveEncoderDiscriminator prediction.shape: {prediction.shape}, latent.shape: {latent.shape}, position.shape: {position.shape}")
        # 分开返回
        return prediction, latent, position
