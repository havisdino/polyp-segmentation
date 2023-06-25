import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x
    

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels,
                                         kernel_size, stride, padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.deconv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout=None):
        super().__init__()
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')
        self.conv_block_1 = ConvBlock(in_channels, out_channels, kernel_size, padding='same')
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding='same')
        self.norm_2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2)
        if dropout:
            self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        skip = self.res_conv(x)
        x = self.conv_block_1(x)
        x = self.conv_2(x) + skip
        x = self.norm_2(x)
        x = self.relu(x)
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        return x
    

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout=None):
        super().__init__()
        self.res_block = ResidualBlock(in_channels, out_channels, kernel_size, dropout)
        self.maxpool = nn.MaxPool2d(2)
    
    def forward(self, x):
        x = self.maxpool(x)
        x = self.res_block(x)
        return x
    
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout=None):
        super().__init__()
        self.res_block = ResidualBlock(in_channels, out_channels, kernel_size, dropout)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
    
    def forward(self, x):        
        x = self.res_block(x)
        x = self.upsample(x)
        return x
    
    
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.downsample = nn.ModuleList([
            ResidualBlock(3, 64, 3, 0.1),
            DownBlock(64, 128, 3, 0.1),
            DownBlock(128, 256, 3, 0.1),
            DownBlock(256, 512, 3, 0.1),
            DownBlock(512, 1024, 3, 0.1),
        ])
        
        self.upsample = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='bilinear'),
            UpBlock(1024 + 512, 512, 3, 0.1),
            UpBlock(512 + 256, 256, 3, 0.1),
            UpBlock(256 + 128, 128, 3, 0.1),
        ])
        
        self.last_blocks = nn.ModuleList([
            ResidualBlock(128 + 64, 64, 3, 0.1),
            nn.Conv2d(64, 1, 3, padding='same'),
            nn.Sigmoid(),
        ])
    
    def forward(self, x):
        skips = []
        for block in self.downsample:
            x = block(x)
            skips.append(x)
        skips = reversed(skips[:-1])
        for block, skip in zip(self.upsample, skips):
            x = block(x)
            x = torch.concat([x, skip], dim=1)
        for block in self.last_blocks:
            x = block(x)
        return x.squeeze()
