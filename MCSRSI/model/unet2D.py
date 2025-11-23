
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super(UNet2D, self).__init__()
        
        self.enc1 = self._make_conv_block(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d((2, 2), stride=(2, 2))
        
        self.enc2 = self._make_conv_block(base_channels, base_channels*2)
        self.pool2 = nn.MaxPool2d((2, 2), stride=(2, 2))
        
        self.enc3 = self._make_conv_block(base_channels*2, base_channels*4)
        self.pool3 = nn.MaxPool2d((2, 2), stride=(2, 2))
        
        self.enc4 = self._make_conv_block(base_channels*4, base_channels*8)
        self.pool4 = nn.MaxPool2d((2, 2), stride=(2, 2))
        
        self.bottleneck = self._make_conv_block(base_channels*8, base_channels*16)
        
        self.up4 = nn.ConvTranspose2d(base_channels*16, base_channels*8, kernel_size=(2, 2), stride=(2, 2))
        self.dec4 = self._make_conv_block(base_channels*16, base_channels*8)
        
        self.up3 = nn.ConvTranspose2d(base_channels*8, base_channels*4, kernel_size=(2, 2), stride=(2, 2))
        self.dec3 = self._make_conv_block(base_channels*8, base_channels*4)
        
        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=(2, 2), stride=(2, 2))
        self.dec2 = self._make_conv_block(base_channels*4, base_channels*2)
        
        self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=(2, 2), stride=(2, 2))
        self.dec1 = self._make_conv_block(base_channels*2, base_channels)
        
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
        self._initialize_weights()
    
    def _make_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)      
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)                
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):                             # [B, 1, H, W]
        
        # 编码路径
        enc1 = self.enc1(x)                           # [B, 32, H, W]
        enc2 = self.enc2(self.pool1(enc1))            # [B, 64, H/2, W/2]
        enc3 = self.enc3(self.pool2(enc2))            # [B, 128, H/4, W/4]
        enc4 = self.enc4(self.pool3(enc3))            # [B, 256, H/8, W/8]
        
        # 瓶颈层
        bottleneck = self.bottleneck(self.pool4(enc4)) # [B, 512, H/16, W/16]
        
        # 解码路径 (带有跳跃连接)
        dec4 = self.up4(bottleneck)                   # [B, 256, H/8, W/8]
        dec4 = torch.cat([dec4, enc4], dim=1)         # [B, 512, H/8, W/8]
        dec4 = self.dec4(dec4)                        # [B, 256, H/8, W/8]
        
        dec3 = self.up3(dec4)                         # [B, 128, H/4, W/4]
        dec3 = torch.cat([dec3, enc3], dim=1)         # [B, 256, H/4, W/4]
        dec3 = self.dec3(dec3)                        # [B, 128, H/4, W/4]
        
        dec2 = self.up2(dec3)                         # [B, 64, H/2, W/2]
        dec2 = torch.cat([dec2, enc2], dim=1)         # [B, 128, H/2, W/2]
        dec2 = self.dec2(dec2)                        # [B, 64, H/2, W/2]
        
        dec1 = self.up1(dec2)                         # [B, 32, H, W]
        dec1 = torch.cat([dec1, enc1], dim=1)         # [B, 64, H, W]
        dec1 = self.dec1(dec1)                        # [B, 32, H, W]
        
        # 输出层
        output = self.final_conv(dec1)                # [B, 1, H, W]
        return output


if __name__ == "__main__":
    model = UNet2D()
    
    x = torch.randn(8, 1, 800, 1280)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"输入尺寸: {x.shape}")
    print(f"输出尺寸: {output.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")