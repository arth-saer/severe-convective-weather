import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.getcwd())

class DropPath(nn.Module):
    
    def __init__(self, drop_prob=0.1):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        
        keep_prob = 1 - self.drop_prob
        
        shape = (x.shape[0], 1, 1)
        random_tensor = torch.rand(shape, device=x.device)
        
        binary_mask = (random_tensor < keep_prob).float()
        
        output = x / keep_prob * binary_mask
        
        return output

class MetaFormerBlock(nn.Module):

    def __init__(self, input_channel, mlp_ratio=8., drop=0., drop_path=0.):
        super(MetaFormerBlock, self).__init__()
        
        # Token Mixer
        self.token_mixer = nn.Sequential(
            nn.LayerNorm(input_channel),
            nn.Linear(input_channel, input_channel),
            nn.GELU(),
            nn.Dropout(drop)
        )
        
        # Channel Mixer
        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(input_channel),
            nn.Linear(input_channel, int(input_channel * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(input_channel * mlp_ratio), input_channel),
            nn.Dropout(drop)
        )
        
        self.drop_path = DropPath(drop_path)
    
    def forward(self, x):
        # x: [B*H*W, T, C]
        x = x + self.drop_path(self.token_mixer(x))
        
        x = x + self.drop_path(self.channel_mixer(x))
        
        return x

class Encoder(nn.Module):
    
    def __init__(self, input_channel, hidden_channels=[64, 128, 256, 512], kernel_sizes=[7, 5, 3, 3]):
        super(Encoder, self).__init__()
        
        layers = []
        cur_channel = input_channel
        
        for i, (hidden_channel, kernel_size) in enumerate(zip(hidden_channels, kernel_sizes)):
            padding = kernel_size // 2
            layers.extend([
                nn.Conv2d(cur_channel, hidden_channel, kernel_size, stride=2, padding=padding),
                nn.BatchNorm2d(hidden_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channel, hidden_channel, 3, stride=1, padding=1),
                nn.BatchNorm2d(hidden_channel),
                nn.ReLU(inplace=True)
            ])
            cur_channel = hidden_channel
        
        self.encoder = nn.Sequential(*layers)
        self.output_channel = hidden_channels[-1]
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        
        encoded = self.encoder(x)
        
        _, C_enc, H_enc, W_enc = encoded.shape
        output = encoded.reshape(B, T, C_enc, H_enc, W_enc)
        
        return output

class Translator(nn.Module):
    
    def __init__(self, T, input_channel, hidden_channel=512, num_blocks=4, mlp_ratio=8.):
        super(Translator, self).__init__()
        
        self.T = T
        self.input_dim = input_channel
        
        self.input_proj = nn.Linear(input_channel, hidden_channel)
        
        # MetaFormer blocks
        self.blocks = nn.ModuleList([
            MetaFormerBlock(hidden_channel, mlp_ratio) for _ in range(num_blocks)
        ])
        
        self.output_proj = nn.Linear(hidden_channel, input_channel)
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.normal_(self.output_proj.weight, std=0.02)
    
    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        
        x = x.view(B, T, C, H * W)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(B * H * W, T, C)
        
        x = self.input_proj(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.output_proj(x)
        
        x = x.reshape(B, H * W, T, C)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(B, T, C, H, W)
        
        return x

class Decoder(nn.Module):
    
    def __init__(self, input_channel, hidden_channels=[256, 128, 64, 32], output_channel=1):
        super(Decoder, self).__init__()
        
        layers = []
        cur_channel = input_channel
        
        for i, hidden_channel in enumerate(hidden_channels):
            layers.extend([
                nn.Conv2d(cur_channel, hidden_channel, 3, padding=1),
                nn.BatchNorm2d(hidden_channel),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(hidden_channel, hidden_channel, 3, padding=1),
                nn.BatchNorm2d(hidden_channel),
                nn.ReLU(inplace=True)
            ])
            cur_channel = hidden_channel
        
        layers.append(nn.Conv2d(cur_channel, output_channel, 1))
        
        self.decoder = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x: [B, C, H, W]
        return self.decoder(x)

class SimVP(nn.Module):

    
    def __init__(self, T=6, input_channel=1, output_channel=1, 
                 encoder_hidden_channels=[64, 128, 256, 512], encoder_kernel_sizes=[7, 5, 3, 3],
                 translator_hidden_channel=512, num_blocks=4, mlp_ratio=8.,
                 decoder_hidden_channels=[256, 128, 64, 32]):
        super(SimVP, self).__init__()
        
        # 编码器
        self.encoder = Encoder(input_channel, encoder_hidden_channels, encoder_kernel_sizes)
        encoder_output_channel = encoder_hidden_channels[-1]
        
        # 翻译器
        self.translator = Translator(
            T=T,
            input_channel=encoder_output_channel,
            hidden_channel=translator_hidden_channel,
            num_blocks=num_blocks,
            mlp_ratio=mlp_ratio
        )
        
        # 解码器
        self.decoder = Decoder(
            input_channel=encoder_output_channel,
            hidden_channels=decoder_hidden_channels,
            output_channel=output_channel
        )
    
    
    
    def forward(self, x):
        
        # 编码器  [B, 6, 1, H, W] → [B, 6, 512, H/16, W/16]
        encoded_features = self.encoder(x)
        
        # 翻译器 [B, 6, 512, H/16, W/16] → [B, 6, 512, H/16, W/16]
        temporal_features = self.translator(encoded_features)
        
        # 取第6帧 [B, 6, 512, H/16, W/16] → [B, 512, H/16, W/16]
        last_frame_feature = temporal_features[:, -1]
        
        # 解码器 [B, 512, H/16, W/16] → [B, 1, H, W]
        seg_mask = self.decoder(last_frame_feature)
        
        return seg_mask

if __name__ == "__main__":
    model = SimVP()
    
    x = torch.randn(8, 6, 1, 800, 1280)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"输入尺寸: {x.shape}")
    print(f"输出尺寸: {output.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")
    
    # 测试输出范围 (应该在0-1之间)
    print(f"输出范围: [{output.min():.3f}, {output.max():.3f}]")
    
    # 打印模型结构
    print("\n模型结构:")
    print(model)