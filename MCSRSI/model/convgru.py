import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.getcwd())

class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvGRUCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        
        self.conv_gates = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=2 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias
        )
        
        self.conv_candidate = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias
        )
    
    def forward(self, input, h_prev):
        
        # input: 当前时间步的输入 (B, input_dim, H, W)
        # h_prev: 前一个时间步的隐藏状态 (B, hidden_dim, H, W)

        combined = torch.cat([input, h_prev], dim=1)
        
        gates = self.conv_gates(combined)
        r, z = torch.split(gates, self.hidden_dim, dim=1)
        r = torch.sigmoid(r)
        z = torch.sigmoid(z)
        
        combined_reset = torch.cat([input, r * h_prev], dim=1)
        n = torch.tanh(self.conv_candidate(combined_reset))
        
        h_cur = (1 - z) * n + z * h_prev
        
        return h_cur
    
    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return torch.zeros(batch_size, self.hidden_dim, height, width)

class ConvGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True, num_layers=1):
        super(ConvGRU, self).__init__()
        
        self.num_layers = num_layers
        
        cell_list = []
        for i in range(num_layers):
            cell_list.append(ConvGRUCell(
                input_dim=input_dim if i == 0 else hidden_dim,
                hidden_dim=hidden_dim,
                kernel_size=kernel_size[i],
                bias=bias
            ))
        self.cell_list = nn.ModuleList(cell_list)
        
    def forward(self, input, hidden_state=None):

        # input: 输入序列 (B, T, C, H, W)
        batch_size, seq_len, _, height, width = input.shape
        
        device = input.device
        dtype = input.dtype
        if hidden_state is None:
            hidden_state = [
                h.to(device=device, dtype=dtype) 
                for h in [
                    self.cell_list[i].init_hidden(batch_size, (height, width)) 
                    for i in range(self.num_layers)
                ]
            ]
        
        cur_input = input
        layer_final_states = []
        
        for layer_idx, cell in enumerate(self.cell_list):
            h = hidden_state[layer_idx]
            layer_outputs_timesteps = []
            
            for t in range(seq_len):
                h = cell(
                    input=cur_input[:, t, :, :, :],
                    h_prev=h
                )            
                layer_outputs_timesteps.append(h)
            
            layer_output = torch.stack(layer_outputs_timesteps, dim=1)
            layer_final_states.append(h)
            
            cur_input = layer_output
                
        return layer_final_states[-1]

class ConvGRUCCD(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=128, output_channels=1,
                 kernel_size=[5, 3, 3], bias=True, num_layers=3):
        super(ConvGRUCCD, self).__init__()     
        
        self.convgru = ConvGRU(
            input_dim=input_channels,
            hidden_dim=hidden_channels,
            kernel_size=kernel_size,
            bias=bias,
            num_layers=num_layers
        )
        
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, output_channels, kernel_size=1)
        )
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

        return self.output_conv(self.convgru(x))

if __name__ == "__main__":

    model = ConvGRUCCD()
    
    x = torch.randn(8, 6, 1, 800, 1280)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"输入尺寸: {x.shape}")
    print(f"输出尺寸: {output.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")
    