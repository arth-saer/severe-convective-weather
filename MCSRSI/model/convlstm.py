import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.getcwd())

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias
        )
    
    def forward(self, input, state):
        
        h_prev, c_prev = state
        combined = torch.cat([input, h_prev], dim=1)
        combined_conv = self.conv(combined)
        
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_cur = f * c_prev + i * g
        h_cur = o * torch.tanh(c_cur)
        
        return h_cur, c_cur
    
    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width),
                torch.zeros(batch_size, self.hidden_dim, height, width))

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernal_size, bias=True, num_layers=1):
        super(ConvLSTM, self).__init__()
        
        self.num_layers = num_layers
        
        cell_list = []
        for i in range(num_layers):
            cell_list.append(ConvLSTMCell(
                input_dim=input_dim if i == 0 else hidden_dim,
                hidden_dim=hidden_dim,
                kernel_size=kernal_size[i],
                bias=bias
            ))
        self.cell_list = nn.ModuleList(cell_list)
        
    def forward(self, input, hidden_state=None):
        
        batch_size, seq_len, _, height, width = input.shape
        
        # if hidden_state is None:
        #     hidden_state = [self.cell_list[i].init_hidden(batch_size, (height, width)) 
        #                     for i in range(self.num_layers)]
        
        device = input.device
        dtype = input.dtype
        if hidden_state is None:
            hidden_state = [
                (h.to(device=device, dtype=dtype), c.to(device=device, dtype=dtype)) 
                for h, c in [
                    self.cell_list[i].init_hidden(batch_size, (height, width)) 
                    for i in range(self.num_layers)
                ]
            ]
        
        cur_input = input
        layer_outputs = []
        layer_final_states = []
        
        for layer_idx, cell in enumerate(self.cell_list):
            h, c = hidden_state[layer_idx]
            layer_outputs_timesteps = []
            
            for t in range(seq_len):
                h, c = cell(
                    input=cur_input[:, t, :, :, :],
                    state=(h, c)
                )            
                layer_outputs_timesteps.append(h)
                
            layer_output = torch.stack(layer_outputs_timesteps, dim=1)
            layer_outputs.append(layer_output)
            layer_final_states.append((h, c))
                
            cur_input = layer_output
                
        return layer_final_states[-1][0] # 最后一层最后一个时间步的输出h


class ConvLSTMCCD(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=128, output_channels=1,
                 kernel_size=[5, 3, 3],bias=True,
                 num_layers=3):
        super(ConvLSTMCCD, self).__init__()     
        
        self.convlstm = ConvLSTM(
            input_dim=input_channels,
            hidden_dim=hidden_channels,
            kernal_size=kernel_size,
            bias=bias,
            num_layers=num_layers
        )
        # self.output = nn.Sequential(
        #     nn.Conv2d(hidden_channels, 32, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 16, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(16, output_channels, kernel_size=1)
        # )
        
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
        
        return self.output_conv(self.convlstm(x))
    
    
if __name__ == "__main__":
    model = ConvLSTMCCD()
    
    x = torch.randn(8, 6, 1, 800, 1280)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"输入尺寸: {x.shape}")
    print(f"输出尺寸: {output.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")