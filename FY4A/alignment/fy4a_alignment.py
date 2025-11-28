import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Tuple
from projection.fy4a_projection_calculator import latlon_to_linecolumn_calc

START_LINE = 190
START_COLUMN = 750


def fy4a_alignment(fy4a_data: np.ndarray, 
                    lat_range: Tuple[float, float], 
                    lon_range: Tuple[float, float],
                    resolution: float = 0.01) -> np.ndarray:
    """
    将FY-4A数据对齐到经纬度网格
    
    Parameters:
    -----------
    fy4a_data : FY-4A数据，形状为(C, H, W)
    lat_range : 纬度范围 (min, max)
    lon_range : 经度范围 (min, max) 
    resolution : 经纬度分辨率
    
    Returns:
    --------
    aligned_data : 对齐后的数据，形状为(C, 纬度点数, 经度点数)
    """
    
    # 解析经纬度范围
    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range
    
    # 创建经纬度网格
    lat_points = np.arange(lat_max, lat_min, -resolution)
    lon_points = np.arange(lon_min, lon_max, resolution)  # 经度点
    
    n_lats = len(lat_points)  # 纬度方向点数
    n_lons = len(lon_points)  # 经度方向点数
    
    # 创建输出数组
    n_channels = fy4a_data.shape[0]
    aligned_data = np.zeros((n_channels, n_lats, n_lons), dtype=fy4a_data.dtype)

    for i, lat in enumerate(lat_points):
        for j, lon in enumerate(lon_points): 
              
            # 将经纬度转换为FY-4A行列号
            line, column = latlon_to_linecolumn_calc([lat], [lon], resolution=4000)
            line, column = int(line[0]) - START_LINE, int(column[0]) - START_COLUMN
            
            if (0 <= line < fy4a_data.shape[1] and 0 <= column < fy4a_data.shape[2]):
                aligned_data[:, i, j] = fy4a_data[:, line, column]
    
    return aligned_data


if __name__ == '__main__':
    fy4a_data = np.load('/mnt/md1/lxw/severe-convective-weather/FY4A/Data/20180713100000/raw_data/20180713100000.npy')
    aligned_data = fy4a_alignment(fy4a_data=fy4a_data, 
                                  lat_range=(19.04, 26.04), 
                                  lon_range=(108.5, 117.5),
                                  resolution=0.01)
    print(fy4a_data.shape)
    print(aligned_data.shape)
    np.save('/mnt/md1/lxw/severe-convective-weather/FY4A/Data/20180713100000/align_data/20180713100000_align.npy', aligned_data)
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    for i in range(14):
        # 计算当前通道的统一颜色范围
        vmin = min(np.nanmin(fy4a_data[i]), np.nanmin(aligned_data[i]))
        vmax = max(np.nanmax(fy4a_data[i]), np.nanmax(aligned_data[i]))
    
        # 原始数据
        plt.imshow(fy4a_data[i], cmap='jet', vmin=vmin, vmax=vmax)
        
        # 添加矩形框 (502<=row<=674, 717<=col<=953)
        rect = patches.Rectangle(
            (717, 502),           # 左上角坐标 (x=col, y=row)
            953 - 717,            # 宽度 (col范围)
            674 - 502,            # 高度 (row范围)
            linewidth=2,          # 线宽
            edgecolor='red',      # 边框颜色
            facecolor='none',     # 无填充
            linestyle='--'        # 虚线样式
        )
        plt.gca().add_patch(rect)
        
        plt.colorbar()
        plt.title(f'RAW_Channel{i + 1}')
        plt.savefig(os.path.join('/mnt/md1/lxw/severe-convective-weather/FY4A/Data/20180713100000/raw_data',
                             f'RAW_Channel{i+1}.png'))
        plt.close()
    
        # 对齐数据（使用相同的vmin/vmax）
        plt.imshow(aligned_data[i], cmap='jet', vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.title(f'ALIGNED_Channel{i + 1}')
        plt.savefig(os.path.join('/mnt/md1/lxw/severe-convective-weather/FY4A/Data/20180713100000/align_data',
                                f'ALIGNED_Channel{i+1}.png'))
        plt.close()
        