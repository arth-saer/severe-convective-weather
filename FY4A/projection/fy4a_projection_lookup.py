import numpy as np
import os
from typing import Union, Tuple, List

#常量定义
R = 6371.0 # 地球半径 [km]

# 不同分辨率对应的参数
RESOLUTION_PARAMS = {
    500: {
        'line_size': 21984,
        'column_size': 21984
    },
    1000: {
        'line_size': 10992,
        'column_size': 10992
    },
    2000: {
        'line_size': 5496,
        'column_size': 5496
    },
    4000: {
        'line_size': 2748,
        'column_size': 2748
    }
}

def get_resolution_params(resolution: int) -> dict:
    """获取分辨率参数"""
    if resolution not in RESOLUTION_PARAMS:
        raise ValueError(f"不支持的分辨率: {resolution}")
    return RESOLUTION_PARAMS[resolution]


def load_grid_data(grid_file_path: str, resolution: int) -> Tuple[np.ndarray, np.ndarray]:
    """加载网格数据"""
    if not os.path.exists(grid_file_path):
        raise FileNotFoundError(f"网格文件不存在: {grid_file_path}")
    
    params = get_resolution_params(resolution)
    line_size = params['line_size']
    column_size = params['column_size']
    
    with open(grid_file_path, 'rb') as f:
        raw_data = f.read()
    
    # 验证文件大小
    expected_size = line_size * column_size * 16
    if len(raw_data) != expected_size:
        raise ValueError(f"文件大小不匹配。预期: {expected_size}字节，实际: {len(raw_data)}字节")
    
    data = np.frombuffer(raw_data, dtype='<f8')
    data = data.reshape(line_size, column_size, 2)
    
    lats_grid = data[:, :, 0]  # 纬度网格    
    lons_grid = data[:, :, 1]  # 经度网格

    return lons_grid, lats_grid

def latlon_to_linecolumn_lookup(lat: Union[float, List[float], np.ndarray], 
                               lon: Union[float, List[float], np.ndarray], 
                               grid_file_path: str = '/mnt/md1/lxw/severe-convective-weather/FY4A/projection_table/table/FullMask_Grid_4000.raw',
                               resolution: int = 4000) -> Tuple[np.ndarray, np.ndarray]:
    """
    查找法：将地理经纬度转换为标称投影的行列号
    
    Parameters:
    -----------
    lat : 纬度或纬度数组 [度]
    lon : 经度或经度数组 [度]
    grid_file_path : 网格文件路径
    resolution : 分辨率
            
    Returns:
    --------
    tuple : (行号数组, 列号数组)
    """
    # 确保输入为numpy数组
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    
    if len(lat) != len(lon):
        raise ValueError("纬度和经度数组长度不匹配")
    
    # 加载网格数据
    lons_grid, lats_grid = load_grid_data(grid_file_path, resolution)
    
    line = np.zeros(len(lat), dtype=np.int32)
    column = np.zeros(len(lat), dtype=np.int32)
    
    # 计算网格的弧度坐标
    lons_rad = np.radians(lons_grid)
    lats_rad = np.radians(lats_grid)
    
    for i in range(len(lat)):
        # 将目标点转换为弧度
        lon_rad = np.radians(lon[i])
        lat_rad = np.radians(lat[i])
        
        # 计算球面距离（Haversine公式）
        dlon = lons_rad - lon_rad
        dlat = lats_rad - lat_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat_rad) * np.cos(lats_rad) * np.sin(dlon/2)**2
        distances = 2 * R * np.arcsin(np.sqrt(a))
        
        # 找到最小距离的索引
        min_idx = np.argmin(distances)
        row, col = np.unravel_index(min_idx, distances.shape)
        
        line[i], column[i] = row, col
        
    return line, column
    
    # line = np.zeros(len(lat), dtype=np.int32)
    # column = np.zeros(len(lat), dtype=np.int32)
        
    # for i in range(len(lat)):
    #     # 查找最接近的纬度行
    #     lat_diff = np.abs(lats_grid - lat[i])
    #     min_idx = np.argmin(lat_diff)
    #     row, _ = np.unravel_index(min_idx, lats_grid.shape)
        
    #     # 在找到的行中查找最接近的经度列
    #     lon_line = lons_grid[row, :]
    #     col = np.argmin(np.abs(lon_line - lon[i]))
        
    #     line[i], column[i] = row, col
    
    # return line, column

def linecolumn_to_latlon_lookup(line: Union[int, List[int], np.ndarray], 
                               column: Union[int, List[int], np.ndarray], 
                               grid_file_path: str = '/mnt/md1/lxw/severe-convective-weather/FY4A/projection_table/table/FullMask_Grid_4000.raw', 
                               resolution: int = 4000) -> Tuple[np.ndarray, np.ndarray]:
    """
    查找法：将标称投影的行列号转换为地理经纬度
    
    Parameters:
    -----------
    line : 行号或行号数组（整数像素坐标）
    column : 列号或列号数组（整数像素坐标）
    grid_file_path : 网格文件路径
    resolution : 分辨率
            
    Returns:
    --------
    tuple : (纬度数组, 经度数组) [度]
    """
    # 确保输入为numpy数组
    line = np.asarray(line, dtype=np.int32)
    column = np.asarray(column, dtype=np.int32)
    
    if len(line) != len(column):
        raise ValueError("行号和列号数组长度不匹配")
    
    # 加载网格数据
    lons_grid, lats_grid = load_grid_data(grid_file_path, resolution)
    
    lat = lats_grid[line, column]
    lon = lons_grid[line, column]
    
    return lat, lon

# 便捷的单点转换函数
def latlon_to_linecolumn_lookup_single(lat: float, lon: float, grid_file_path: str, resolution: int = 4000) -> Tuple[int, int]:

    line, column = latlon_to_linecolumn_lookup([lat], [lon], grid_file_path, resolution)
    return int(line[0]), int(column[0])

def linecolumn_to_latlon_lookup_single(line: int, column: int, grid_file_path: str, resolution: int = 4000) -> Tuple[float, float]:
    
    lat, lon = linecolumn_to_latlon_lookup([line], [column], grid_file_path, resolution)
    return float(lat[0]), float(lon[0])


if __name__ == "__main__":

    
    grid_file = '/mnt/md1/lxw/severe-convective-weather/FY4A/projection_table/table/FullMask_Grid_4000.raw' # 替换为实际文件路径
    
    lat, lon = 30.0, 110.0
    line, column = latlon_to_linecolumn_lookup_single(lat, lon, grid_file, 4000)
    print(f"查找法 - 经纬度 ({lat}, {lon}) -> 行列号 ({line}, {column})")
        
    lat_back, lon_back = linecolumn_to_latlon_lookup_single(line, column, grid_file, 4000)
    print(f"查找法 - 行列号 ({line}, {column}) -> 经纬度 ({lat_back:.6f}, {lon_back:.6f})")
        
    # 测试批量转换
    lats = [30.0, 31.0, 32.0]
    lons = [110.0, 111.0, 112.0]
    lines, columns = latlon_to_linecolumn_lookup(lats, lons, grid_file, 4000)
    print(f"查找法批量转换 - 行号: {lines}, 列号: {columns}")