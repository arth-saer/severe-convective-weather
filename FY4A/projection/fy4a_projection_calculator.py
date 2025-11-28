import math
import numpy as np
from typing import Union, Tuple, List

# 常量定义
PI = 3.1415926535897932384626
EA = 6378.137  # 地球半长轴 [km]
EB = 6356.7523  # 地球短半轴 [km]
H = 42164.0  # 地心到卫星质心的距离 [km]
LAMBDA_D = 104.7  # 卫星星下点所在经度 [度]

# 全圆盘经纬度范围
NORTH_LAT = 80.56672132
SOUTH_LAT = -80.56672132
EAST_LON = -174.71662309
WEST_LON = 24.11662309

# 不同分辨率对应的参数
RESOLUTION_PARAMS = {
    500: {
        'LOFF': 10991.5,
        'COFF': 10991.5,
        'LFAC': 81865099,
        'CFAC': 81865099,
        'line_size': 21984,
        'column_size': 21984
    },
    1000: {
        'LOFF': 5495.5,
        'COFF': 5495.5,
        'LFAC': 40932549,
        'CFAC': 40932549,
        'line_size': 10992,
        'column_size': 10992
    },
    2000: {
        'LOFF': 2747.5,
        'COFF': 2747.5,
        'LFAC': 20466274,
        'CFAC': 20466274,
        'line_size': 5496,
        'column_size': 5496
    },
    4000: {
        'LOFF': 1373.5,
        'COFF': 1373.5,
        'LFAC': 10233137,
        'CFAC': 10233137,
        'line_size': 2748,
        'column_size': 2748
    }
}

def get_resolution_params(resolution: int) -> dict:
    """获取分辨率参数"""
    if resolution not in RESOLUTION_PARAMS:
        raise ValueError(f"不支持的分辨率: {resolution}")
    return RESOLUTION_PARAMS[resolution]


def latlon_to_linecolumn_calc(lat: Union[float, List[float], np.ndarray], 
                            lon: Union[float, List[float], np.ndarray], 
                            resolution: int = 4000) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算法：将地理经纬度转换为标称投影的行列号
    
    Parameters:
    -----------
    lat : 纬度或纬度数组 [度]
    lon : 经度或经度数组 [度]
    resolution : 分辨率
            
    Returns:
    --------
    tuple : (行号数组, 列号数组) - 整数像素坐标
    """
    # 确保输入为numpy数组
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    
    # 获取分辨率参数
    params = get_resolution_params(resolution)
    LOFF = params['LOFF']
    COFF = params['COFF']
    LFAC = params['LFAC']
    CFAC = params['CFAC']
    
    # 将角度转换为弧度
    lon_rad = lon * PI / 180.0
    lat_rad = lat * PI / 180.0
    
    # 将地理经纬度转换为地心经纬度
    lambda_e = lon_rad
    phi_e = np.arctan((EB**2 / EA**2) * np.tan(lat_rad))
    
    # 求Re
    re = EB / np.sqrt(1 - ((EA**2 - EB**2) / EA**2) * np.cos(phi_e)**2)
    
    # 求r1, r2, r3
    lambda_d_rad = LAMBDA_D * PI / 180.0
    r1 = H - re * np.cos(phi_e) * np.cos(lambda_e - lambda_d_rad)
    r2 = -re * np.cos(phi_e) * np.sin(lambda_e - lambda_d_rad)
    r3 = re * np.sin(phi_e)
    
    # 求rn, x, y
    rn = np.sqrt(r1**2 + r2**2 + r3**2)
    x = np.arctan(-r2 / r1) * 180.0 / PI
    y = np.arcsin(-r3 / rn) * 180.0 / PI
    
    # 求浮点数行列号
    column_float = COFF + x * 2**(-16) * CFAC
    line_float = LOFF + y * 2**(-16) * LFAC
    
    # 转换为整数像素坐标
    line = np.round(line_float).astype(np.int32)
    column = np.round(column_float).astype(np.int32)
    
    return line, column

def linecolumn_to_latlon_calc(line: Union[int, List[int], np.ndarray], 
                            column: Union[int, List[int], np.ndarray], 
                            resolution: int = 4000) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算法：将标称投影的行列号转换为地理经纬度
    
    Parameters:
    -----------
    line : 行号或行号数组（整数像素坐标）
    column : 列号或列号数组（整数像素坐标）
    resolution : 分辨率
            
    Returns:
    --------
    tuple : (纬度数组, 经度数组) [度]
    """
    # 确保输入为numpy数组
    line = np.asarray(line, dtype=np.float64)
    column = np.asarray(column, dtype=np.float64)
    
    # 获取分辨率参数
    params = get_resolution_params(resolution)
    LOFF = params['LOFF']
    COFF = params['COFF']
    LFAC = params['LFAC']
    CFAC = params['CFAC']
    
    
    # 将整数行列号转换为浮点数进行计算
    line_float = line.astype(np.float64)
    column_float = column.astype(np.float64)
    
    # 求x, y
    x = (PI * (column_float - COFF)) / (180.0 * 2**(-16) * CFAC)
    y = (PI * (line_float - LOFF)) / (180.0 * 2**(-16) * LFAC)
    
    # 求sd, sn, s1, s2, s3, sxy
    term1 = (H * np.cos(x) * np.cos(y))**2
    term2 = (np.cos(y)**2 + (EA**2 / EB**2) * np.sin(y)**2) * (H**2 - EA**2)
    sd = np.sqrt(term1 - term2)
    
    denominator = np.cos(y)**2 + (EA**2 / EB**2) * np.sin(y)**2
    sn = (H * np.cos(x) * np.cos(y) - sd) / denominator
    
    s1 = H - sn * np.cos(x) * np.cos(y)
    s2 = sn * np.sin(x) * np.cos(y)
    s3 = -sn * np.sin(y)
    sxy = np.sqrt(s1**2 + s2**2)
    
    # 求经纬度
    lambda_d_rad = LAMBDA_D * PI / 180.0
    lon = np.arctan(s2 / s1) * 180.0 / PI + LAMBDA_D
    lat = np.arctan((EA**2 / EB**2) * s3 / sxy) * 180.0 / PI
    
    return lat, lon

# 便捷的单点转换函数
def latlon_to_linecolumn_calc_single(lat: float, lon: float, resolution: int = 4000) -> Tuple[int, int]:

    line, column = latlon_to_linecolumn_calc([lat], [lon], resolution)
    return int(line[0]), int(column[0])

def linecolumn_to_latlon_calc_single(line: int, column: int, resolution: int = 4000) -> Tuple[float, float]:

    lat, lon = linecolumn_to_latlon_calc([line], [column], resolution)
    return float(lat[0]), float(lon[0])

if __name__ == "__main__":
    
    lat, lon = 30.0, 110.0
    line, column = latlon_to_linecolumn_calc_single(lat, lon, 4000)
    print(f"查找法 - 经纬度 ({lat}, {lon}) -> 行列号 ({line}, {column})")
        
    lat_back, lon_back = linecolumn_to_latlon_calc_single(line, column, 4000)
    print(f"查找法 - 行列号 ({line}, {column}) -> 经纬度 ({lat_back:.6f}, {lon_back:.6f})")
        
    # 测试批量转换
    lats = [30.0, 31.0, 32.0]
    lons = [110.0, 111.0, 112.0]
    lines, columns = latlon_to_linecolumn_calc(lats, lons, 4000)
    print(f"查找法批量转换 - 行号: {lines}, 列号: {columns}")