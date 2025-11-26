import os
from typing import Dict
import random
import numpy as np
from scipy import ndimage
from skimage import measure
import cv2

from hdfpreprocess_multi import read_hdf
from visualize_results import results_visualize


def generate_convective_labels(all_channel_data: np.ndarray, satellite_type='Himawari', thresholds=None):
    """
    根据卫星类型选择通道数据并调用亮温差法生成强对流云标签
    
    Args:
        all_channel_data: 包含所有通道亮温数据的三维np数组
        satellite_type: 卫星类型，支持 'Himawari' 和 'FY4A'
        
    Returns:
        result_dict: 包括最终标签以及中间结果，类型是字典
        result_dict['final_labels']: 如果不需要中间结果，也可直接返回最终标签，类型是二维np数组
    """
    
    if satellite_type == 'Himawari':
        TBB9 = all_channel_data[8]   # 6.25μm
        TBB12 = all_channel_data[13] # 10.8μm
        TBB13 = all_channel_data[15] # 12.0μm
    elif satellite_type == 'FY4A':
        TBB9 = all_channel_data[9]   # 6.25μm
        TBB12 = all_channel_data[12]  # 10.8μm
        TBB13 = all_channel_data[13] # 12.0μm
    else:
        raise ValueError("Unsupported satellite type. Supported types are 'Himawari' and 'FY4A'.")
    
    result_dict = generate_convective_labels_with_btd(TBB9, TBB12, TBB13, thresholds=thresholds)
    
    # 如果只需要最终标签，可以取消下面的注释
    # return result_dict['final_labels']
    
    return result_dict


def generate_convective_labels_with_btd(TBB9: np.ndarray,
                                        TBB12: np.ndarray,
                                        TBB13: np.ndarray,
                                        thresholds=None) -> Dict:
    """
    亮温差法生成强对流云标签
    
    Args:
        TBB9: 水汽通道亮温数据 (6.25μm)
        TBB12: 红外窗区通道1亮温数据 (10.8μm)  
        TBB13: 红外窗区通道2亮温数据 (12.0μm)
        
    Returns:
        result_dict: 包含最终标签和详细中间结果的字典
    """
    default_thresholds = {
        'TBB9': 220,      # K - 6.25微米水汽通道亮温阈值
        'TBB12': 215,     # K - 10.8微米通道亮温阈值
        'TBB9_TBB12': -4, # K - 水汽-红外窗区亮温差阈值
        'TBB12_TBB13': 2, # K - 分裂窗亮温差阈值
        'min_area': 4     # 像素数 - 最小面积阈值
    }
    
    if thresholds is None:
        thresholds = default_thresholds
    else:
        # 确保用户提供的阈值字典包含所有必要的键
        for key in default_thresholds.keys():
            if key not in thresholds:
                thresholds[key] = default_thresholds[key]

    
    # 1. 光谱特征提取
    spectral_features = {
        'TBB9': TBB9,
        'TBB12': TBB12,
        'TBB9_TBB12': TBB9 - TBB12,  # 水汽-红外窗区亮温差
        'TBB12_TBB13': TBB12 - TBB13  # 分裂窗亮温差
    }
    
    # 2. 光谱特征二值化
    binary_masks = {
        'TBB9': spectral_features['TBB9'] < thresholds['TBB9'],
        'TBB12': spectral_features['TBB12'] < thresholds['TBB12'],
        'TBB9_TBB12': spectral_features['TBB9_TBB12'] > thresholds['TBB9_TBB12'],
        'TBB12_TBB13': spectral_features['TBB12_TBB13'] < thresholds['TBB12_TBB13']
    }
    
    # 3. 形态学闭操作，使用3*3的椭圆核，参数可改
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    closed_masks = {}
    for key, mask in binary_masks.items():
        closed_mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        closed_masks[key] = closed_mask.astype(bool)
    
    # 4. 交集运算
    # （TBB9和TBB12的交集）
    intersection_mask = np.logical_and(
        closed_masks['TBB9'], 
        closed_masks['TBB12']
    )
    # 再跟TBB9-TBB12做交集
    intersection_mask = np.logical_and(
        intersection_mask,
        closed_masks['TBB9_TBB12']
    )
    
    # 5. 面积滤波，去除小面积噪声（滤除零星噪点）
    labeled, num = ndimage.label(intersection_mask)
    regions = measure.regionprops(labeled)
    
    final_labels = np.zeros_like(intersection_mask, dtype=bool)
    for region in regions:
        if region.area >= thresholds['min_area']:
            final_labels[labeled == region.label] = True
    
    # 返回结果
    result_dict = {
        'final_labels': final_labels,
        'spectral_features': spectral_features,
        'binary_masks': binary_masks,
        'closed_masks': closed_masks,
        'intersection_mask': intersection_mask
    }
    
    return result_dict


def save_labels(final_labels: np.ndarray, save_path, save_type="png"):
    """
    保存最终标签结果
    
    Args:
        save_path: 标签结果的保存路径
        save_type: 标签结果的保存方式，支持png、npy
    """
    final_labels_uint8 = final_labels.astype(np.uint8) * 255
    if save_type == 'png':
        cv2.imwrite(save_path, final_labels_uint8)
    elif save_type == 'npy':
        np.save(save_path, final_labels)
    else:
        raise ValueError("Unsupported save type. Supported types are 'png' and 'npy'.")



if __name__ == '__main__':
    
    samples_root_path = '/mnt/md1/lxw/severe-convective-weather/SCCLabel/Data/examples/20230301'
    labels_root_path = '/mnt/md1/lxw/severe-convective-weather/SCCLabel/Data/labels/20230301'
    vis_root_path = '/mnt/md1/lxw/severe-convective-weather/SCCLabel/Data/vis'
    
    if not os.path.exists(labels_root_path):
        os.makedirs(labels_root_path)
    
    sample_files = sorted(os.listdir(samples_root_path))
    for sample_file in sample_files:
        
        sample_time = sample_file.split('_')[-4] + '_' + sample_file.split('_')[-3]
        
        sample_np = read_hdf(file_path=os.path.join(samples_root_path, sample_file))
        result_dict = generate_convective_labels(all_channel_data=sample_np, satellite_type='FY4A')
        final_labels = result_dict['final_labels']
    
        # 保存标签结果
        save_labels(final_labels, 
                    save_path=os.path.join(labels_root_path, sample_time + '.png'), save_type='png')
        
        # 可视化，用于调试
        # p = random.random()
        # if p < 0.1:
        #     results_visualize(result_dict, root_path=vis_root_path, dir_name=sample_time)
    
                
    print(sample_np.shape)
    print(final_labels.shape)