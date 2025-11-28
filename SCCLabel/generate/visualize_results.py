import os
from typing import Dict
import numpy as np
from scipy import ndimage
from skimage import measure
import matplotlib.pyplot as plt

def results_visualize(result_dict, root_path, dir_name):
    """
    结果可视化函数，用于展示中间、最终结果以及统计结果
    
    Args:
        result_dict: 包含结果的字典
        root_path: 可视化结果的根目录
        dir_name: 根目录下某时刻结果的存放目录，例如20230801000000-20230801001459
    """
    dir_path = os.path.join(root_path, dir_name)
    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # 统计结果
    stats = calculate_statistics(result_dict)
    save_statistics_to_txt(stats, txt_path=os.path.join(dir_path, 'statistics.txt'))
    
    # 可视化光谱特征
    for key, feature in result_dict['spectral_features'].items():
        plt.imshow(feature, cmap='jet')
        plt.colorbar()
        plt.title(f'Spectral Feature: {key}')
        plt.savefig(os.path.join(dir_path, f'spectral_features_{key}.png'))
        plt.close()
    
    # 可视化二值化掩码
    for key, mask in result_dict['binary_masks'].items():
        plt.imshow(mask, cmap='gray')
        plt.title(f'Binary Mask: {key}')
        plt.savefig(os.path.join(dir_path, f'binary_masks_{key}.png'))
        plt.close()
    
    # 可视化闭操作结果
    for key, mask in result_dict['closed_masks'].items():
        plt.imshow(mask, cmap='gray')
        plt.title(f'Closed Mask: {key}')
        plt.savefig(os.path.join(dir_path, f'closed_masks_{key}.png'))
        plt.close()
    
    # 可视化交集掩码
    plt.imshow(result_dict['intersection_mask'], cmap='gray')
    plt.title('Intersection Mask')
    plt.savefig(os.path.join(dir_path, 'intersection_mask.png'))
    plt.close()
    
    # 可视化最终标签
    plt.imshow(result_dict['final_labels'], cmap='gray')
    plt.title('Final Labels')
    plt.savefig(os.path.join(dir_path, 'final_labels.png'))
    plt.close()



def calculate_statistics(result_dict) -> Dict:
    """
    计算各种统计量
    """
    stats = {}
    
    final_labels = result_dict['final_labels']
    spectral_features = result_dict['spectral_features']
    binary_masks = result_dict['binary_masks']
    closed_masks = result_dict['closed_masks']
    intersection_mask = result_dict['intersection_mask']
    
    
    # 基本形状信息
    stats['image_shape'] = final_labels.shape
    stats['total_pixels'] = final_labels.size
    
    # 光谱特征统计
    for key, feature in spectral_features.items():
        stats[f'{key}_mean'] = np.mean(feature)
        stats[f'{key}_std'] = np.std(feature)
        stats[f'{key}_min'] = np.min(feature)
        stats[f'{key}_max'] = np.max(feature)
    
    # 各掩码的像素数量统计
    for key, mask in binary_masks.items():
        stats[f'binary_masks_{key}_count'] = np.sum(mask)
        stats[f'binary_masks_{key}_ratio'] = np.sum(mask) / mask.size
    
    for key, mask in closed_masks.items():
        stats[f'closed_masks_{key}_count'] = np.sum(mask)
        stats[f'closed_masks_{key}_ratio'] = np.sum(mask) / mask.size
    
    stats['intersection_mask_count'] = np.sum(intersection_mask)
    stats['intersection_mask_ratio'] = np.sum(intersection_mask) / intersection_mask.size
    
    stats['final_labels_count'] = np.sum(final_labels)
    stats['final_labels_ratio'] = np.sum(final_labels) / final_labels.size
    
    # 连通区域统计（滤除零星噪点前后）
    labeled, num_regions = ndimage.label(intersection_mask)
    regions = measure.regionprops(labeled)
    
    stats['num_regions'] = num_regions
    if num_regions > 0:
        areas = [region.area for region in regions]
        stats['max_region_area'] = max(areas)
        stats['min_region_area'] = min(areas)
        stats['mean_region_area'] = np.mean(areas)
        stats['total_region_area'] = sum(areas)
    else:
        stats['max_region_area'] = 0
        stats['min_region_area'] = 0
        stats['mean_region_area'] = 0
        stats['total_region_area'] = 0
        
    labeled, num_regions_final = ndimage.label(final_labels)
    regions = measure.regionprops(labeled)
    
    stats['num_regions_final'] = num_regions_final
    if num_regions_final > 0:
        areas = [region.area for region in regions]
        stats['max_region_area_final'] = max(areas)
        stats['min_region_area_final'] = min(areas)
        stats['mean_region_area_final'] = np.mean(areas)
        stats['total_region_area_final'] = sum(areas)
    else:
        stats['max_region_area_final'] = 0
        stats['min_region_area_final'] = 0
        stats['mean_region_area_final'] = 0
        stats['total_region_area_final'] = 0    
    
    return stats


def save_statistics_to_txt(statistics, txt_path):
    """
    将统计量保存到TXT文件
    """
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("强对流云标签统计报告\n")
        f.write("=" * 50 + "\n\n")
        
        # 基本图像信息
        f.write("1. 图像基本信息:\n")
        f.write(f"   图像形状: {statistics['image_shape']}\n")
        f.write(f"   总像素数: {statistics['total_pixels']:,}\n\n")
        
        # 光谱特征统计
        f.write("2. 光谱特征统计:\n")
        for key in ['TBB9', 'TBB12', 'TBB9_TBB12', 'TBB12_TBB13']:
            f.write(f"   {key}:\n")
            f.write(f"     均值: {statistics[f'{key}_mean']:.2f} K\n")
            f.write(f"     标准差: {statistics[f'{key}_std']:.2f} K\n")
            f.write(f"     最小值: {statistics[f'{key}_min']:.2f} K\n")
            f.write(f"     最大值: {statistics[f'{key}_max']:.2f} K\n")       
        f.write("\n")
        
        # 像素数量统计
        f.write("3. 像素数量统计:\n")
        
        for key in ['TBB9', 'TBB12', 'TBB9_TBB12', 'TBB12_TBB13']:
            binary_count = statistics[f'binary_masks_{key}_count']
            binary_ratio = statistics[f'binary_masks_{key}_ratio']
            closed_count = statistics[f'closed_masks_{key}_count']
            closed_ratio = statistics[f'closed_masks_{key}_ratio']
            f.write(f"   {key}:\n")
            f.write(f"      初始二值掩码: {binary_count:,} ({binary_ratio:.2%})\n")
            f.write(f"      形态学闭操作后: {closed_count:,} ({closed_ratio:.2%})\n")
            
        f.write(f"   交集掩码像素数: {statistics['intersection_mask_count']:,} "
               f"({statistics['intersection_mask_ratio']:.2%})\n")            
        f.write(f"   最终标签像素数: {statistics['final_labels_count']:,} "
               f"({statistics['final_labels_ratio']:.2%})\n")

        f.write("\n")
        
        # 连通区域统计
        f.write("4. 连通区域统计:\n")
        f.write("   滤除零星噪点前（即交集掩码）:\n")
        f.write(f"      区域数量: {statistics['num_regions']}\n")
        if statistics['num_regions'] > 0:
            f.write(f"      最大区域面积: {statistics['max_region_area']:,} 像素\n")
            f.write(f"      最小区域面积: {statistics['min_region_area']:,} 像素\n")
            f.write(f"      平均区域面积: {statistics['mean_region_area']:.1f} 像素\n")
            f.write(f"      区域总面积: {statistics['total_region_area']:,} 像素\n")
            
        f.write("   滤除零星噪点后（即最终标签）:\n")
        f.write(f"      区域数量: {statistics['num_regions_final']}\n")
        if statistics['num_regions_final'] > 0:
            f.write(f"      最大区域面积: {statistics['max_region_area_final']:,} 像素\n")
            f.write(f"      最小区域面积: {statistics['min_region_area_final']:,} 像素\n")
            f.write(f"      平均区域面积: {statistics['mean_region_area_final']:.1f} 像素\n")
            f.write(f"      区域总面积: {statistics['total_region_area_final']:,} 像素\n")
        

    

    