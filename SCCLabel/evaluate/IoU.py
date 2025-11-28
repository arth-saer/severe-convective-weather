import cv2
import numpy as np

def calculate_iou(png_path_a, png_path_b, threshold=0.5):
    """
    计算两个PNG标签文件的正样本IoU，支持非严格二值图像。
    
    参数:
        png_path_a (str): 第一个标签文件的路径
        png_path_b (str): 第二个标签文件的路径  
        threshold (float): 二值化阈值，范围[0, 1]，默认0.5
                          （对于0-255的图像，实际阈值=threshold*255）
    返回:
        float: 正样本的IoU值
    """
    
    # 1. 读取图像
    labels_a = cv2.imread(png_path_a, cv2.IMREAD_GRAYSCALE)
    labels_b = cv2.imread(png_path_b, cv2.IMREAD_GRAYSCALE)
    
    if labels_a is None or labels_b is None:
        raise FileNotFoundError("无法读取PNG文件，请检查路径是否正确。")
    
    if labels_a.shape != labels_b.shape:
        raise ValueError("两张标签图的尺寸必须相同！")
    
    # 2. 归一化到[0, 1]范围，然后根据阈值二值化
    # 将uint8类型的0-255映射到0-1的浮点数
    norm_a = labels_a.astype(np.float32) / 255.0
    norm_b = labels_b.astype(np.float32) / 255.0
    
    # 使用阈值进行二值化
    binary_a = norm_a > threshold
    binary_b = norm_b > threshold
    
    # 3. 计算交集和并集
    intersection = np.logical_and(binary_a, binary_b)
    union = np.logical_or(binary_a, binary_b)
    
    intersection_count = np.sum(intersection)
    union_count = np.sum(union)
    
    # 4. 计算IoU
    if union_count == 0:
        iou = 1.0
    else:
        iou = intersection_count / union_count
        
    # return iou
    return iou, intersection_count, union_count

if __name__ == '__main__':
    iou, intersection_count, union_count = calculate_iou("/mnt/md1/ConvectionAirport/Datasets/Satellite/FY4A/MCSRSI14C_2018_v3/20180916123000/output/20180916123000_20180916123416.png", 
                                       "/mnt/md1/ConvectionAirport/Datasets/Satellite/FY4A/MCSRSI14C_2018_v3/20180916123000/output/20180916123000_20180916123416_btd.png")
    print(f"IoU计算结果: {iou:.4f}")
    print(intersection_count)
    print(union_count)