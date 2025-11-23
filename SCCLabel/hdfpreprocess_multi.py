import os
import h5py
import numpy as np
import pandas as pd
from PIL import Image
import time
from datetime import datetime
import shutil
import tqdm
# import win32api
# from config import get_config

# config = get_config('config.txt')
# if not os.path.exists('tmp'):
#     os.mkdir('tmp')


# 风云数据每个通道的索引
channels = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]

# 光伏的场站的归一化代码
# max_values = [0.4,0.4,0.4,0.15,0.4,0.32,320,320,280,300,300,300,300,280]
# min_values = [  0,  0,  0,   0,  0,   0,220,220,220,200,210,210,210,220]

# 2025-07-01
# 使用Panoply打开FY4A HDF后得到的每个通道辐射定标后亮温的大概取值范围，
# 其中，为了方便与祖传代码做对照实验，第12通道（10.8微米）通道祖传的180-320这个阈值
# 这个min_values和max_values的作用是1.限定亮温度的温度范围 2.用于将亮温图归一化为PNG图片。由于我目前提供的范围都是偏大的，所以对于功能1是没影响的，有影响的是功能2，因为这可能会导致图片偏暗色
# max_values = [1.0,1.0,1.0,1.0,1.0,1.0,500,340,270,280,310,320,340,280]
# min_values = [  0,  0,  0,   0,  0,   0,200,160,110,110,120,180,110,110]

# 2025-07-02
# 非常尴尬！今天开完组会后，我又检查了一下风云的数据，发现每个月的定标表的取值范围都不一样，我决定直接按照风云官方的数据集
# 整一个100-500作为定标表
max_values = [1.5,1.5,1.5,1.5,1.5,1.5,500,500,500,500,500,500,500,500]
min_values = [  0,  0,  0,   0,  0,   0,100,100,100,100,100,100,100,100]

def save_numpy(image_path, store_path, file_name):

    nom_imgs = read_hdf(image_path)
    if nom_imgs is not None and nom_imgs.shape == (14, 730, 1280):
        file_store_dir = os.path.join(store_path, file_name[44:50])
        if not os.path.exists(file_store_dir):
            os.makedirs(file_store_dir)
        nom_imgs = nom_imgs.astype(np.float32)
        np.save(os.path.join(file_store_dir, file_name[44:58] + '.npy'), nom_imgs)
def save_img(image_path, store_path, file_name):
    '''
    将FY-4A的HDF文件处理成14个通道的png图像
    image_path: hdf文件路径
    store_path: 存储路径
    file_name: 文件名
    '''
    nom_imgs = read_hdf(image_path)
    if nom_imgs is not None and nom_imgs.shape == (14, 730, 1280):
        file_store_dir = os.path.join(store_path, file_name[44:50], file_name[44:58])
        if not os.path.exists(file_store_dir):
            os.makedirs(file_store_dir)
        nom_imgs = nom_imgs.astype(np.float32)
        
        for c in range(nom_imgs.shape[0]):
            if c < 6:
                bright = 255 * (nom_imgs[c] - min_values[c]) / (max_values[c] - min_values[c]) 
            else:
                bright = 255 * (max_values[c] - nom_imgs[c]) / (max_values[c] - min_values[c])
            bright_img = Image.fromarray(bright.astype('uint8'))
            bright_img.save(os.path.join(file_store_dir, f"channel{c+1}.png"))

def save_hdf(image_path, store_path, file_name):
    '''
    将HDF文件转存到指定的路径
    image_path: hdf文件路径
    store_path: 存储路径
    file_name: 文件名
    '''
    year=file_name[44:48]
    month=file_name[48:50]
    date=file_name[44:52]
    file_store_dir = os.path.join(store_path, year, month, date)
    if not os.path.exists(file_store_dir):
        os.makedirs(file_store_dir)
    
    # 将hdf复制到指定目录
    shutil.copy(os.path.join(image_path,file_name), file_store_dir)
# # 将缓存的数据转换为图像输出
# def data2png(imgs, store_path, file_name):

#     # 当前文件
#     # 可见光图像，直接归一化输出
#     # light = 255 * imgs[0, :, :] / (max_values[0] - min_values[0])
#     # 反转归一化输出
#     # light = 255 * (max_values[0] - imgs[0, :, :]) / (max_values[0] - min_values[0])
#     #
#     # light = 255 * (imgs[0] - np.min(imgs[0])) / (np.max(imgs[0]) - np.min(imgs[0]))
#     # # light = 255 * imgs[0]
#     #
#     # bright = 255 * (max_values[0] - imgs[0]) / (max_values[0] - min_values[0])


#     # light_img = Image.fromarray(light.astype('uint8'))
#     # light_img.save(os.path.join(path, 'light_img.png'))

#     # 亮温通道，直接反转归一化输出 1 - 0-1 / 1 -0 * 255
#     # bright = 255 * ( imgs[0] -  min_values[0]) / (max_values[0] - min_values[0])
#     numChannels, _, _= imgs.shape
#     for i in range(0,numChannels):
#         if i < 6:
#             bright = 255 * (imgs[i]-  min_values[i]) / (max_values[i] - min_values[i])
#         else:
#             bright = 255 * (max_values[i] - imgs[i]) / (max_values[i] - min_values[i])
#         bright_img = Image.fromarray(bright.astype('uint8'))
#         # bright_img.save(path[:-3] + "png")
#         saveFolder = store_path + "/" + str(i+1) + "/"
#         savedir = os.path.join(saveFolder, file_name[15:19], file_name[44:48], file_name[44:50], file_name[44:52]) + "/"
#         if not os.path.exists(savedir):
#             os.makedirs(savedir)
#         bright_img.save(savedir + file_name[:-3] + "png")


# 读卫星图像
def read_hdf(file_path, nb_cols=1280, nb_rows=730):
    """
    :param file_path: 客户端传来的参数，当前文件路径
    :param nb_cols: 图片的宽度
    :param nb_rows: 图片的长度
    :return:
    """
    # print(file_path)
    fdi_file = h5py.File(file_path)
    # 文件在全员盘中的开始列
    hdf_start_col = fdi_file.attrs['Begin Pixel Number'][0]
    # 文件在全员盘中的开始行
    hdf_start_row = fdi_file.attrs['Begin Line Number'][0]
    # 中国区在全圆盘的开始行
    absolute_start_row = 190
    # 中国区在全圆盘的开始列
    absolute_start_col = 750
    row_start, row_end, col_start, col_end = get_region_from_all(hdf_start_row, hdf_start_col, absolute_start_row,
                                                                 absolute_start_col, nb_cols, nb_rows)
    nom_all = np.ones((len(channels), nb_rows, nb_cols)) * 65535
    if 'NOMChannel01' in fdi_file.keys():
        for i, c in enumerate(channels):
            nom_c = fdi_file['NOMChannel'+str(c).zfill(2)][:, col_start:col_end]
            nom_c = nom_c[row_start:row_end, :]

            cal_c = fdi_file['CALChannel'+str(c).zfill(2)][:]
            idx_err = (nom_c >= len(cal_c))
            nom_c[idx_err] = 0
            nom_c = cal_c[nom_c]
            nom_c[idx_err] = 65535

            nom_all[i, :, :] = nom_c

            # 取值范围限制，异常值65535在PNG图片中对应的值应该为0像素，即全黑
            nom_all[i, :, :] = np.clip(nom_all[i, :, :], min_values[i], max_values[i])
        fdi_file.close()
        return nom_all
    else:
        # fo = open("incompletefile.txt", "a")
        # fo.write( f"{file_path}\n")
        # fdi_file.close()
        return None

# 将ppt中给的坐标转换到文件中相应区域的坐标
def get_region_from_all(start_row, start_col, region_start_row, region_start_col, nb_cols, nb_rows):
    # 完全包在里面
    if region_start_row >= start_row:
        row_start = region_start_row - start_row
        row_end = region_start_row - start_row + nb_rows
        col_start = region_start_col - start_col
        col_end = region_start_col - start_col + nb_cols
    else:
        row_start = 0
        row_end = region_start_row + nb_rows - start_row
        col_start = region_start_col - start_col
        col_end = region_start_col - start_col + nb_cols
    return row_start, row_end, col_start, col_end    


import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def process_file(args):
    '''
    数据处理,同时生成npy和img文件
    '''
    file_path, store_dir, file = args
    try:
        hdf_path = os.path.join(file_path, file)
        # 保存PNG图片
        png_store_dir=os.path.join(store_dir,"PNG")
        save_img(hdf_path, png_store_dir, file)
        # 保存NPY文件
        npy_store_dir=os.path.join(store_dir,"NPY")
        save_numpy(hdf_path, npy_store_dir, file)
    except Exception as e:
        print(f"Error processing {hdf_path}: {e}")

def move_files(root_dir, store_dir):
    '''
    将FY4A原始数据保存为PNG和npy
    '''
    file_args = []
    for reg in os.listdir(root_dir):
        for day_dir in sorted(os.listdir(os.path.join(root_dir, reg))):
            day_path = os.path.join(root_dir, reg, day_dir)
            for file in sorted(os.listdir(day_path)):
                time_region = file[44:58]
                date_time = time.strptime(time_region, '%Y%m%d%H%M%S')
                time_stamp = time.mktime(date_time)  # 秒数
                minutes = int(time_stamp // 60)
                if minutes % 15 != 0: 
                    continue

                # 归档数据
                # save_hdf(day_path, store_dir, file)
                # 将FY-4A的HDF文件处理成14个通道的png图像和NPY图象
                process_file((day_path,store_dir,file))            
            print(f"{reg} {day_dir} ok...")
def move_files_FY4A15minData(root_dir, store_dir):
    '''
    遍历FY4A15minData文件夹，将该文件夹下的FY数据数据都变为PNG和npy
    '''
    file_args = []
    # 遍历年份
    for year in os.listdir(root_dir):
        year_dir=os.path.join(root_dir,year)
        # 判断year_dir是否为文件夹
        if not os.path.isdir(year_dir):
            continue
        # 遍历月份
        sorted_month_list=sorted(os.listdir(year_dir))
        for month in sorted_month_list:
            month_path = os.path.join(year_dir, month)
            # 遍历日期yyyymmdd
            sorted_day_list=sorted(os.listdir(month_path))
            for day in sorted_day_list:
                day_path = os.path.join(month_path, day)
                # 判断是否为文件夹
                if not os.path.isdir(day_path):
                    continue
                    
                # 遍历文件
                sorted_file_list=sorted(os.listdir(day_path))
                for file in sorted_file_list:
                    time_region = file[44:58]
                    date_time = time.strptime(time_region, '%Y%m%d%H%M%S')
                    time_stamp = time.mktime(date_time)  # 秒数
                    minutes = int(time_stamp // 60)
                    if minutes % 15 != 0: 
                        continue
                    
                    # 归档数据
                    # save_hdf(day_path, store_dir, file)
                    # 将FY-4A的HDF文件处理成14个通道的png图像和NPY图象
                    process_file((day_path,store_dir,file))            
                print(f"{day_path} ok...")
if __name__ == '__main__':
    # root_dir = "/data/FengYunData/FY4A"
    # store_dir = "/data/FengYunDataAfterProcessing/FY4A"
    # move_files(root_dir, store_dir)

    root_dir = "/data/FY4A15minData"
    store_dir = "/data/FengYunDataAfterProcessing/FY4A"
    move_files_FY4A15minData(root_dir, store_dir)

   