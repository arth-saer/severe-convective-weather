import os
import numpy as np
import cv2

from generate_labels import generate_convective_labels, save_labels

root_path = '/mnt/md1/ConvectionAirport/Datasets/Satellite/FY4A/MCSRSI14C_2018_v3'

i = 0

dirs = sorted(os.listdir(root_path))
print(len(dirs))

for dir_name in dirs:
    
    np_file_path = os.path.join(root_path, dir_name, 'raw_data', dir_name + '.npy')
    all_channel_data = np.load(np_file_path)
    result_dict = generate_convective_labels(all_channel_data=all_channel_data, satellite_type='FY4A')
    final_labels = result_dict['final_labels']
    
    # 保存标签结果
    output_path = os.path.join(root_path, dir_name, 'output')
    output_name = os.listdir(output_path)[0]
    save_path = os.path.join(output_path, output_name.replace('.png', '_btd.png'))
    save_labels(final_labels, save_path=save_path, save_type='png')
    
    
    if i == 0:
        print(all_channel_data.shape)
        print(final_labels.shape)
    if i % 100 == 0:
        print(i)
        
    i += 1
    
    
    # output_path = os.path.join(root_path, dir_name, 'output')
    # for output_name in os.listdir(output_path):
        
    #     img = cv2.imread(os.path.join(output_path, output_name), cv2.IMREAD_GRAYSCALE)

    #     print(output_name)
    #     print(type(img))  # <class 'numpy.ndarray'>
    #     print(img.shape)  # (高度, 宽度)
    #     print(img.dtype)  # 通常是 uint8
    #     print(img.max())
    #     print(img.min())