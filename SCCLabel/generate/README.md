### generate_labels.py
按论文里标签生成过程，论文3.2.2节提取的纹理特征以及TBB12_TBB13好像没用上？
形态学闭运算采用3*3的椭圆核，是否有更好的选择？

#### 函数含义
* generate_convective_labels - 根据多通道卫星数据生成对流云标签，支持风4和葵花
    输入三维np数组，输出包含中间结果的字典或者只需要最终标签的二维np数组
    用户可传入字典自定义阈值参数
* generate_convective_labels_with_btd - 亮温差法的具体实现
* save_labels - 保存标签为png或npy文件
#### 中间结果的含义
* spectral_features - 光谱特征, 字典, key包括TBB9、TBB12、TBB9_TBB12、TBB12_TBB13
* binary_masks - 初始二值掩码, 字典, key包括TBB9、TBB12、TBB9_TBB12、TBB12_TBB13
* closed_masks - 形态学闭运算后的掩码, 字典, key包括TBB9、TBB12、TBB9_TBB12、TBB12_TBB13
* intersection_mask - TBB9、TBB12和TBB9_TBB12掩码的交集, np二维二值数组
* final_labels - 最终生成的去除零星噪点后的对流云标签, np二维二值数组


### visualize_results.py
可视化中间结果、最终标签结果以及展示统计结果，统计结果在statistics.txt

### Data目录结构说明
* Data/samples:风4数据样例（hdf文件），是2023年8月1号的数据
* Data/labels:最终标签的png文件
* Data/vis:可视化结果存放目录