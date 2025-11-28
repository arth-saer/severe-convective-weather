import os
import numpy as np
import matplotlib.pyplot as plt
from IoU import calculate_iou

root_path = '/mnt/md1/ConvectionAirport/Datasets/Satellite/FY4A/MCSRSI14C_2018_v3'


months = [f"2018{str(i).zfill(2)}" for i in range(1, 13)]

iou = {}
intersection_count = {}
union_count = {}
for month in months:
    iou[month] = []
    intersection_count[month] = []
    union_count[month] = []
iou['all'] = []
intersection_count['all'] = []
union_count['all'] = []


dirs = sorted(os.listdir(root_path))

for dir_name in dirs:
    
    output_path = os.path.join(root_path, dir_name, 'output')
    output_name = os.listdir(output_path)
    png_path_a = os.path.join(output_path, output_name[0])
    png_path_b = os.path.join(output_path, output_name[1])
    
    iou_, intersection_count_, union_count_ = calculate_iou(png_path_a, png_path_b)
    
    month_info = output_name[0][0:6]
    iou[month_info].append(iou_)
    iou['all'].append(iou_)
    intersection_count[month_info].append(intersection_count_)
    intersection_count['all'].append(intersection_count_)
    union_count[month_info].append(union_count_)
    union_count['all'].append(union_count_)

micro_means = [] #微平均
macro_means = [] #宏平均
for month in months:
    micro_means.append(np.sum(intersection_count[month]) / np.sum(union_count[month]))
    macro_means.append(np.mean(iou[month]))

# 整年的
all_micro = np.sum(intersection_count["all"]) / np.sum(union_count["all"])
all_macro = np.mean(iou["all"])

with open('/mnt/md1/lxw/severe-convective-weather/SCCLabel/evaluate/result/result.txt', 'w', encoding='utf-8') as f:
    f.write("2018年风云4A卫星数据-强对流云标注：亮温差法与人工标注的IoU结果\n")
    
    f.write("\n整年的:\n")
    f.write(f'   size: {len(iou["all"])}\n')
    f.write(f'   micro_mean（微平均）: {all_micro}\n')
    f.write(f'   macro_mean（宏平均）: {all_macro}\n')

    f.write("\n每月份的:\n")
    for i, month in enumerate(months):
        f.write(month + ':\n')
        f.write(f'   size: {len(iou[month])}\n')
        f.write(f'   micro_mean（微平均）: {micro_means[i]}\n')
        f.write(f'   macro_mean（宏平均）: {macro_means[i]}\n')
        

# 绘图
fig, ax1 = plt.subplots(figsize=(16, 8))

ax1.plot(months, micro_means, 'o-', linewidth=2, markersize=8, 
         label='Micro Average IoU', color="#38AB2E")
ax1.plot(months, macro_means, 's-', linewidth=2, markersize=8, 
         label='Macro Average IoU', color="#3B3DA2")
ax1.set_ylabel('IoU', fontsize=12, fontweight='bold')
ax1.set_ylim(0, 0.8)
for i, (micro, macro) in enumerate(zip(micro_means, macro_means)):
    ax1.annotate(f'{micro:.2f}', (i, micro), textcoords="offset points", 
                 xytext=(0,10),  ha='center', fontsize=8, color="#38AB2E")
    ax1.annotate(f'{macro:.2f}', (i, macro), textcoords="offset points", 
                 xytext=(0,-15), ha='center', fontsize=8, color="#3B3DA2")

ax2 = ax1.twinx()
x_pos = len(months)
bar_width = 0.35
ax2.bar(x_pos - bar_width/2, all_micro, bar_width, 
        label='All Year Micro Average IoU', color="#38AB2E", alpha=0.7)
ax2.bar(x_pos + bar_width/2, all_macro, bar_width, 
        label='All Year Macro Average IoU', color="#3B3DA2", alpha=0.7)
ax2.set_ylabel('IoU', fontsize=12, fontweight='bold')
ax2.set_ylim(0, 0.8)

ax1.set_title('Monthly IoU Trends + All Year Average', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('Month', fontsize=12, fontweight='bold')
ax1.set_xticks([i for i in range(13)])
ax1.set_xticklabels(months + ["all year"]) 
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.savefig('/mnt/md1/lxw/severe-convective-weather/SCCLabel/evaluate/result/result.png', dpi=300, bbox_inches='tight')
plt.close()
