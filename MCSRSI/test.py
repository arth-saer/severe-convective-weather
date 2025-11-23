import os
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


from data.dataset import SeqsDataSet
from model.unet3D import UNet3D
from model.factory import TrainingComponentFactory
from utils.utils import set_seed, evaluate_model, format_time
def test():
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
    set_seed()
       
    device_count = torch.cuda.device_count()
    
    device = torch.device('cuda:0')
    print(f"使用主设备: {device}")
    
    
    # 模型定义
    model = UNet3D(in_channels=1, out_channels=1, base_channels=32)
    if device_count > 1:
        print(f"启用数据并行，使用 {device_count} 个GPU")
        model = nn.DataParallel(model)
    else:
        print("单GPU模式")
        
    model = model.to(device)
    
    model_name = 'UNet3D'
    loss_name = 'FocalDice'
    optimizer_name = 'AdamW'
    scheduler_name = 'cosine'
    
    criterion = TrainingComponentFactory.build_loss_fn(loss_name=loss_name)
    
    optimizer = TrainingComponentFactory.build_optimizer(
        optimizer_name=optimizer_name, model=model,
        lr=0.0001
    )
    
    experiment_dir = f"./result/{model_name}_{optimizer_name}_{loss_name}_{scheduler_name}_1031_1455"
    
    best_model_path = os.path.join(experiment_dir, 'best_model.pth')
    
    if os.path.exists(best_model_path):
        if device_count > 1:
            model.module.load_state_dict(torch.load(best_model_path, weights_only=True))
        else:
            model.load_state_dict(torch.load(best_model_path, weights_only=True))
        print(f"加载最佳模型: {best_model_path}")
        
    images_path = '/mnt/md1/ConvectionAirport/Datasets/Satellite/FY4A/MCSRSI/bright_images'
    labels_path = '/mnt/md1/ConvectionAirport/Datasets/Satellite/FY4A/MCSRSI/labels_v2/all'        
        
    test_dataset = SeqsDataSet(
        json_path='./data/test.json',
        images_path=images_path,
        labels_path=labels_path
    )
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    print(f"测试样本数: {len(test_dataset)}")
    print(f"测试批次数: {len(test_loader)}")    
    
    test_loss, test_metrics = evaluate_model(model, criterion, test_loader, device)
    
    print(f"测试损失: {test_loss:.4f}")
    print(f"精确率: {test_metrics['precision']:.4f}")
    print(f"误报率: {test_metrics['far']:.4f}")
    print(f"召回率: {test_metrics['recall/pod']:.4f}")
    print(f"漏报率: {test_metrics['fnr']:.4f}")
    print(f"F1分数: {test_metrics['f1']:.4f}")
    print(f"CSI: {test_metrics['csi']:.4f}")
    
test()
