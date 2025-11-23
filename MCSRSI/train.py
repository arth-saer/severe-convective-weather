import os
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


from data.dataset import SeqsDataSet, NoSeqsDataSet
from model.unet3D import UNet3D
from model.convlstm import ConvLSTMCCD
from model.unet2D import UNet2D
from model.factory import TrainingComponentFactory
from utils.utils import set_seed, evaluate_model, format_time



def train(model, model_name, loss_name='Focal', optimizer_name='Adam', scheduler_name='None'):
    
    # 环境配置
    os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
    set_seed()
    
    device_count = torch.cuda.device_count()
    print(f"检测到 {device_count} 个GPU")
    
    for i in range(device_count):
        gpu_name = torch.cuda.get_device_name(i)
        memory = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {gpu_name} ({memory:.1f} GB)")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    
    # 准备数据
    images_path = '/mnt/md1/ConvectionAirport/Datasets/Satellite/FY4A/MCSRSI/bright_images'
    labels_path = '/mnt/md1/ConvectionAirport/Datasets/Satellite/FY4A/MCSRSI/labels_v2/all'
       
    train_dataset = SeqsDataSet(
        json_path='./data/train_seqs_6_15.json',
        images_path=images_path,
        labels_path=labels_path
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    print(f"训练样本数: {len(train_dataset)}")
    print(f"训练批次数: {len(train_loader)}")
    
    valid_dataset = SeqsDataSet(
        json_path='./data/valid_seqs_6_15.json',
        images_path=images_path,
        labels_path=labels_path
    )
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)
    print(f"验证样本数: {len(valid_dataset)}")
    
    test_dataset = SeqsDataSet(
        json_path='./data/test_seqs_6_15.json',
        images_path=images_path,
        labels_path=labels_path
    )
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    print(f"测试样本数: {len(test_dataset)}")
    print(f"测试批次数: {len(test_loader)}")    
    
    
    
    model = model.to(device)
    
    if device_count > 1:
        print(f"启用数据并行，使用 {device_count} 个GPU")
        model = nn.DataParallel(model)
    else:
        print("单GPU模式")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params}")

    
    criterion = TrainingComponentFactory.build_loss_fn(loss_name=loss_name, alpha=0.7)
    print(f"损失函数: {loss_name}")
    
    optimizer = TrainingComponentFactory.build_optimizer(
        optimizer_name=optimizer_name, model=model,
        lr=0.0005
    )
    print(f"优化器: {optimizer_name}")
    
    scheduler = TrainingComponentFactory.build_scheduler(
        scheduler_name=scheduler_name, optimizer=optimizer
    )
    print(f"学习率调度器: {scheduler_name}")
    
    
    experiment_dir = f"./result/{model_name}_{optimizer_name}_{loss_name}_{scheduler_name}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    best_f1_score = 0.0
    best_model_path = os.path.join(experiment_dir, 'best_model.pth')
    
    

    # 模型训练
    train_start_time = time.time()
    start_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    print("\n" + "*" * 90)
    print("开始训练...")
    print(f"开始时间: {start_datetime}")
    print("*" * 90 + "\n")
     
    for epoch in range(10):
        model.train()
        train_loss = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device).float()
            labels = labels.to(device).float()
            
            target_labels = labels[:, -1]
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, target_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, 训练损失: {loss.item():.4f}')
                
        if scheduler is not None:
            scheduler.step()
        
        avg_train_loss = train_loss / len(train_loader)
        
        avg_val_loss, avg_metrics = evaluate_model(model, criterion, valid_loader, device)
        
        print(f"Epoch {epoch+1} 完成:")
        print(f"训练损失: {avg_train_loss:.4f}")
        print(f"验证损失: {avg_val_loss:.4f}")
        print(f"精确率: {avg_metrics['precision']:.4f}")
        print(f"误报率: {avg_metrics['far']:.4f}")
        print(f"召回率: {avg_metrics['recall/pod']:.4f}")
        print(f"漏报率: {avg_metrics['fnr']:.4f}")
        print(f"F1分数: {avg_metrics['f1']:.4f}")
        print(f"CSI: {avg_metrics['csi']:.4f}")
        
        if epoch % 2 == 1:            
            checkpoint_path = os.path.join(experiment_dir, f'model_epoch_{epoch+1}.pth')
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), checkpoint_path)
            else:
                torch.save(model.state_dict(), checkpoint_path)
            print(f"保存检查点: {checkpoint_path}")
        
        if avg_metrics['f1'] >= best_f1_score:
            best_f1_score = avg_metrics['f1']
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), best_model_path)
            else:
                torch.save(model.state_dict(), best_model_path)
            print(f"保存最佳模型 (f1分数: {best_f1_score:.4f})")
        
        print("-" * 90)
        
    train_end_time = time.time()
    end_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    train_duration = train_end_time - train_start_time
    
    print("\n" + "*" * 90)    
    print(f"训练完成！最佳验证f1分数: {best_f1_score:.4f}")   
    print(f"完成时间: {end_datetime}")
    print(f"训练完成 (总耗时: {format_time(train_duration)})")
    print("*" * 90)



    # 模型测试
    print("\n" + "=" * 90)
    print("测试集评估")
    print("="*90)
    
    if os.path.exists(best_model_path):
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
        else:
            model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
        print(f"加载最佳模型: {best_model_path}")
    
    test_loss, test_metrics = evaluate_model(model, criterion, test_loader, device)
    
    print(f"测试损失: {test_loss:.4f}")
    print(f"精确率: {test_metrics['precision']:.4f}")
    print(f"误报率: {test_metrics['far']:.4f}")
    print(f"召回率: {test_metrics['recall/pod']:.4f}")
    print(f"漏报率: {test_metrics['fnr']:.4f}")
    print(f"F1分数: {test_metrics['f1']:.4f}")
    print(f"CSI: {test_metrics['csi']:.4f}")



if __name__ == '__main__':
    model = UNet3D()
    model_name = 'UNet3D'
    train(model=model, model_name=model_name)