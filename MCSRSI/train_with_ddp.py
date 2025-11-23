import os
import time
import logging
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from data.dataset import SeqsDataSet, NoSeqsDataSet
from model.unet3D import UNet3D
from model.convlstm import ConvLSTMCCD
from model.unet2D import UNet2D
from model.factory import TrainingComponentFactory
from utils.utils import set_seed, evaluate_model, format_time, setup_file_logging, find_available_port




def main_worker(local_rank, world_size, model, loss_name='Focal', optimizer_name='Adam', scheduler_name='None', experiment_dir=None):

    set_seed(43 + local_rank)
    
    log_file_path = os.path.join(experiment_dir, f'training_rank{local_rank}.log')
    logger = setup_file_logging(log_file_path)
    
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=local_rank
    )
    logger.info(f"进程 {local_rank} 初始化完成")
    
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    logger.info(f"进程 {local_rank} 使用 GPU {local_rank}")
    
    images_path = '/mnt/md1/ConvectionAirport/Datasets/Satellite/FY4A/MCSRSI/bright_images'
    labels_path = '/mnt/md1/ConvectionAirport/Datasets/Satellite/FY4A/MCSRSI/labels_v2/all'
    
    train_dataset = SeqsDataSet(
        json_path='./data/train_seqs_6_15.json',
        images_path=images_path, labels_path=labels_path
    )
    
    valid_dataset = SeqsDataSet(
        json_path='./data/valid_seqs_6_15.json',
        images_path=images_path, labels_path=labels_path
    )
    
    test_dataset = SeqsDataSet(
        json_path='./data/test_seqs_6_15.json',
        images_path=images_path, labels_path=labels_path
    )
    
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size, rank=local_rank, shuffle=True, seed=123
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=2, sampler=train_sampler, num_workers=4,
        pin_memory=True, drop_last=True
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=2, shuffle=False, num_workers=4
    ) if local_rank == 0 else None
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=2, shuffle=False, num_workers=4
    ) if local_rank == 0 else None
    
    if local_rank == 0:
        logger.info(f"训练样本数: {len(train_dataset)}")
        logger.info(f"训练批次数: {len(train_loader) * world_size}")
        logger.info(f"验证样本数: {len(valid_dataset)}")
        logger.info(f"测试样本数: {len(test_dataset)}")
        logger.info(f"测试批次数: {len(test_loader)}")
    
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])
    
    if local_rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"模型参数量: {total_params}")
    
    lr = 0.00025
    focal_loss_alpha = 0.67
    num_epochs = 10
    
    criterion = TrainingComponentFactory.build_loss_fn(loss_name=loss_name, alpha=focal_loss_alpha)
    optimizer = TrainingComponentFactory.build_optimizer(
        optimizer_name=optimizer_name, model=model.module,
        lr=lr
    )
    scheduler = TrainingComponentFactory.build_scheduler(
        scheduler_name=scheduler_name, optimizer=optimizer
    )
    
    if local_rank == 0:
        logger.info(f"损失函数: {loss_name}")
        logger.info(f"优化器: {optimizer_name}")
        logger.info(f"学习率调度器: {scheduler_name}")
        logger.info(f"实验目录: {experiment_dir}")
        logger.info(f"日志文件: {log_file_path}")
        
    best_model_path = os.path.join(experiment_dir, 'best_model.pth') if local_rank == 0 else None
    best_f1_score = 0.0
    
    dist.barrier()
    
    if local_rank == 0:
        train_start_time = time.time()
        start_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.info("=" * 80)
        logger.info("开始训练")
        logger.info(f"开始时间: {start_datetime}")
        logger.info("=" * 80)
    
    
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).float()
            
            target_labels = labels[:, -1]
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, target_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 50 == 0 and local_rank == 0:
                logger.info(f'Epoch {epoch+1}, Batch {batch_idx}, 训练损失: {loss.item():.4f}')
        
        train_loss_tensor = torch.tensor(train_loss / len(train_loader)).to(device)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        avg_train_loss = train_loss_tensor.item() / world_size

        if local_rank == 0:
            avg_val_loss, avg_metrics = evaluate_model(model, criterion, valid_loader, device)
        else:
            avg_val_loss, avg_metrics = 0.0, {}
        
        if scheduler is not None:
            scheduler.step()
        
        if local_rank == 0:
            logger.info(f"Epoch {epoch+1} 完成:")
            logger.info(f"训练损失: {avg_train_loss:.4f}")
            logger.info(f"验证损失: {avg_val_loss:.4f}")  
            logger.info(f"精确率: {avg_metrics.get('precision', 0):.4f}")
            logger.info(f"误报率: {avg_metrics.get('far', 0):.4f}")
            logger.info(f"召回率: {avg_metrics.get('recall/pod', 0):.4f}")
            logger.info(f"漏报率: {avg_metrics.get('fnr', 0):.4f}")
            logger.info(f"F1分数: {avg_metrics.get('f1', 0):.4f}")
            logger.info(f"CSI: {avg_metrics.get('csi', 0):.4f}")
            
            if epoch % 2 == 1:            
                checkpoint_path = os.path.join(experiment_dir, f'model_epoch_{epoch+1}.pth')
                torch.save(model.module.state_dict(), checkpoint_path)
                logger.info(f"保存检查点: {checkpoint_path}")
            
            if avg_metrics.get('f1', 0) >= best_f1_score:
                best_f1_score = avg_metrics['f1']
                torch.save(model.module.state_dict(), best_model_path)
                logger.info(f"保存最佳模型 (f1分数: {best_f1_score:.4f})")
            
            logger.info("-" * 60)
        
        dist.barrier()
    
    if local_rank == 0:
        train_end_time = time.time()
        end_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        train_duration = train_end_time - train_start_time
        
        logger.info("=" * 80)    
        logger.info("训练完成")
        logger.info(f"最佳验证f1分数: {best_f1_score:.4f}")   
        logger.info(f"完成时间: {end_datetime}")
        logger.info(f"总耗时: {format_time(train_duration)}")
        logger.info("=" * 80)
    
    if local_rank == 0:
        logger.info("测试集评估")
        logger.info("=" * 60)
        
        if os.path.exists(best_model_path):
            model.module.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
            logger.info(f"加载最佳模型: {best_model_path}")
        
        test_loss, test_metrics = evaluate_model(model, criterion, test_loader, device)
        
        logger.info(f"测试损失: {test_loss:.4f}")
        logger.info(f"精确率: {test_metrics.get('precision', 0):.4f}")
        logger.info(f"误报率: {test_metrics.get('far', 0):.4f}")
        logger.info(f"召回率: {test_metrics.get('recall/pod', 0):.4f}")
        logger.info(f"漏报率: {test_metrics.get('fnr', 0):.4f}")
        logger.info(f"F1分数: {test_metrics.get('f1', 0):.4f}")
        logger.info(f"CSI: {test_metrics.get('csi', 0):.4f}")
        
    
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    
    dist.destroy_process_group()


def train(model, model_name, loss_name='Focal', optimizer_name='Adam', scheduler_name='None'):
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
    set_seed(42)
    
    world_size = torch.cuda.device_count()
    
    os.environ['MASTER_ADDR'] = 'localhost'
    port = str(find_available_port())
    os.environ['MASTER_PORT'] = port
    os.environ['WORLD_SIZE'] = str(torch.cuda.device_count())
    
    experiment_time = datetime.now().strftime('%Y%m%d_%H:%M')
    experiment_dir = f"./result/{model_name}_{optimizer_name}_{loss_name}_{scheduler_name}_{experiment_time}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    main_log_file = os.path.join(experiment_dir, 'main.log')
    main_logger = setup_file_logging(main_log_file)
    
    main_logger.info(f"检测到 {world_size} 个GPU")
    
    for i in range(world_size):
        gpu_name = torch.cuda.get_device_name(i)
        memory = torch.cuda.get_device_properties(i).total_memory / 1e9
        main_logger.info(f"GPU {i}: {gpu_name} ({memory:.1f} GB)")
    
    main_logger.info("使用端口: " + port)
    main_logger.info("开始分布式训练...")
    
    for handler in main_logger.handlers[:]:
        handler.close()
        main_logger.removeHandler(handler)
    
    mp.spawn(
        main_worker,
        args=(world_size, model, loss_name, optimizer_name, scheduler_name, experiment_dir),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    model = UNet3D()
    model_name = 'UNet3D'
    
    train(model=model, model_name=model_name)