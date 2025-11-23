import json
import os
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.7, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, input, target):
        # bce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        # prob = torch.exp(-bce_loss)
        # focal_loss = self.alpha * (1 - prob)**self.gamma * bce_loss
        # if self.reduction == 'mean':
        #     return focal_loss.mean()
        # elif self.reduction == 'sum':
        #     return focal_loss.sum()
        # else:
        #     return focal_loss
        bce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        p_t = torch.exp(-bce_loss)
        alpha_t = target * self.alpha + (1.0 - target) * (1.0 - self.alpha)
        focal_factor = (1.0 - p_t) ** self.gamma

        loss = alpha_t * focal_factor * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
    def forward(self, input, target, threshold=0.5):
        pred = torch.sigmoid(input)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice
        
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss
        
class FocalDiceLoss(nn.Module):
    def __init__(self, focal_alpha=0.7, focal_gamma=2.0, dice_smooth=1.0,
                 focal_weigth=0.7, reduction='mean'):
        super(FocalDiceLoss, self).__init__()
        self.focal_weight = focal_weigth
        self.dice_weight = 1 - focal_weigth
        self.reduction = reduction
        self.focal_loss = FocalLoss(focal_alpha, focal_gamma, reduction=reduction)
        self.dice_loss = DiceLoss(dice_smooth, reduction=reduction)
    
    def forward(self, input, target):
        return self.focal_weight * self.focal_loss(input, target) + self.dice_weight * self.dice_loss(input, target)
                
def calculate_metrics(input, target, threshold=0.5):
    
    pred = torch.sigmoid(input)
    pred_binary = (pred > threshold).float()
    
    target_binary = (target > threshold).float()
     
    epsilon = 1e-8
    # 混淆矩阵
    tp = (pred_binary * target_binary).sum()             # TP
    fp = (pred_binary * (1 - target_binary)).sum()       # FP
    tn = ((1 - pred_binary) * (1 - target_binary)).sum() # TN
    fn = ((1 - pred_binary) * target_binary).sum()       # FN
    
    precision = tp / (tp + fp + epsilon)    # 精确率
    far = 1 - precision                     # 误报率
    recall = tp / (tp + fn + epsilon)       # 召回率 / 检出率，pod
    fnr = 1 - recall                        # 漏报率
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    csi = tp / (tp + fp + fn + epsilon)
    
    return {
        'precision': precision.item(),
        'far': far.item(),
        'recall/pod': recall.item(),
        'fnr': fnr.item(),
        'f1': f1.item(),
        'csi': csi.item()
    }

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_file_logging(log_file_path):
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

import socket

def is_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return True
        except OSError:
            return False

def find_available_port(start_port=29550, max_attempts=400):
    for port in range(start_port, start_port + max_attempts):
        if is_port_available(port):
            return port
    return start_port

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}分{seconds:.1f}秒"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}时{int(minutes)}分{seconds:.1f}秒"
    
    
    
def evaluate_model(model, criterion, data_loader, device, threshold=0.5):

    model.eval()
    eval_loss = 0
    all_metrics = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
            images = images.to(device).float()
            labels = labels.to(device).float()
            target_labels = labels[:, -1]
            
            outputs = model(images)
            loss = criterion(outputs, target_labels)
            eval_loss += loss.item()
            
            metrics = calculate_metrics(outputs, target_labels, threshold)
            all_metrics.append(metrics)
            
    avg_eval_loss = eval_loss / len(data_loader)
    
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    return avg_eval_loss, avg_metrics