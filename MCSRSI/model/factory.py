from typing import Union, Dict, Any, Optional
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
import torch
import torch.nn as nn

class TrainingComponentFactory:
    
    # 支持的组件映射
    OPTIMIZERS = {
        'Adam', 'AdamW', 'SGD', 'RMSprop', 'Adagrad'
    }
    SCHEDULERS = {
        'linear', 'cosine', 'polynomial', 'MultiStep', 'ReduceLROnPlateau', 'CosineAnnealing'
    }
    LOSS_FUNCTIONS = {
        'BCE', 'CE', 'MSE', 'L1', 'SmoothL1', 'Focal', 'Dice', 'FocalDice'
    }
    
    @staticmethod
    def build_optimizer(optimizer_name:str, model:nn.Module, lr:float=1e-3, 
                       weight_decay:float=0.0, **kwargs) -> torch.optim.Optimizer:

        optimizer_name = optimizer_name.lower()
        
        if optimizer_name == 'adam':
            return torch.optim.Adam(
                model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay,
                betas=kwargs.get('betas', (0.9, 0.999)),
                eps=kwargs.get('eps', 1e-8)
            )
        elif optimizer_name == 'adamw':
            return torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=kwargs.get('betas', (0.9, 0.999))
            )
        elif optimizer_name == 'sgd':
            return torch.optim.SGD(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=kwargs.get('momentum', 0.9),
                nesterov=kwargs.get('nesterov', True)
            )
        elif optimizer_name == 'rmsprop':
            return torch.optim.RMSprop(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=kwargs.get('momentum', 0.9)
            )
        elif optimizer_name == 'adagrad':
            return torch.optim.Adagrad(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"不支持的优化器: {optimizer_name}。"
                          f"支持的优化器: {list(TrainingComponentFactory.OPTIMIZERS)}")
    
    @staticmethod
    def build_scheduler(scheduler_name:str, optimizer:torch.optim.Optimizer,
                       num_warmup_steps: Optional[int]=None,
                       num_training_steps: Optional[int]=None,
                       **kwargs) -> torch.optim.lr_scheduler._LRScheduler:

        scheduler_name = scheduler_name.lower()
        
        if scheduler_name == 'linear':
            return get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps, num_training_steps
            )
        elif scheduler_name == 'cosine':
            return get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps, num_training_steps,
                num_cycles=kwargs.get('num_cycles', 0.5)
            )
        elif scheduler_name == 'polynomial':
            return get_polynomial_decay_schedule_with_warmup(
                optimizer, num_warmup_steps, num_training_steps,
                power=kwargs.get('power', 1.0)
            )
        elif scheduler_name == 'multistep':
            return torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=kwargs.get('milestones', [10, 45, 85]),
                gamma=kwargs.get('gamma', 0.1)
            )
        elif scheduler_name == 'reducelronplateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=kwargs.get('patience', 5),
                factor=kwargs.get('factor', 0.1),
                mode=kwargs.get('mode', 'min')
            )
        elif scheduler_name == 'cosineannealing':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=kwargs.get('T_max', num_training_steps or 100)
            )
        elif scheduler_name == 'none':
            return None
        else:
            raise ValueError(f"不支持的调度器: {scheduler_name}。"
                          f"支持的调度器: {list(TrainingComponentFactory.SCHEDULERS)}")
    
    @staticmethod
    def build_loss_fn(loss_name: str, **kwargs) -> nn.Module:

        loss_name = loss_name.upper()
        
        if loss_name == 'BCE':
            return nn.BCEWithLogitsLoss(
                pos_weight=kwargs.get('pos_weight'),
                reduction=kwargs.get('reduction', 'mean')
            )
        elif loss_name == 'CE':
            return nn.CrossEntropyLoss(
                weight=kwargs.get('weight'),
                ignore_index=kwargs.get('ignore_index', -100)
            )
        elif loss_name == 'MSE':
            return nn.MSELoss(reduction=kwargs.get('reduction', 'mean'))
        elif loss_name == 'L1':
            return nn.L1Loss(reduction=kwargs.get('reduction', 'mean'))
        elif loss_name == 'SMOOTHL1':
            return nn.SmoothL1Loss(reduction=kwargs.get('reduction', 'mean'))
        elif loss_name == 'FOCAL':
            from utils.utils import FocalLoss
            return FocalLoss(
                alpha=kwargs.get('alpha', 0.7),
                gamma=kwargs.get('gamma', 2.0),
                reduction=kwargs.get('reduction', 'mean')
            )
        elif loss_name == 'DICE':
            from utils.utils import DiceLoss
            return DiceLoss(
                smooth=kwargs.get('smooth', 1.0),
                reduction=kwargs.get('reduction', 'mean')
            )
        elif loss_name == 'FOCALDICE':
            from utils.utils import FocalDiceLoss
            return FocalDiceLoss(
                focal_alpha=kwargs.get('focal_alpha', 0.7),
                focal_gamma=kwargs.get('focal_gamma', 2.0),
                dice_smooth=kwargs.get('dice_smooth', 1.0),
                focal_weigth=kwargs.get('focal_weigth', 0.7),
                reduction=kwargs.get('reduction', 'mean')
            )
        else:
            raise ValueError(f"不支持的损失函数: {loss_name}。"
                          f"支持的损失函数: {list(TrainingComponentFactory.LOSS_FUNCTIONS)}")
    
    @staticmethod
    def get_available_components() -> Dict[str, list]:
        return {
            'optimizers': list(TrainingComponentFactory.OPTIMIZERS),
            'schedulers': list(TrainingComponentFactory.SCHEDULERS),
            'loss_functions': list(TrainingComponentFactory.LOSS_FUNCTIONS)
        }