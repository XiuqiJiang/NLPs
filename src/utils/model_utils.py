import os
import torch
from typing import Dict, Any, Optional
from config.train_config import MODEL_SAVE_DIR

class EarlyStopping:
    """早停类
    
    监控验证损失，当损失不再改善时停止训练。
    """
    
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0,
        verbose: bool = False
    ) -> None:
        """初始化早停类
        
        Args:
            patience: 容忍的epoch数
            min_delta: 最小改善阈值
            verbose: 是否打印信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
    
    def __call__(self, val_loss: float) -> None:
        """检查是否需要早停
        
        Args:
            val_loss: 当前验证损失
        """
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    save_dir: str,
    is_best: bool = False
) -> None:
    """保存模型检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前epoch
        val_loss: 验证损失
        save_dir: 保存目录
        is_best: 是否是最佳模型
    """
    os.makedirs(save_dir, exist_ok=True)
    
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'val_loss': val_loss
    }
    
    # 保存当前检查点
    filename = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(state, filename)
    
    # 如果是最佳模型，额外保存
    if is_best:
        best_filename = os.path.join(save_dir, 'best_model.pth')
        torch.save(state, best_filename)

def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    checkpoint_path: str = None
) -> Dict[str, Any]:
    """加载模型检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        checkpoint_path: 检查点路径
        
    Returns:
        加载的状态字典
    """
    if checkpoint_path is None:
        checkpoint_path = os.path.join(MODEL_SAVE_DIR, 'best_model.pth')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    return {
        'epoch': checkpoint['epoch'],
        'val_loss': checkpoint['val_loss']
    } 