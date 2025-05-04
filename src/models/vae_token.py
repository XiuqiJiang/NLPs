import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer
from typing import Tuple, Optional, List, Dict
from config.config import (
    ESM_MODEL_NAME,
    LATENT_DIM,
    HIDDEN_DIMS,
    ESM_EMBEDDING_DIM
)

class ESMVAEToken(nn.Module):
    """基于ESM嵌入的VAE模型，输出token序列"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        latent_dim: int,
        vocab_size: int,
        max_sequence_length: int,
        pad_token_id: int,
        use_layer_norm: bool = True,
        dropout: float = 0.1
    ):
        """初始化VAE模型
        
        Args:
            input_dim: 输入维度（ESM嵌入维度）
            hidden_dims: 隐藏层维度列表
            latent_dim: 潜在空间维度
            vocab_size: 词汇表大小
            max_sequence_length: 最大序列长度
            pad_token_id: padding token的ID
            use_layer_norm: 是否使用LayerNorm
            dropout: dropout比率
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pad_token_id = pad_token_id
        self.use_layer_norm = use_layer_norm
        
        # 构建编码器和解码器
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        
        # 编码器输出层
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # 解码器输出层 - 使用hidden_dims[0]作为输入维度
        self.fc_out = nn.Linear(hidden_dims[0], vocab_size)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def build_encoder(self) -> nn.Module:
        """构建编码器网络"""
        layers = []
        in_dim = self.input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if self.use_layer_norm else nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(0.1)
            ])
            in_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def build_decoder(self) -> nn.Module:
        """构建解码器网络"""
        layers = []
        in_dim = self.latent_dim
        
        for hidden_dim in reversed(self.hidden_dims):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if self.use_layer_norm else nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(0.1)
            ])
            in_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def _init_weights(self, module):
        """初始化模型权重"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def encode(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码输入序列
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, input_dim]
            attention_mask: 注意力掩码，形状为 [batch_size, seq_len]
            
        Returns:
            mu: 均值，形状为 [batch_size, latent_dim]
            logvar: 对数方差，形状为 [batch_size, latent_dim]
        """
        # 验证输入形状
        if x.dim() != 3:
            raise ValueError(f"输入张量必须是3维的 [batch_size, seq_len, input_dim]，但收到了 {x.shape}")
        if x.size(-1) != self.input_dim:
            raise ValueError(f"输入维度必须是 {self.input_dim}，但收到了 {x.size(-1)}")
        
        # 验证注意力掩码
        if attention_mask is not None:
            if attention_mask.shape != x.shape[:2]:
                raise ValueError(f"注意力掩码形状 {attention_mask.shape} 与输入形状 {x.shape[:2]} 不匹配")
            if attention_mask.dtype != torch.long:
                raise ValueError(f"注意力掩码必须是长整型，但收到了 {attention_mask.dtype}")
        
        # 使用注意力掩码进行池化
        if attention_mask is not None:
            # 将掩码扩展到与输入相同的维度
            mask = attention_mask.unsqueeze(-1).float()
            # 计算掩码下的平均值
            x = (x * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # 添加小量避免除零
        else:
            # 如果没有掩码，直接取平均值
            x = x.mean(dim=1)
        
        # 通过编码器网络
        x = self.encoder(x)
        
        # 计算均值和方差
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧
        
        Args:
            mu: 均值
            logvar: 对数方差
            
        Returns:
            z: 采样的潜在向量
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """解码潜在向量
        
        Args:
            z: 潜在向量，形状为 [batch_size, latent_dim]
            
        Returns:
            logits: 输出logits，形状为 [batch_size, max_sequence_length, vocab_size]
        """
        # 验证输入形状
        if z.dim() != 2:
            raise ValueError(f"潜在向量必须是2维的 [batch_size, latent_dim]，但收到了 {z.shape}")
        if z.size(-1) != self.latent_dim:
            raise ValueError(f"潜在维度必须是 {self.latent_dim}，但收到了 {z.size(-1)}")
        
        # 通过解码器网络
        x = self.decoder(z)  # [batch_size, hidden_dims[-1]]
        
        # 扩展序列维度
        x = x.unsqueeze(1).expand(-1, self.max_sequence_length, -1)  # [batch_size, max_sequence_length, hidden_dims[-1]]
        
        # 通过输出层
        logits = self.fc_out(x)  # [batch_size, max_sequence_length, vocab_size]
        
        # 验证输出形状
        if logits.size(-1) != self.vocab_size:
            raise ValueError(f"输出维度必须是 {self.vocab_size}，但收到了 {logits.size(-1)}")
        if logits.size(1) != self.max_sequence_length:
            raise ValueError(f"序列长度必须是 {self.max_sequence_length}，但收到了 {logits.size(1)}")
        
        return logits
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, input_dim]
            attention_mask: 注意力掩码，形状为 [batch_size, seq_len]
            
        Returns:
            logits: 重建的logits，形状为 [batch_size, max_sequence_length, vocab_size]
            mu: 均值
            logvar: 对数方差
        """
        # 编码
        mu, logvar = self.encode(x, attention_mask)
        
        # 重参数化
        z = self.reparameterize(mu, logvar)
        
        # 解码
        logits = self.decode(z)
        
        return logits, mu, logvar

def vae_token_loss(
    recon_logits: torch.Tensor,
    input_ids: torch.Tensor,
    mean: torch.Tensor,
    logvar: torch.Tensor,
    epoch: int,  # 添加epoch参数
    pad_token_id: int = 1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算VAE损失
    
    Args:
        recon_logits: 预测的logits，形状为(batch_size, sequence_length, vocab_size)
        input_ids: 真实的token ID，形状为(batch_size, sequence_length)
        mean: 潜在空间均值
        logvar: 潜在空间对数方差
        epoch: 当前epoch
        pad_token_id: padding token的ID
        
    Returns:
        总损失、重建损失和KL散度
    """
    # 计算当前epoch的beta值
    beta = min(epoch/50.0, 1.0)
    
    # 重建损失 (使用交叉熵损失)
    # 将logits和targets展平为2D张量
    recon_loss = nn.CrossEntropyLoss(ignore_index=pad_token_id)(
        recon_logits.view(-1, recon_logits.size(-1)),
        input_ids.view(-1)
    )
    
    # KL散度
    kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    
    # 总损失
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss 