import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer
from typing import Tuple, Optional, List, Dict
from config.config import (
    ESM_MODEL_NAME,
    LATENT_DIM,
    HIDDEN_DIMS,
    ESM_EMBEDDING_DIM,
    get_beta,
    MAX_BETA,
    ANNEALING_EPOCHS,
    KLD_TARGET
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
        dropout: float = 0.1,
        rnn_hidden_dim: int = 256,
        num_rnn_layers: int = 1
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
            rnn_hidden_dim: RNN隐藏层维度
            num_rnn_layers: RNN层数
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pad_token_id = pad_token_id
        self.use_layer_norm = use_layer_norm
        self.rnn_hidden_dim = rnn_hidden_dim
        self.num_rnn_layers = num_rnn_layers
        
        # 构建编码器
        self.encoder = self.build_encoder()
        
        # 编码器输出层
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # RNN解码器相关层
        self.latent_to_rnn_hidden = nn.Linear(latent_dim, rnn_hidden_dim * num_rnn_layers)
        self.decoder_embedding = nn.Embedding(vocab_size, rnn_hidden_dim)
        self.decoder_rnn = nn.GRU(
            input_size=rnn_hidden_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=num_rnn_layers,
            batch_first=True,
            dropout=dropout if num_rnn_layers > 1 else 0
        )
        self.fc_out = nn.Linear(rnn_hidden_dim, vocab_size)
        
        # 特殊token IDs
        self.sos_token_id = 0  # 开始token
        self.eos_token_id = 2  # 结束token
        
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
    
    def decode(self, z: torch.Tensor, target_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """解码潜在向量
        
        Args:
            z: 潜在向量，形状为 [batch_size, latent_dim]
            target_ids: 目标token IDs (用于Teacher Forcing)，形状为 [batch_size, max_sequence_length]
            
        Returns:
            如果是 Teacher Forcing，返回 logits: [batch_size, max_sequence_length, vocab_size]
            如果是 Autoregressive Generation，返回生成的 token IDs: [batch_size, max_sequence_length]
        """
        batch_size = z.size(0)
        device = z.device # 获取设备

        # 准备RNN初始隐藏状态
        h0 = self.latent_to_rnn_hidden(z)  # [batch_size, rnn_hidden_dim * num_layers]
        h0 = h0.view(self.num_rnn_layers, batch_size, self.rnn_hidden_dim)
        
        if target_ids is not None:
            # --- Teacher Forcing (保持不变) ---
            decoder_input_ids = target_ids[:, :-1]  # [batch_size, max_seq_len - 1]
            sos_tokens = torch.full((batch_size, 1), self.sos_token_id, device=device)
            decoder_input_ids = torch.cat([sos_tokens, decoder_input_ids], dim=1)
            
            decoder_inputs = self.decoder_embedding(decoder_input_ids)
            rnn_outputs, _ = self.decoder_rnn(decoder_inputs, h0)
            logits = self.fc_out(rnn_outputs)
            return logits # 返回 logits 用于计算损失
        else:
            # --- Autoregressive Generation ---
            output_sequence = torch.full(
                (batch_size, self.max_sequence_length),
                self.pad_token_id,
                dtype=torch.long, # 确保是 long 类型
                device=device
            )
            
            current_input = torch.full(
                (batch_size, 1),
                self.sos_token_id,
                dtype=torch.long, # 确保是 long 类型
                device=device
            )
            
            hidden = h0
            
            # 跟踪每个序列是否已完成
            finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for t in range(self.max_sequence_length):
                # 跳过已经完成的序列
                if finished_sequences.all():
                    break
                    
                # 获取当前输入的嵌入
                decoder_input = self.decoder_embedding(current_input)
                
                # 通过RNN
                rnn_output, hidden = self.decoder_rnn(decoder_input, hidden)
                
                # 计算当前时间步的logits
                step_logits = self.fc_out(rnn_output.squeeze(1)) # (batch_size, vocab_size)
                
                # 获取预测的token
                predicted_tokens = torch.argmax(step_logits, dim=-1) # (batch_size,)
                
                # 只更新未完成的序列
                active_mask = ~finished_sequences
                output_sequence[active_mask, t] = predicted_tokens[active_mask]
                
                # 更新当前输入（只对未完成的序列）
                current_input = predicted_tokens.unsqueeze(1) # (batch_size, 1)
                
                # 检查是否生成了EOS token
                eos_mask = (predicted_tokens == self.eos_token_id)
                finished_sequences = finished_sequences | eos_mask
            
            return output_sequence # 直接返回生成的token IDs序列
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, input_dim]
            attention_mask: 注意力掩码，形状为 [batch_size, seq_len]
            target_ids: 目标token IDs，形状为 [batch_size, max_sequence_length]
            
        Returns:
            如果是 Teacher Forcing，返回 (logits, mu, logvar)
            如果是 Autoregressive Generation，返回 (token_ids, mu, logvar)
        """
        # 编码
        mu, logvar = self.encode(x, attention_mask)
        
        # 重参数化
        z = self.reparameterize(mu, logvar)
        
        # 解码
        output = self.decode(z, target_ids)
        
        return output, mu, logvar

def vae_token_loss(
    recon_logits: torch.Tensor,
    input_ids: torch.Tensor,
    mean: torch.Tensor,
    logvar: torch.Tensor,
    epoch: int,
    pad_token_id: int = 1,
    max_beta: float = MAX_BETA,  # 使用配置文件中的值
    annealing_epochs: int = ANNEALING_EPOCHS,  # 使用配置文件中的值
    kld_target: float = KLD_TARGET  # 使用配置文件中的值
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算VAE损失，使用Free Bits策略
    
    Args:
        recon_logits: 预测的logits，形状为(batch_size, sequence_length, vocab_size)
        input_ids: 真实的token ID，形状为(batch_size, sequence_length)
        mean: 潜在空间均值，形状为(batch_size, latent_dim)
        logvar: 潜在空间对数方差，形状为(batch_size, latent_dim)
        epoch: 当前epoch
        pad_token_id: padding token的ID
        max_beta: KL散度的最大权重
        annealing_epochs: beta退火的周期数
        kld_target: 目标KLD下限K，单位是nats
        
    Returns:
        总损失、重建损失和KL散度
    """
    # 计算当前epoch的beta值，使用传入的参数
    beta = get_beta(epoch, max_beta=max_beta, annealing_epochs=annealing_epochs)
    
    # 创建交叉熵损失函数，忽略padding token
    recon_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction='mean')
    
    # 计算重建损失
    # 将logits和targets展平为2D张量
    recon_loss = recon_loss_fn(
        recon_logits.view(-1, recon_logits.size(-1)),
        input_ids.view(-1)
    )
    
    # 计算KL散度
    kl_div_per_sample = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)
    kl_loss = torch.mean(kl_div_per_sample)
    
    # 直接使用原始KL损失
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss  # 返回原始kl_loss用于监控 