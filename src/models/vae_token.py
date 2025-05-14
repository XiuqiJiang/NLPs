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
        num_rnn_layers: int = 1,
        ring_embedding_dim: int = 32  # 添加环数嵌入维度
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
            ring_embedding_dim: 环数嵌入维度
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
        self.ring_embedding_dim = ring_embedding_dim
        
        # 构建编码器
        self.encoder = self.build_encoder()
        
        # 编码器输出层
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # 环数编码器
        self.ring_encoder = nn.Sequential(
            nn.Linear(1, ring_embedding_dim),
            nn.LayerNorm(ring_embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(ring_embedding_dim, ring_embedding_dim),
            nn.LayerNorm(ring_embedding_dim),
            nn.LeakyReLU()
        )
        
        # RNN解码器相关层
        self.latent_to_rnn_hidden = nn.Linear(latent_dim + ring_embedding_dim, rnn_hidden_dim * num_rnn_layers)
        self.decoder_embedding = nn.Embedding(vocab_size, rnn_hidden_dim)
        self.decoder_rnn = nn.GRU(
            input_size=rnn_hidden_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=num_rnn_layers,
            batch_first=True,
            dropout=dropout if num_rnn_layers > 1 else 0
        )
        self.fc_out = nn.Linear(rnn_hidden_dim, vocab_size)
        
        # 环数预测器
        self.ring_predictor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)  # 预测环数
        )
        
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
    
    def encode_ring_info(self, ring_info: torch.Tensor) -> torch.Tensor:
        """编码环数信息
        
        Args:
            ring_info: 环数张量 [batch_size]
            
        Returns:
            环数嵌入 [batch_size, ring_embedding_dim]
        """
        # 将环数转换为浮点数并添加维度
        ring_info = ring_info.float().unsqueeze(-1)  # [batch_size, 1]
        return self.ring_encoder(ring_info)
    
    def decode(self, z: torch.Tensor, target_ids: Optional[torch.Tensor] = None, ring_info: Optional[torch.Tensor] = None) -> torch.Tensor:
        """解码潜在向量
        
        Args:
            z: 潜在向量，形状为 [batch_size, latent_dim]
            target_ids: 目标token IDs (用于Teacher Forcing)，形状为 [batch_size, max_sequence_length]
            ring_info: 环数信息，形状为 [batch_size]
            
        Returns:
            如果是 Teacher Forcing，返回 logits: [batch_size, max_sequence_length, vocab_size]
            如果是 Autoregressive Generation，返回生成的 token IDs: [batch_size, max_sequence_length]
        """
        batch_size = z.size(0)
        device = z.device
        
        # 编码环数信息
        if ring_info is not None:
            ring_embedding = self.encode_ring_info(ring_info)
            # 将潜在向量和环数嵌入连接
            z = torch.cat([z, ring_embedding], dim=-1)
        
        # 准备RNN初始隐藏状态
        h0 = self.latent_to_rnn_hidden(z)  # [batch_size, rnn_hidden_dim * num_layers]
        h0 = h0.view(self.num_rnn_layers, batch_size, self.rnn_hidden_dim)
        
        if target_ids is not None:
            # 验证target_ids的维度
            if target_ids.dim() != 2:
                raise ValueError(f"target_ids必须是2维的 [batch_size, seq_len]，但收到了 {target_ids.shape}")
            
            # --- Teacher Forcing ---
            decoder_input_ids = target_ids[:, :-1]  # [batch_size, max_seq_len - 1]
            sos_tokens = torch.full((batch_size, 1), self.sos_token_id, device=device)
            decoder_input_ids = torch.cat([sos_tokens, decoder_input_ids], dim=1)
            
            decoder_inputs = self.decoder_embedding(decoder_input_ids)
            rnn_outputs, _ = self.decoder_rnn(decoder_inputs, h0)
            logits = self.fc_out(rnn_outputs)
            return logits
        else:
            # --- Autoregressive Generation ---
            output_sequence = torch.full(
                (batch_size, self.max_sequence_length),
                self.pad_token_id,
                device=device
            )
            
            # 设置起始token
            output_sequence[:, 0] = self.sos_token_id
            
            # 自回归生成
            for t in range(1, self.max_sequence_length):
                # 获取当前输入
                current_input = output_sequence[:, :t]
                decoder_inputs = self.decoder_embedding(current_input)
                
                # 运行RNN
                rnn_output, h0 = self.decoder_rnn(decoder_inputs, h0)
                
                # 获取最后一个时间步的输出
                last_output = rnn_output[:, -1]
                
                # 预测下一个token
                logits = self.fc_out(last_output)
                next_token = torch.argmax(logits, dim=-1)
                
                # 更新序列
                output_sequence[:, t] = next_token
                
                # 如果生成了结束token，提前结束
                if (next_token == self.eos_token_id).all():
                    break
            
            return output_sequence
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_ids: Optional[torch.Tensor] = None,
        ring_info: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, input_dim]
            attention_mask: 注意力掩码，形状为 [batch_size, seq_len]
            target_ids: 目标token IDs，形状为 [batch_size, max_sequence_length]
            ring_info: 环数信息，形状为 [batch_size]
            
        Returns:
            (重构logits, 输入token IDs, 均值, 对数方差, 环数预测)
        """
        # 编码
        mu, logvar = self.encode(x, attention_mask)
        z = self.reparameterize(mu, logvar)
        
        # 预测环数
        ring_pred = self.ring_predictor(z)
        
        # 解码
        if target_ids is not None:
            logits = self.decode(z, target_ids, ring_info)
            return logits, target_ids, mu, logvar, ring_pred
        else:
            generated_ids = self.decode(z, None, ring_info)
            return generated_ids, None, mu, logvar, ring_pred

def vae_token_loss(
    recon_logits: torch.Tensor,
    input_ids: torch.Tensor,
    mean: torch.Tensor,
    logvar: torch.Tensor,
    ring_pred: torch.Tensor,
    ring_info: torch.Tensor,
    epoch: int
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """计算VAE损失
    
    Args:
        recon_logits: 重构的logits，形状为 [batch_size, seq_len, vocab_size]
        input_ids: 输入token IDs，形状为 [batch_size, seq_len]
        mean: 均值，形状为 [batch_size, latent_dim]
        logvar: 对数方差，形状为 [batch_size, latent_dim]
        ring_pred: 预测的环数，形状为 [batch_size, 1]
        ring_info: 真实的环数，形状为 [batch_size]
        epoch: 当前训练轮次
        
    Returns:
        (总损失, 损失字典)
    """
    # 计算重构损失
    recon_loss = F.cross_entropy(
        recon_logits.view(-1, recon_logits.size(-1)),
        input_ids.view(-1),
        ignore_index=0  # 忽略padding token
    )
    
    # 计算KL散度
    kld_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    
    # 计算环数预测损失
    ring_loss = F.mse_loss(ring_pred.squeeze(-1), ring_info.float())
    
    # 获取beta值
    beta = get_beta(epoch)
    
    # 计算总损失
    total_loss = recon_loss + beta * kld_loss + 0.1 * ring_loss
    
    # 返回损失字典
    loss_dict = {
        'total_loss': total_loss.item(),
        'recon_loss': recon_loss.item(),
        'kld_loss': kld_loss.item(),
        'ring_loss': ring_loss.item(),
        'beta': beta
    }
    
    return total_loss, loss_dict 