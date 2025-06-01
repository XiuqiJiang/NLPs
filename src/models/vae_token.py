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
    KLD_TARGET,
    KLD_TARGET_WEIGHT,
    BETA_FREEZE_EPOCHS
)

RING_LOSS_WEIGHT = 0  # 只训练重构，不加环数损失

class ESMVAEToken(nn.Module):
    """基于ESM嵌入的VAE模型，输出token序列"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        vocab_size: int,
        max_sequence_length: int,
        pad_token_id: int,
        latent_dim: int = 128,
        use_layer_norm: bool = True,
        dropout: float = 0.5,
        rnn_hidden_dim: int = 256,
        num_rnn_layers: int = 1,
        ring_embedding_dim: int = 128,  # 条件embedding维度加大
        num_classes: int = 3
    ):
        """初始化VAE模型
        
        Args:
            input_dim: 输入维度（ESM嵌入维度）
            hidden_dims: 隐藏层维度列表
            vocab_size: 词汇表大小
            max_sequence_length: 最大序列长度
            pad_token_id: padding token的ID
            latent_dim: 潜在空间维度
            use_layer_norm: 是否使用LayerNorm
            dropout: dropout比率
            rnn_hidden_dim: RNN隐藏层维度
            num_rnn_layers: RNN层数
            ring_embedding_dim: 环数嵌入维度
            num_classes: 环数分类类别数（0-8个C，共9类）
        """
        super().__init__()
        self.dropout = dropout  # 先定义dropout参数，供build_encoder使用
        self.token_dropout_rate = 0.0  # 关闭Token Dropout
        self.unk_token_id = getattr(self, 'unk_token_id', 3)  # 默认3为<UNK>，如有特殊定义可调整
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
        self.num_classes = num_classes
        
        # 构建编码器
        self.encoder = self.build_encoder()
        
        # 编码器输出层
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # 环数编码器：使用Embedding层直接编码类别
        self.ring_encoder = nn.Embedding(num_classes, ring_embedding_dim)
        
        # RNN解码器相关层
        self.latent_to_rnn_hidden = nn.Linear(latent_dim + ring_embedding_dim, rnn_hidden_dim * num_rnn_layers)
        self.decoder_embedding = nn.Embedding(vocab_size, rnn_hidden_dim)
        self.decoder_rnn = nn.GRU(
            input_size=rnn_hidden_dim + ring_embedding_dim,  # 输入维度是embedding_dim + ring_embedding_dim
            hidden_size=rnn_hidden_dim,
            num_layers=num_rnn_layers,
            batch_first=True,
            dropout=0.2  # 降低RNN层Dropout
        )
        self.decoder_dropout = nn.Dropout(0.2)  # 解码器输出Dropout降低
        self.fc_out = nn.Linear(rnn_hidden_dim, vocab_size)
        
        # 环数预测器：极简单层
        self.ring_dropout = nn.Dropout(0.2)  # 环数预测头Dropout降低
        self.ring_predictor = nn.Linear(latent_dim, num_classes)
        
        # 特殊token IDs
        self.sos_token_id = 0  # 开始token
        self.eos_token_id = 2  # 结束token
        
        # 初始化权重
        self.apply(self._init_weights)
        
        # 打印模型配置
        print(f"Model configuration:")
        print(f"- Input dimension: {input_dim}")
        print(f"- Hidden dimensions: {hidden_dims}")
        print(f"- Latent dimension: {latent_dim}")
        print(f"- Vocabulary size: {vocab_size}")
        print(f"- RNN hidden dimension: {rnn_hidden_dim}")
        print(f"- Ring embedding dimension: {ring_embedding_dim}")
        print(f"- Number of classes: {num_classes}")
    
    def build_encoder(self) -> nn.Module:
        """构建编码器网络"""
        layers = []
        in_dim = self.input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if self.use_layer_norm else nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(0.2)  # 编码器MLP Dropout降低
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
            ring_info: 环数张量 [batch_size]，值为0到num_classes-1
            
        Returns:
            环数嵌入 [batch_size, ring_embedding_dim]
        """
        # 确保ring_info是long类型
        ring_info = ring_info.long()
        return self.ring_encoder(ring_info)
    
    def decode(self, z: torch.Tensor, target_ids: Optional[torch.Tensor] = None, ring_info: Optional[torch.Tensor] = None, gumbel_softmax: bool = False) -> torch.Tensor:
        """解码潜在向量
        
        Args:
            z: 潜在向量，形状为 [batch_size, latent_dim]
            target_ids: 目标token IDs (用于Teacher Forcing)，形状为 [batch_size, max_sequence_length]
            ring_info: 环数信息，形状为 [batch_size]
            gumbel_softmax: 是否使用Gumbel-Softmax采样
            
        Returns:
            如果是 Teacher Forcing，返回 logits: [batch_size, max_sequence_length, vocab_size]
            如果是 Autoregressive Generation，返回生成的 token IDs: [batch_size, max_sequence_length]
        """
        batch_size = z.size(0)
        device = z.device
        
        # 编码环数信息
        if ring_info is not None:
            if ring_info.device != z.device:
                ring_info = ring_info.to(z.device)
            # 确保ring_info是long类型
            ring_info = ring_info.long()
            # 检查ring_info的值范围
            if ring_info.min() < 0 or ring_info.max() >= self.num_classes:
                raise ValueError(f"ring_info的值必须在[0, {self.num_classes-1}]范围内，但实际范围是[{ring_info.min()}, {ring_info.max()}]")
            ring_embedding = self.ring_encoder(ring_info)  # [batch, ring_emb_dim]
        else:
            ring_embedding = torch.zeros(z.size(0), self.ring_embedding_dim, device=z.device)
        
        # 拼接z和ring_embedding初始化RNN隐藏状态
        try:
            z = torch.cat([z, ring_embedding], dim=-1)
        except RuntimeError as e:
            raise
        
        h0 = self.latent_to_rnn_hidden(z)
        h0 = h0.view(self.num_rnn_layers, batch_size, self.rnn_hidden_dim)
        
        if target_ids is not None:
            # Teacher Forcing
            # 确保序列长度一致
            seq_length = target_ids.size(1)
            if seq_length > self.max_sequence_length:
                target_ids = target_ids[:, :self.max_sequence_length]
                seq_length = self.max_sequence_length
            
            decoder_input_ids = target_ids[:, :-1]
            sos_tokens = torch.full((batch_size, 1), self.sos_token_id, device=device)
            decoder_input_ids = torch.cat([sos_tokens, decoder_input_ids], dim=1)
            # Token Dropout/Word Dropout：训练时以一定概率将输入token替换为<UNK>
            if self.training and self.token_dropout_rate > 0:
                mask = torch.rand(decoder_input_ids.shape, device=decoder_input_ids.device) < self.token_dropout_rate
                decoder_input_ids = decoder_input_ids.clone()
                decoder_input_ids[mask] = self.unk_token_id
            decoder_inputs = self.decoder_embedding(decoder_input_ids)  # [batch, seq_len, emb_dim]
            
            cond_emb = ring_embedding.unsqueeze(1).repeat(1, decoder_inputs.size(1), 1)  # [batch, seq_len, ring_emb_dim]
            
            decoder_inputs = torch.cat([decoder_inputs, cond_emb], dim=-1)  # [batch, seq_len, emb+cond]
            
            try:
                # 使用pack_padded_sequence处理变长序列
                rnn_outputs, _ = self.decoder_rnn(decoder_inputs, h0)
                
                logits = self.fc_out(self.decoder_dropout(rnn_outputs))
                return logits
            except RuntimeError as e:
                raise
        else:
            # Autoregressive Generation
            output_sequence = torch.full(
                (batch_size, self.max_sequence_length),
                self.pad_token_id,
                device=device
            )
            output_sequence[:, 0] = self.sos_token_id
            for t in range(1, self.max_sequence_length):
                current_input = output_sequence[:, :t]
                decoder_inputs = self.decoder_embedding(current_input)
                cond_emb = ring_embedding.unsqueeze(1).repeat(1, decoder_inputs.size(1), 1)
                decoder_inputs = torch.cat([decoder_inputs, cond_emb], dim=-1)
                try:
                    rnn_output, h0 = self.decoder_rnn(decoder_inputs, h0)
                    last_output = rnn_output[:, -1]
                    logits = self.fc_out(self.decoder_dropout(last_output))
                    # 默认greedy采样（argmax），只有gumbel_softmax=True时才用Gumbel-Softmax
                    if gumbel_softmax:
                        next_token = torch.nn.functional.gumbel_softmax(logits, tau=1.0, hard=True).argmax(dim=-1)
                    else:
                        next_token = torch.argmax(logits, dim=-1)
                    output_sequence[:, t] = next_token
                    if (next_token == self.eos_token_id).all():
                        break
                except RuntimeError as e:
                    raise
            return output_sequence
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_ids: Optional[torch.Tensor] = None,
        ring_info: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, input_dim]
            attention_mask: 注意力掩码，形状为 [batch_size, seq_len]
            target_ids: 目标token IDs，形状为 [batch_size, max_sequence_length]
            ring_info: 环数信息，形状为 [batch_size]，可为None
            
        Returns:
            (重构logits/生成token IDs, 输入token IDs, 均值, 对数方差, 环数预测/None)
        """
        # 编码
        mu, logvar = self.encode(x, attention_mask)
        z = self.reparameterize(mu, logvar)
        # 预测环数（仅有条件时）
        ring_pred = self.ring_predictor(self.ring_dropout(mu)) if ring_info is not None else None
        # 解码
        if target_ids is not None:
            logits = self.decode(z, target_ids, ring_info)
            return logits, target_ids, mu, logvar, ring_pred
        else:
            generated_ids = self.decode(z, None, ring_info)
            return generated_ids, None, mu, logvar, ring_pred

    def sample_from_z(self, z, ring_info, temperature=1.0, max_length=None):
        """
        从给定z和条件ring_info采样生成序列，支持temperature softmax采样。
        """
        self.eval()
        batch_size = z.size(0)
        device = z.device
        max_length = max_length or self.max_sequence_length

        # 编码环数信息
        if ring_info.device != z.device:
            ring_info = ring_info.to(z.device)
        ring_info = ring_info.long()
        ring_embedding = self.ring_encoder(ring_info)

        # 拼接z和ring_embedding初始化RNN隐藏状态
        z_cat = torch.cat([z, ring_embedding], dim=-1)
        h0 = self.latent_to_rnn_hidden(z_cat)
        h0 = h0.view(self.num_rnn_layers, batch_size, self.rnn_hidden_dim)

        output_sequence = torch.full((batch_size, max_length), self.pad_token_id, device=device)
        output_sequence[:, 0] = self.sos_token_id

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for t in range(1, max_length):
            current_input = output_sequence[:, :t]
            decoder_inputs = self.decoder_embedding(current_input)
            cond_emb = ring_embedding.unsqueeze(1).repeat(1, decoder_inputs.size(1), 1)
            decoder_inputs = torch.cat([decoder_inputs, cond_emb], dim=-1)
            rnn_output, h0 = self.decoder_rnn(decoder_inputs, h0)
            last_output = rnn_output[:, -1]
            logits = self.fc_out(self.decoder_dropout(last_output))
            if temperature is not None and abs(temperature) < 1e-8:
                # 贪婪解码
                next_token = torch.argmax(logits, dim=-1)
            else:
                probs = torch.softmax(logits / (temperature if temperature > 0 else 1e-8), dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            # 对已完成序列，强制填充pad
            next_token = torch.where(finished, torch.full_like(next_token, self.pad_token_id), next_token)
            output_sequence[:, t] = next_token
            # 更新finished标记
            finished = finished | (next_token == self.eos_token_id)
            if finished.all():
                break
        return output_sequence

def vae_token_loss(
    recon_logits: torch.Tensor,  # [batch_size, seq_len, vocab_size]
    input_ids: torch.Tensor,     # [batch_size, seq_len]
    mu: torch.Tensor,            # [batch_size, latent_dim]
    logvar: torch.Tensor,        # [batch_size, latent_dim]
    ring_pred: torch.Tensor,     # [batch_size]
    ring_info: torch.Tensor,     # [batch_size]
    epoch: int,
    pad_token_id: int = 1,
    max_beta: float = 1.0,
    annealing_epochs: int = 10
) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, float, float, float]:
    """只优化重构损失和环数损失，KLD仅做日志输出"""
    device = recon_logits.device
    input_ids = input_ids.to(device)
    mu = mu.to(device)
    logvar = logvar.to(device)
    ring_pred = ring_pred.to(device)
    ring_info = ring_info.to(device)
    
    # 计算重构损失
    recon_loss = F.cross_entropy(
        recon_logits.view(-1, recon_logits.size(-1)),
        input_ids.view(-1),
        ignore_index=pad_token_id
    )
    
    # 计算KL散度（仅日志用，不参与loss）
    kld_element = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # [batch, latent_dim]
    kld_element = kld_element.mean(dim=0)  # [latent_dim]
    free_bits = 0.02  # 每维最小KL
    kld_element = torch.clamp(kld_element, min=free_bits)
    kld_loss = kld_element.sum()  # 总KL
    kld_raw_mean = kld_loss.detach().item() if isinstance(kld_loss, torch.Tensor) else float(kld_loss)
    free_bits_kld_mean = kld_element.mean().item() if isinstance(kld_element, torch.Tensor) else float(kld_element.mean())
    
    # 计算环数预测损失
    ring_loss = F.cross_entropy(ring_pred, ring_info.long())
    
    # 总损失只包含重构和环数损失
    total_loss = recon_loss + RING_LOSS_WEIGHT * ring_loss
    
    # beta和kld_target只做日志输出
    beta = 0.0
    kld_target = KLD_TARGET
    
    return total_loss, recon_loss, beta, ring_loss, kld_target, free_bits_kld_mean, kld_raw_mean 