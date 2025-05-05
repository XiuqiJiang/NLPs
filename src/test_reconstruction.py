import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import logging
from pathlib import Path
import torch.nn.functional as F

# 添加项目根目录到Python路径
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from src.models.vae_token import ESMVAEToken
from src.utils.data_utils import create_data_loaders
from config.config import (
    EMBEDDING_FILE,
    ESM_MODEL_PATH,
    BATCH_SIZE,
    TRAIN_TEST_SPLIT,
    NUM_WORKERS,
    PIN_MEMORY,
    MAX_SEQUENCE_LENGTH,
    MODEL_SAVE_DIR,
    DEVICE
)

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('reconstruction_test.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_model(model_path: str, device: str) -> ESMVAEToken:
    """加载训练好的模型
    
    Args:
        model_path: 模型文件路径
        device: 设备
        
    Returns:
        加载的模型
    """
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device)
    
    # 加载tokenizer以获取词汇表大小
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_PATH)
    vocab_size = tokenizer.vocab_size
    pad_token_id = tokenizer.pad_token_id
    
    # 初始化模型
    model = ESMVAEToken(
        input_dim=640,  # ESM-2的嵌入维度
        hidden_dims=[512, 256],
        latent_dim=256,
        vocab_size=vocab_size,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        pad_token_id=pad_token_id,
        use_layer_norm=True,
        dropout=0.1
    )
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, tokenizer

def reconstruct_sequences(
    model: ESMVAEToken,
    tokenizer: AutoTokenizer,
    data_loader: DataLoader,
    device: str,
    epoch: int,
    num_samples: int = 5
) -> None:
    """重构序列并比较
    
    Args:
        model: VAE模型
        tokenizer: tokenizer
        data_loader: 数据加载器
        device: 设备
        epoch: 当前epoch
        num_samples: 要重构的样本数量
    """
    logger = logging.getLogger(__name__)
    logger.info(f"\n{'='*50}")
    logger.info(f"Epoch {epoch} 的重构结果:")
    logger.info(f"{'='*50}")
    
    # 选择前num_samples个批次
    for i, batch in enumerate(data_loader):
        if i >= num_samples:
            break
            
        # 将数据移到设备
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # 获取原始序列
        original_token_ids = batch['token_ids']
        
        # 编码
        with torch.no_grad():
            # 使用编码器获取mu和logvar
            mu, logvar = model.encode(batch['embeddings'])
            
            # 直接使用mu进行重构（不使用重参数化）
            z = mu
            
            # 解码（使用自回归模式）
            reconstructed_token_ids = model.decode(z, target_ids=None)  # 直接获取生成的token IDs
        
        # 将token ids转换回序列
        for j in range(len(original_token_ids)):
            original_seq = tokenizer.decode(original_token_ids[j], skip_special_tokens=True)
            reconstructed_seq = tokenizer.decode(reconstructed_token_ids[j], skip_special_tokens=True)
            
            # 计算序列相似度
            similarity = sum(a == b for a, b in zip(original_seq, reconstructed_seq)) / len(original_seq)
            
            logger.info(f"\n样本 {i*BATCH_SIZE + j + 1}:")
            logger.info(f"原始序列: {original_seq}")
            logger.info(f"重构序列: {reconstructed_seq}")
            logger.info(f"序列相似度: {similarity:.2%}")

def generate_sequences(
    model: ESMVAEToken,
    tokenizer: AutoTokenizer,
    epoch: int,
    num_sequences: int = 10,
    device: str = DEVICE,
    temperature: float = 0.7,  # 添加温度参数，默认0.7
    top_k: int = 50,  # 添加top_k参数，默认50
    top_p: float = 0.9  # 添加top_p参数，默认0.9
) -> None:
    """生成新的序列，使用温度采样和Top-k/Top-p采样
    
    Args:
        model: VAE模型
        tokenizer: tokenizer
        epoch: 当前epoch
        num_sequences: 要生成的序列数量
        device: 设备
        temperature: 温度参数，控制采样的随机性
        top_k: 只考虑概率最高的k个token
        top_p: 累积概率阈值，只考虑累积概率达到p的token
    """
    logger = logging.getLogger(__name__)
    logger.info(f"\n{'='*50}")
    logger.info(f"Epoch {epoch} 的生成结果 (温度={temperature}, top_k={top_k}, top_p={top_p}):")
    logger.info(f"{'='*50}")
    
    # 生成随机潜在向量
    z = torch.randn(num_sequences, model.latent_dim, device=device)
    
    # 初始化输出序列
    batch_size = z.size(0)
    output_sequence = torch.full(
        (batch_size, model.max_sequence_length),
        model.pad_token_id,
        dtype=torch.long,
        device=device
    )
    
    # 初始化当前输入（SOS token）
    current_input = torch.full(
        (batch_size, 1),
        model.sos_token_id,
        dtype=torch.long,
        device=device
    )
    
    # 初始化RNN隐藏状态
    h0 = model.latent_to_rnn_hidden(z)
    h0 = h0.view(model.num_rnn_layers, batch_size, model.rnn_hidden_dim)
    hidden = h0
    
    # 跟踪每个序列是否已完成
    finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    with torch.no_grad():
        for t in range(model.max_sequence_length):
            # 跳过已经完成的序列
            if finished_sequences.all():
                break
                
            # 获取当前输入的嵌入
            decoder_input = model.decoder_embedding(current_input)
            
            # 通过RNN
            rnn_output, hidden = model.decoder_rnn(decoder_input, hidden)
            
            # 计算当前时间步的logits
            step_logits = model.fc_out(rnn_output.squeeze(1))  # (batch_size, vocab_size)
            
            # 应用温度采样
            if temperature != 1.0:
                step_logits = step_logits / temperature
            
            # 应用softmax
            probs = F.softmax(step_logits, dim=-1)
            
            # 应用Top-k采样
            if top_k > 0:
                top_k_probs, top_k_indices = torch.topk(probs, k=min(top_k, probs.size(-1)))
                probs = torch.zeros_like(probs).scatter_(-1, top_k_indices, top_k_probs)
                probs = probs / probs.sum(dim=-1, keepdim=True)
            
            # 应用Top-p采样
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # 创建掩码，只保留累积概率小于top_p的token
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # 将不需要的token概率设为0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                probs = probs.masked_fill(indices_to_remove, 0.0)
                probs = probs / probs.sum(dim=-1, keepdim=True)
            
            # 从处理后的概率分布中采样
            predicted_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
            # 只更新未完成的序列
            active_mask = ~finished_sequences
            output_sequence[active_mask, t] = predicted_tokens[active_mask]
            
            # 更新当前输入（只对未完成的序列）
            current_input = predicted_tokens.unsqueeze(1)
            
            # 检查是否生成了EOS token
            eos_mask = (predicted_tokens == model.eos_token_id)
            finished_sequences = finished_sequences | eos_mask
    
    # 将生成的token ids转换回序列
    for i in range(num_sequences):
        sequence = tokenizer.decode(output_sequence[i], skip_special_tokens=True)
        logger.info(f"\n生成的序列 {i+1}:")
        logger.info(sequence)
    
    logger.info("序列生成完成！")

def test_epoch(epoch: int, model_path: str, val_loader: DataLoader) -> None:
    """测试特定epoch的模型
    
    Args:
        epoch: 要测试的epoch
        model_path: 模型文件路径
        val_loader: 验证数据加载器
    """
    logger = logging.getLogger(__name__)
    logger.info(f"\n{'='*50}")
    logger.info(f"开始测试 Epoch {epoch} 的模型")
    logger.info(f"{'='*50}")
    
    # 加载模型
    model, tokenizer = load_model(model_path, DEVICE)
    
    # 重构序列
    reconstruct_sequences(
        model=model,
        tokenizer=tokenizer,
        data_loader=val_loader,
        device=DEVICE,
        epoch=epoch,
        num_samples=2
    )
    
    # 使用不同的采样参数生成序列
    logger.info("\n使用不同采样参数生成序列：")
    
    # 1. 使用较低温度
    generate_sequences(
        model=model,
        tokenizer=tokenizer,
        epoch=epoch,
        num_sequences=5,
        temperature=0.7,
        top_k=0,
        top_p=1.0
    )
    
    # 2. 使用Top-k采样
    generate_sequences(
        model=model,
        tokenizer=tokenizer,
        epoch=epoch,
        num_sequences=5,
        temperature=1.0,
        top_k=50,
        top_p=1.0
    )
    
    # 3. 使用Top-p采样
    generate_sequences(
        model=model,
        tokenizer=tokenizer,
        epoch=epoch,
        num_sequences=5,
        temperature=1.0,
        top_k=0,
        top_p=0.9
    )
    
    # 4. 组合使用温度和Top-p
    generate_sequences(
        model=model,
        tokenizer=tokenizer,
        epoch=epoch,
        num_sequences=5,
        temperature=0.7,
        top_k=0,
        top_p=0.9
    )

def main():
    """主函数"""
    logger = setup_logging()
    logger.info("开始重构测试...")
    
    # 创建数据加载器
    logger.info("创建数据加载器...")
    train_loader, val_loader = create_data_loaders(
        data_file=EMBEDDING_FILE,
        batch_size=BATCH_SIZE,
        train_test_split=TRAIN_TEST_SPLIT,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        max_sequence_length=MAX_SEQUENCE_LENGTH
    )
    
    # 测试不同epoch的模型
    epochs_to_test = [200, 400, 600, 800, 1000]
    for epoch in epochs_to_test:
        model_path = os.path.join(MODEL_SAVE_DIR, f'checkpoint_epoch_{epoch}.pt')
        if os.path.exists(model_path):
            test_epoch(epoch, model_path, val_loader)
        else:
            logger.warning(f"找不到 Epoch {epoch} 的模型文件: {model_path}")
    
    logger.info("测试完成！")

if __name__ == "__main__":
    main() 