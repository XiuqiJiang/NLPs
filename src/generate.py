import os
import sys
import torch
import logging
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse

# 添加项目根目录到Python路径
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from src.models.vae_token import ESMVAEToken
from config.config import (
    ESM_MODEL_PATH,
    MODEL_SAVE_DIR,
    DEVICE,
    MAX_SEQUENCE_LENGTH,
    GENERATED_SEQUENCES_PATH,
    ESM_EMBEDDING_DIM,
    HIDDEN_DIMS,
    LATENT_DIM
)

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_model(model_path: str = None) -> tuple[ESMVAEToken, AutoTokenizer]:
    """加载模型
    
    Args:
        model_path: 模型路径，如果为None则使用最佳模型
        
    Returns:
        (模型, tokenizer)
    """
    if model_path is None:
        model_path = os.path.join(MODEL_SAVE_DIR, 'best_model.pt')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到最佳模型文件: {model_path}")
    
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_PATH, local_files_only=True)
    
    # 初始化模型
    model = ESMVAEToken(
        input_dim=ESM_EMBEDDING_DIM,
        hidden_dims=HIDDEN_DIMS,
        latent_dim=LATENT_DIM,
        vocab_size=tokenizer.vocab_size,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        pad_token_id=tokenizer.pad_token_id,
        use_layer_norm=True,
        dropout=0.1
    ).to(DEVICE)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, tokenizer

def generate_sequences(
    model: ESMVAEToken,
    tokenizer: AutoTokenizer,
    num_sequences: int = 10,
    device: str = DEVICE,
    save_path: str = GENERATED_SEQUENCES_PATH
) -> list[str]:
    """生成新的序列并保存
    
    Args:
        model: VAE模型
        tokenizer: tokenizer
        num_sequences: 要生成的序列数量
        device: 设备
        save_path: 保存路径
        
    Returns:
        生成的序列列表
    """
    logger = logging.getLogger(__name__)
    logger.info(f"开始生成 {num_sequences} 个新序列...")
    
    # 检查tokenizer的特殊token
    logger.info("检查tokenizer的特殊token...")
    logger.info(f"BOS Token ID: {tokenizer.bos_token_id}")
    logger.info(f"EOS Token ID: {tokenizer.eos_token_id}")
    logger.info(f"CLS Token ID: {tokenizer.cls_token_id}")
    logger.info(f"PAD Token ID: {tokenizer.pad_token_id}")
    
    # 选择起始token
    if tokenizer.cls_token_id is not None:
        logger.info("使用CLS token作为起始token")
        start_token_id = tokenizer.cls_token_id
    elif tokenizer.bos_token_id is not None:
        logger.info("使用BOS token作为起始token")
        start_token_id = tokenizer.bos_token_id
    else:
        logger.warning("没有找到合适的起始token，将使用PAD token")
        start_token_id = tokenizer.pad_token_id
    
    # 选择结束token
    if tokenizer.eos_token_id is not None:
        logger.info("使用EOS token作为结束token")
        end_token_id = tokenizer.eos_token_id
    else:
        logger.warning("没有找到结束token，将使用PAD token")
        end_token_id = tokenizer.pad_token_id
    
    # 生成随机潜在向量
    z = torch.randn(num_sequences, model.latent_dim, device=device)
    
    # 解码（使用自回归模式）
    with torch.no_grad():
        generated_token_ids = model.decode(z, target_ids=None)  # 直接获取生成的token IDs
    
    # 将生成的token ids转换回序列
    generated_sequences = []
    for i in range(num_sequences):
        sequence = tokenizer.decode(generated_token_ids[i], skip_special_tokens=True)
        generated_sequences.append(sequence)
        logger.info(f"\n生成的序列 {i+1}:")
        logger.info(sequence)
    
    # 保存生成的序列
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        for seq in generated_sequences:
            f.write(seq + '\n')
    
    logger.info(f"序列已保存到 {save_path}")
    logger.info("序列生成完成！")
    
    return generated_sequences

def generate_with_temperature(
    model: ESMVAEToken,
    tokenizer: AutoTokenizer,
    num_sequences: int = 10,
    temperature: float = 1.0,
    device: str = DEVICE,
    save_path: str = GENERATED_SEQUENCES_PATH
) -> list[str]:
    """使用温度采样生成序列
    
    Args:
        model: VAE模型
        tokenizer: tokenizer
        num_sequences: 要生成的序列数量
        temperature: 温度参数，控制采样的随机性
        device: 设备
        save_path: 保存路径
        
    Returns:
        生成的序列列表
    """
    logger = logging.getLogger(__name__)
    logger.info(f"开始使用温度 {temperature} 生成 {num_sequences} 个新序列...")
    
    # 检查tokenizer的特殊token
    logger.info("检查tokenizer的特殊token...")
    logger.info(f"BOS Token ID: {tokenizer.bos_token_id}")
    logger.info(f"EOS Token ID: {tokenizer.eos_token_id}")
    logger.info(f"CLS Token ID: {tokenizer.cls_token_id}")
    logger.info(f"PAD Token ID: {tokenizer.pad_token_id}")
    
    # 选择起始token
    if tokenizer.cls_token_id is not None:
        logger.info("使用CLS token作为起始token")
        start_token_id = tokenizer.cls_token_id
    elif tokenizer.bos_token_id is not None:
        logger.info("使用BOS token作为起始token")
        start_token_id = tokenizer.bos_token_id
    else:
        logger.warning("没有找到合适的起始token，将使用PAD token")
        start_token_id = tokenizer.pad_token_id
    
    # 选择结束token
    if tokenizer.eos_token_id is not None:
        logger.info("使用EOS token作为结束token")
        end_token_id = tokenizer.eos_token_id
    else:
        logger.warning("没有找到结束token，将使用PAD token")
        end_token_id = tokenizer.pad_token_id
    
    # 生成随机潜在向量
    z = torch.randn(num_sequences, model.latent_dim, device=device)
    
    # 初始化生成序列
    generated_sequences = []
    current_token_ids = torch.full((num_sequences, 1), start_token_id, dtype=torch.long, device=device)
    
    # 逐步生成序列
    with torch.no_grad():
        for _ in range(MAX_SEQUENCE_LENGTH):
            # 获取当前时间步的logits
            logits = model.decode(z, current_token_ids)  # 使用teacher forcing模式获取logits
            
            # 应用温度采样
            logits = logits[:, -1:] / temperature
            probs = torch.softmax(logits, dim=-1)
            
            # 为每个序列采样下一个token
            next_token_ids = torch.multinomial(probs.squeeze(1), 1)
            
            # 将新token添加到序列中
            current_token_ids = torch.cat([current_token_ids, next_token_ids], dim=1)
            
            # 检查是否生成了结束符
            if (next_token_ids == end_token_id).any():
                break
    
    # 将生成的token ids转换回序列
    for i in range(num_sequences):
        sequence = tokenizer.decode(current_token_ids[i], skip_special_tokens=True)
        generated_sequences.append(sequence)
        logger.info(f"\n生成的序列 {i+1}:")
        logger.info(sequence)
    
    # 保存生成的序列
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        for seq in generated_sequences:
            f.write(seq + '\n')
    
    logger.info(f"序列已保存到 {save_path}")
    logger.info("序列生成完成！")
    
    return generated_sequences

def generate_sequences_with_rings(
    model: ESMVAEToken,
    tokenizer: AutoTokenizer,
    num_sequences: int = 10,
    num_rings: int = 3,
    device: str = DEVICE,
    save_path: str = GENERATED_SEQUENCES_PATH
) -> list[str]:
    """生成指定环数的新序列并保存
    
    Args:
        model: VAE模型
        tokenizer: tokenizer
        num_sequences: 要生成的序列数量
        num_rings: 每个序列的环数（半胱氨酸数量）
        device: 设备
        save_path: 保存路径
        
    Returns:
        生成的序列列表
    """
    logger = logging.getLogger(__name__)
    logger.info(f"开始生成 {num_sequences} 个具有 {num_rings} 个环的新序列...")
    
    # 加载训练数据中的三环序列的潜在向量
    logger.info("加载训练数据中的三环序列...")
    from src.data.dataset import ProteinDataset
    from torch.utils.data import DataLoader
    
    # 加载数据集
    dataset = ProteinDataset(
        data_path=os.path.join(root_dir, 'data', 'processed', 'train.csv'),
        tokenizer=tokenizer,
        max_length=MAX_SEQUENCE_LENGTH
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # 收集三环序列的潜在向量
    three_ring_vectors = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            # 获取潜在向量
            mu, _ = model.encode(batch['embeddings'])
            
            # 检查每个序列的半胱氨酸数量
            for i, seq in enumerate(batch['input_ids']):
                sequence = tokenizer.decode(seq, skip_special_tokens=True)
                if sequence.count('C') == num_rings:
                    three_ring_vectors.append(mu[i].cpu())
    
    if not three_ring_vectors:
        logger.warning(f"在训练数据中没有找到具有 {num_rings} 个环的序列，将使用随机采样")
        z = torch.randn(num_sequences, model.latent_dim, device=device)
    else:
        # 计算三环序列潜在向量的均值和标准差
        three_ring_vectors = torch.stack(three_ring_vectors)
        mean = three_ring_vectors.mean(dim=0)
        std = three_ring_vectors.std(dim=0)
        
        # 在三环序列的潜在空间区域采样
        logger.info(f"在三环序列的潜在空间区域采样 (均值: {mean.mean():.3f}, 标准差: {std.mean():.3f})")
        z = mean + std * torch.randn(num_sequences, model.latent_dim, device=device)
    
    # 解码（使用自回归模式）
    with torch.no_grad():
        generated_token_ids = model.decode(z, target_ids=None)
    
    # 将生成的token ids转换回序列
    valid_sequences = []
    
    for i in range(num_sequences):
        sequence = tokenizer.decode(generated_token_ids[i], skip_special_tokens=True)
        cysteine_count = sequence.count('C')
        if cysteine_count == num_rings:
            valid_sequences.append(sequence)
            logger.info(f"\n生成的序列 {len(valid_sequences)}:")
            logger.info(sequence)
            logger.info(f"环数: {cysteine_count}")
    
    # 如果有效序列不足，继续生成直到达到要求
    while len(valid_sequences) < num_sequences:
        # 在三环序列的潜在空间区域采样
        if three_ring_vectors:
            z = mean + std * torch.randn(num_sequences, model.latent_dim, device=device)
        else:
            z = torch.randn(num_sequences, model.latent_dim, device=device)
        
        # 解码
        with torch.no_grad():
            generated_token_ids = model.decode(z, target_ids=None)
        
        # 检查新生成的序列
        for i in range(num_sequences):
            sequence = tokenizer.decode(generated_token_ids[i], skip_special_tokens=True)
            cysteine_count = sequence.count('C')
            if cysteine_count == num_rings and sequence not in valid_sequences:
                valid_sequences.append(sequence)
                logger.info(f"\n生成的序列 {len(valid_sequences)}:")
                logger.info(sequence)
                logger.info(f"环数: {cysteine_count}")
                
                if len(valid_sequences) >= num_sequences:
                    break
    
    # 保存生成的序列
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        for seq in valid_sequences:
            f.write(seq + '\n')
    
    logger.info(f"序列已保存到 {save_path}")
    logger.info("序列生成完成！")
    
    return valid_sequences

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='生成蛋白质序列')
    parser.add_argument('--num_sequences', type=int, default=5,
                      help='要生成的序列数量')
    parser.add_argument('--num_rings', type=int, default=3,
                      help='每个序列的环数（半胱氨酸数量）')
    parser.add_argument('--temperature', type=float, default=1.0,
                      help='采样温度')
    args = parser.parse_args()
    
    logger = setup_logging()
    logger.info("开始生成序列...")
    
    # 加载模型
    model_path = os.path.join(MODEL_SAVE_DIR, 'best_model.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型文件: {model_path}")
    
    logger.info(f"加载模型: {model_path}")
    model, tokenizer = load_model(model_path)
    
    # 生成指定环数的序列
    logger.info(f"\n生成具有 {args.num_rings} 个环的序列...")
    sequences = generate_sequences_with_rings(
        model=model,
        tokenizer=tokenizer,
        num_sequences=args.num_sequences,
        num_rings=args.num_rings
    )
    
    logger.info("所有序列生成完成！")

if __name__ == "__main__":
    main() 