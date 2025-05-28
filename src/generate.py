import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import logging
from pathlib import Path
import argparse
from typing import List
import torch.nn as nn

# 添加项目根目录到Python路径
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from src.models.vae_token import ESMVAEToken
from src.utils.data_utils import ProteinDataset  # 修改导入路径
from config.config import (
    EMBEDDING_FILE,
    ESM_MODEL_PATH,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    MAX_SEQUENCE_LENGTH,
    MODEL_SAVE_DIR,
    DEVICE,
    ESM_EMBEDDING_DIM,
    HIDDEN_DIMS,
    LATENT_DIM,
    GENERATED_SEQUENCES_PATH
)

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('generation.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_model(model_path: str) -> tuple[ESMVAEToken, AutoTokenizer]:
    """加载模型
    
    Args:
        model_path: 模型文件路径
        
    Returns:
        (模型, tokenizer)
    """
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
        dropout=0.1,
        num_classes=8  # 保证与训练时一致
    ).to(DEVICE)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 赋值tokenizer属性，便于后续decode
    model.tokenizer = tokenizer
    
    return model, tokenizer

def generate_sequences(
    model: nn.Module,
    num_sequences: int,
    target_rings: int,
    device: torch.device,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    ring_variance: int = 1  # 允许的环数变化范围
) -> List[str]:
    """生成序列
    
    Args:
        model: VAE模型
        num_sequences: 要生成的序列数量
        target_rings: 目标环数
        device: 设备
        temperature: 采样温度
        top_k: top-k采样的k值
        top_p: nucleus采样的p值
        ring_variance: 允许的环数变化范围（±ring_variance）
        
    Returns:
        生成的序列列表
    """
    model.eval()
    sequences = []
    logger = logging.getLogger(__name__)
    
    with torch.no_grad():
        # 生成潜在向量
        z = torch.randn(num_sequences, model.latent_dim, device=device)
        
        # 在目标环数附近采样
        ring_range = range(target_rings - ring_variance, target_rings + ring_variance + 1)
        ring_info = torch.randint(
            min(ring_range), 
            max(ring_range) + 1, 
            (num_sequences,), 
            device=device
        )
        
        logger.info(f"目标环数: {target_rings}，采样范围: {min(ring_range)}-{max(ring_range)}")
        
        # 解码生成序列
        generated_ids = model.decode(z, target_ids=None, ring_info=ring_info)
        
        # 将token IDs转换为序列
        valid_sequences = []
        for ids in generated_ids:
            # 找到EOS token的位置
            eos_pos = (ids == model.eos_token_id).nonzero()
            if len(eos_pos) > 0:
                ids = ids[:eos_pos[0].item()]
            
            # 移除padding和特殊token
            ids = ids[ids != model.pad_token_id]
            ids = ids[ids != model.sos_token_id]
            ids = ids[ids != model.eos_token_id]
            
            # 转换为序列，确保为list
            if isinstance(ids, torch.Tensor):
                ids = ids.detach().cpu().tolist()
            sequence = model.tokenizer.decode(ids)
            cysteine_count = sequence.count('C')
            
            # 检查环数是否在允许范围内
            if cysteine_count in ring_range:
                valid_sequences.append(sequence)
                logger.info(f"生成序列 (环数: {cysteine_count}): {sequence}")
    
    logger.info(f"生成了 {len(valid_sequences)} 个有效序列")
    return valid_sequences

def generate_sequences_with_rings(
    model: ESMVAEToken,
    tokenizer: AutoTokenizer,
    num_sequences: int,
    target_rings: int,
    temperature: float = 1.0,
    max_length: int = MAX_SEQUENCE_LENGTH,
    device: str = DEVICE
) -> List[str]:
    """生成指定环数的序列"""
    model.eval()
    generated_sequences = []
    logger = logging.getLogger(__name__)
    
    # 加载数据，自动兼容dict或list格式
    logger.info(f"正在加载数据文件: {EMBEDDING_FILE}")
    data = torch.load(EMBEDDING_FILE)
    if isinstance(data, dict) and 'sequences' in data and 'embeddings' in data:
        sequences = data['sequences']
        embeddings = data['embeddings']
    elif isinstance(data, list):
        sequences = [item['sequence'] for item in data]
        embeddings = [item['embeddings'] for item in data]
        embeddings = torch.stack(embeddings)
    else:
        raise ValueError("未知的数据格式，请检查EMBEDDING_FILE内容！")
    logger.info(f"加载了 {len(sequences)} 个序列")
    
    # 创建数据集
    dataset = ProteinDataset(
        sequences=sequences,
        embeddings=embeddings,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    # 创建数据加载器
    if str(device).startswith('cuda'):
        num_workers = NUM_WORKERS
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # 收集三环序列的潜在向量
    three_ring_vectors = []
    total_sequences = 0
    sequences_with_target_rings = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="处理训练数据"):
            total_sequences += len(batch['input_ids'])
            # 确保embeddings维度正确 [batch_size, seq_len, input_dim]
            embeddings = batch['embeddings'].to(device)
            if len(embeddings.shape) == 2:
                # 如果embeddings是2维的，需要重塑为3维
                batch_size = embeddings.size(0)
                embeddings = embeddings.view(batch_size, -1, model.input_dim)
            
            # 获取潜在向量
            mu, _ = model.encode(embeddings)
            
            # 检查每个序列的半胱氨酸数量
            for i, seq in enumerate(batch['input_ids']):
                sequence = tokenizer.decode(seq, skip_special_tokens=True)
                cysteine_count = sequence.count('C')
                if cysteine_count == target_rings:
                    sequences_with_target_rings += 1
                    three_ring_vectors.append(mu[i].cpu())
                    if sequences_with_target_rings <= 5:  # 只打印前5个匹配的序列
                        logger.info(f"找到匹配序列 {sequences_with_target_rings}: {sequence} (环数: {cysteine_count})")
    
    logger.info(f"总共处理了 {total_sequences} 个序列")
    logger.info(f"其中包含 {target_rings} 个环的序列有 {sequences_with_target_rings} 个")
    
    if len(three_ring_vectors) == 0:
        logger.warning(f"在训练数据中没有找到具有 {target_rings} 个环的序列，将使用随机采样")
        z = torch.randn(num_sequences, model.latent_dim, device=device)
    else:
        # 计算三环序列潜在向量的均值和标准差
        three_ring_vectors = torch.stack(three_ring_vectors)
        mean = three_ring_vectors.mean(dim=0).to(device)  # 移动到正确的设备
        std = three_ring_vectors.std(dim=0).to(device)    # 移动到正确的设备
        
        # 在三环序列的潜在空间区域采样
        logger.info(f"在三环序列的潜在空间区域采样 (均值: {mean.mean():.3f}, 标准差: {std.mean():.3f})")
        logger.info(f"潜在向量维度: {mean.shape}")
        
        # 生成多个批次，增加采样多样性
        all_z = []
        for _ in range(5):  # 生成5个批次
            z = mean + std * torch.randn(num_sequences, model.latent_dim, device=device)
            all_z.append(z)
        z = torch.cat(all_z, dim=0)  # 合并所有批次
        logger.info(f"总共采样了 {z.shape[0]} 个潜在向量")
    
    # 创建条件向量（环数）
    condition = torch.full((z.shape[0],), target_rings, device=device)
    
    # 解码（使用自回归模式，加入条件）
    logger.info("开始解码生成序列...")
    with torch.no_grad():
        generated_token_ids = model.decode(z, target_ids=None, ring_info=condition)
    
    # 将生成的token ids转换回序列
    valid_sequences = []
    total_generated = 0
    c_count_stat = {}
    for i in range(len(generated_token_ids)):
        total_generated += 1
        ids = generated_token_ids[i]
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        sequence = tokenizer.decode(ids, skip_special_tokens=True)
        cysteine_count = sequence.count('C')
        # 统计C数量分布
        c_count_stat[cysteine_count] = c_count_stat.get(cysteine_count, 0) + 1
        # 放宽判定条件
        if abs(cysteine_count - target_rings) <= 1:
            valid_sequences.append(sequence)
            logger.info(f"\n生成的序列 {len(valid_sequences)}:")
            logger.info(sequence)
            logger.info(f"环数: {cysteine_count}")
    logger.info(f"采样到的C数量分布: {c_count_stat}")
    logger.info(f"第一轮生成了 {total_generated} 个序列，其中 {len(valid_sequences)} 个C数在[{target_rings-1},{target_rings+1}]范围内")
    
    # 如果有效序列不足，继续生成直到达到要求
    generation_attempts = 0
    max_attempts = 100  # 设置最大尝试次数
    
    while len(valid_sequences) < num_sequences and generation_attempts < max_attempts:
        generation_attempts += 1
        logger.info(f"\n第 {generation_attempts} 次尝试生成序列...")
        
        # 在三环序列的潜在空间区域采样
        if len(three_ring_vectors) > 0:
            # 生成多个批次，增加采样多样性
            all_z = []
            for _ in range(5):  # 生成5个批次
                z = mean + std * torch.randn(num_sequences, model.latent_dim, device=device)
                all_z.append(z)
            z = torch.cat(all_z, dim=0)  # 合并所有批次
        else:
            z = torch.randn(num_sequences, model.latent_dim, device=device)
        
        # 创建条件向量（环数）
        condition = torch.full((z.shape[0],), target_rings, device=device)
        
        # 解码
        with torch.no_grad():
            generated_token_ids = model.decode(z, target_ids=None, ring_info=condition)
        
        # 检查新生成的序列
        batch_valid = 0
        for i in range(len(generated_token_ids)):
            ids = generated_token_ids[i]
            if isinstance(ids, torch.Tensor):
                ids = ids.detach().cpu().tolist()
            sequence = tokenizer.decode(ids, skip_special_tokens=True)
            cysteine_count = sequence.count('C')
            # 统计C数量分布
            c_count_stat[cysteine_count] = c_count_stat.get(cysteine_count, 0) + 1
            # 放宽判定条件
            if abs(cysteine_count - target_rings) <= 1:
                valid_sequences.append(sequence)
                batch_valid += 1
                logger.info(f"\n生成的序列 {len(valid_sequences)}:")
                logger.info(sequence)
                logger.info(f"环数: {cysteine_count}")
                
                if len(valid_sequences) >= num_sequences:
                    break
        
        logger.info(f"本次尝试生成了 {batch_valid} 个有效序列")
        
        if generation_attempts >= max_attempts:
            logger.warning(f"达到最大尝试次数 {max_attempts}，停止生成")
            break
    
    logger.info(f"总共生成了 {len(valid_sequences)} 个有效序列")
    
    # 保存生成的序列
    output_dir = os.path.dirname(GENERATED_SEQUENCES_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(GENERATED_SEQUENCES_PATH, 'w') as f:
        for seq in valid_sequences:
            f.write(seq + '\n')
    
    logger.info(f"序列已保存到 {GENERATED_SEQUENCES_PATH}")
    logger.info("序列生成完成！")
    
    print(f"采样到的C数量分布: {c_count_stat}")
    
    return valid_sequences

def global_sample_sequences(
    model,
    tokenizer,
    num_sequences=100,
    ring_info_value=2,  # 目标环数类别（如3环就填2，需与训练映射一致）
    temperature=1.0,
    device='cuda'
):
    """
    全局采样：直接从标准正态分布采样z，指定环数条件，生成蛋白质序列
    """
    model.eval()
    results = []
    with torch.no_grad():
        # 1. 采样z
        z = torch.randn(num_sequences, model.latent_dim, device=device) * temperature
        # 2. 构造环数条件
        ring_info = torch.full((num_sequences,), ring_info_value, dtype=torch.long, device=device)
        # 3. 解码生成token id序列
        generated_ids = model.decode(z, target_ids=None, ring_info=ring_info)
        # 4. 转为蛋白质序列字符串
        for ids in generated_ids:
            # 截断到EOS
            eos_pos = (ids == model.eos_token_id).nonzero()
            if len(eos_pos) > 0:
                ids = ids[:eos_pos[0].item()]
            # 移除特殊token
            ids = ids[(ids != model.pad_token_id) & (ids != model.sos_token_id) & (ids != model.eos_token_id)]
            seq = tokenizer.decode(ids.cpu().tolist())
            results.append(seq)
    return results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="蛋白质VAE序列生成")
    parser.add_argument('--model_path', type=str, required=True, help='模型权重路径')
    parser.add_argument('--num_sequences', type=int, default=100, help='生成序列数量')
    parser.add_argument('--ring_info', type=int, default=2, help='目标环数类别（如3环就填2，需与训练映射一致）')
    parser.add_argument('--temperature', type=float, default=1.0, help='采样温度')
    parser.add_argument('--global_sample', action='store_true', help='是否启用全局采样模式')
    args = parser.parse_args()
    
    logger = setup_logging()
    logger.info("开始生成序列...")
    
    # 使用默认模型路径
    model_path = args.model_path
    logger.info(f"使用默认模型路径: {model_path}")
    
    # 加载模型
    logger.info(f"从 {model_path} 加载模型...")
    model, tokenizer = load_model(model_path)
    
    if args.global_sample:
        print(f"[全局采样] 采样{args.num_sequences}条，环数类别={args.ring_info}")
        samples = global_sample_sequences(
            model,
            tokenizer,
            num_sequences=args.num_sequences,
            ring_info_value=args.ring_info,
            temperature=args.temperature,
            device=DEVICE
        )
        for i, seq in enumerate(samples):
            print(f">Sample_{i+1}\n{seq}")
        print(f"共生成{len(samples)}条序列")
        return

    # 支持多个环数批量生成，局部高斯采样
    ring_list = [args.ring_info]

    for ring in ring_list:
        logger.info(f"\n=== 目标环数: {ring} ===")
        sequences = generate_sequences_with_rings(
            model=model,
            tokenizer=tokenizer,
            num_sequences=args.num_sequences,
            target_rings=ring,
            temperature=args.temperature,
            max_length=MAX_SEQUENCE_LENGTH,
            device=DEVICE
        )
        unique_seqs = set(sequences)
        logger.info(f"目标环数{ring}: 采样{args.num_sequences}次，unique序列数: {len(unique_seqs)}，unique比例: {len(unique_seqs)/args.num_sequences:.2f}")
        for i, seq in enumerate(unique_seqs, 1):
            print(f">Ring{ring}_Sample{i}\n{seq}")

if __name__ == "__main__":
    main() 