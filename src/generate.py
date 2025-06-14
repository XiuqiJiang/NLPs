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
import torch.nn.functional as F
import Levenshtein
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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
    GENERATED_SEQUENCES_PATH,
    RNN_HIDDEN_DIM
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
    # ====== 若无bos_token_id则补充为0 ======
    if not hasattr(tokenizer, 'bos_token_id') or tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = 0
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
        rnn_hidden_dim=RNN_HIDDEN_DIM  # 显式指定RNN隐藏层维度
    ).to(DEVICE)
    # ====== 保证model.sos_token_id与tokenizer.bos_token_id一致 ======
    model.sos_token_id = tokenizer.bos_token_id
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    # 赋值tokenizer属性，便于后续decode
    model.tokenizer = tokenizer
    # ====== 断言特殊token id一致性（只检查pad/eos） ======
    assert tokenizer.pad_token_id == model.pad_token_id, f"pad_token_id不一致: {tokenizer.pad_token_id} vs {model.pad_token_id}"
    assert hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id == model.eos_token_id, f"eos_token_id不一致: {getattr(tokenizer, 'eos_token_id', None)} vs {model.eos_token_id}"
    
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
        generated_ids = model.decode(z, target_ids=None)
        
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
    target_rings: int,  # 实际目标C数量
    temperature: float = 1.0,
    max_length: int = MAX_SEQUENCE_LENGTH,
    device: str = DEVICE
) -> List[str]:
    """生成指定环数的序列
    
    Args:
        model: VAE模型
        tokenizer: tokenizer
        num_sequences: 要生成的序列数量
        target_rings: 实际目标半胱氨酸数量
        temperature: 采样温度
        max_length: 最大序列长度
        device: 设备
        
    Returns:
        生成的序列列表
    """
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
    
    # 收集目标C数量序列的潜在向量
    target_ring_vectors = []
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
                    target_ring_vectors.append(mu[i].cpu())
                    if sequences_with_target_rings <= 5:  # 只打印前5个匹配的序列
                        logger.info(f"找到匹配序列 {sequences_with_target_rings}: {sequence} (C数量: {cysteine_count})")
    
    logger.info(f"总共处理了 {total_sequences} 个序列")
    logger.info(f"其中包含 {target_rings} 个C的序列有 {sequences_with_target_rings} 个")
    
    if len(target_ring_vectors) == 0:
        logger.warning(f"在训练数据中没有找到具有 {target_rings} 个C的序列，将使用随机采样")
        z = torch.randn(num_sequences, model.latent_dim, device=device)
    else:
        # 计算目标C数量序列潜在向量的均值和标准差
        target_ring_vectors = torch.stack(target_ring_vectors)
        mean = target_ring_vectors.mean(dim=0).to(device)  # 移动到正确的设备
        std = target_ring_vectors.std(dim=0).to(device)    # 移动到正确的设备
        
        # 在目标C数量序列的潜在空间区域采样
        logger.info(f"在{target_rings}个C序列的潜在空间区域采样 (均值: {mean.mean():.3f}, 标准差: {std.mean():.3f})")
        logger.info(f"潜在向量维度: {mean.shape}")
        
        # 生成多个批次，增加采样多样性
        all_z = []
        for _ in range(5):  # 生成5个批次
            z = mean + std * torch.randn(num_sequences, model.latent_dim, device=device)
            all_z.append(z)
        z = torch.cat(all_z, dim=0)  # 合并所有批次
        logger.info(f"总共采样了 {z.shape[0]} 个潜在向量")
    
    # 创建条件向量（将实际C数量转换为模型内部标签）
    decoder_label = c_count_to_label(target_rings)  # 正确映射
    condition = torch.full((z.shape[0],), decoder_label, device=device)
    logger.info(f"为解码器设置的条件标签: {decoder_label} (对应目标 {target_rings}个C)")
    
    # 解码（使用自回归模式，加入条件）
    logger.info("开始解码生成序列...")
    with torch.no_grad():
        generated_token_ids = model.decode(z, target_ids=None)
    
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
            logger.info(f"C数量: {cysteine_count}")
    logger.info(f"采样到的C数量分布: {c_count_stat}")
    logger.info(f"第一轮生成了 {total_generated} 个序列，其中 {len(valid_sequences)} 个C数在[{target_rings-1},{target_rings+1}]范围内")
    
    # 如果有效序列不足，继续生成直到达到要求
    generation_attempts = 0
    max_attempts = 100  # 设置最大尝试次数
    
    while len(valid_sequences) < num_sequences and generation_attempts < max_attempts:
        generation_attempts += 1
        logger.info(f"\n第 {generation_attempts} 次尝试生成序列...")
        
        # 在目标C数量序列的潜在空间区域采样
        if len(target_ring_vectors) > 0:
            # 生成多个批次，增加采样多样性
            all_z = []
            for _ in range(5):  # 生成5个批次
                z = mean + std * torch.randn(num_sequences, model.latent_dim, device=device)
                all_z.append(z)
            z = torch.cat(all_z, dim=0)  # 合并所有批次
        else:
            z = torch.randn(num_sequences, model.latent_dim, device=device)
        
        # 创建条件向量（将实际C数量转换为模型内部标签）
        decoder_label = c_count_to_label(target_rings)  # 正确映射
        condition = torch.full((z.shape[0],), decoder_label, device=device)
        
        # 解码
        with torch.no_grad():
            generated_token_ids = model.decode(z, target_ids=None)
        
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
                logger.info(f"C数量: {cysteine_count}")
                
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
        generated_ids = model.decode(z, target_ids=None)
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

def c_count_to_label(c_count):
    if c_count == 3:
        return 0
    elif c_count == 4:
        return 1
    elif c_count == 5:
        return 2
    else:
        raise ValueError(f"只支持3~5C，收到{c_count}")

def sample_with_various_z(model, tokenizer, num_sequences=100, ring_info_value=0, device='cuda', sampling='greedy', temperature=1.0):
    """
    采样极端z（全0、全1、全-1、全正、全负、随机），并生成序列，统计unique比例。
    支持greedy和softmax采样。
    """
    z_types = {
        'zero': torch.zeros(num_sequences, model.latent_dim, device=device),
        'one': torch.ones(num_sequences, model.latent_dim, device=device),
        'minus_one': -torch.ones(num_sequences, model.latent_dim, device=device),
        'random': torch.randn(num_sequences, model.latent_dim, device=device),
        'all_pos': torch.abs(torch.randn(num_sequences, model.latent_dim, device=device)),
        'all_neg': -torch.abs(torch.randn(num_sequences, model.latent_dim, device=device)),
    }
    ring_info = torch.full((num_sequences,), ring_info_value, dtype=torch.long, device=device)
    for z_type, z in z_types.items():
        print(f"\n==== z类型: {z_type} ====")
        with torch.no_grad():
            # 支持softmax采样
            if sampling == 'softmax':
                generated_ids = []
                h0 = model.latent_to_rnn_hidden(torch.cat([z, model.ring_encoder(ring_info)], dim=-1))
                h0 = h0.view(model.num_rnn_layers, num_sequences, model.rnn_hidden_dim)
                output_sequence = torch.full((num_sequences, model.max_sequence_length), model.pad_token_id, device=device)
                output_sequence[:, 0] = model.sos_token_id
                for t in range(1, model.max_sequence_length):
                    current_input = output_sequence[:, :t]
                    decoder_inputs = model.decoder_embedding(current_input)
                    cond_emb = model.ring_encoder(ring_info).unsqueeze(1).repeat(1, decoder_inputs.size(1), 1)
                    decoder_inputs = torch.cat([decoder_inputs, cond_emb], dim=-1)
                    rnn_output, h0 = model.decoder_rnn(decoder_inputs, h0)
                    last_output = rnn_output[:, -1]
                    logits = model.fc_out(model.decoder_dropout(last_output))
                    probs = F.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    output_sequence[:, t] = next_token
                    if (next_token == model.eos_token_id).all():
                        break
                generated_ids = output_sequence
            else:
                generated_ids = model.decode(z, target_ids=None)
        seqs = []
        for i, ids in enumerate(generated_ids):
            if isinstance(ids, torch.Tensor):
                ids = ids.detach().cpu().tolist()
            seq = tokenizer.decode(ids, skip_special_tokens=True)
            seqs.append(seq)
            print(f"z[{i}] ({z_type}): {z[i].cpu().numpy()[:5]}... => {seq}")
        unique_seqs = set(seqs)
        print(f"z类型: {z_type}，采样{num_sequences}条，unique序列数: {len(unique_seqs)}，unique比例: {len(unique_seqs)/num_sequences:.2f}")

def edit_distance(s1, s2):
    """标准Levenshtein编辑距离"""
    return Levenshtein.distance(s1, s2)

# 局部扰动采样实验
# 选取代表性肽、编码、扰动z、条件解码、评估

def select_representative_peptides(dataset, num_per_class=3):
    class_to_peptides = {0: [], 1: [], 2: []}
    for i in range(len(dataset)):
        seq = dataset.sequences[i]
        c_count = seq.count('C')
        if c_count in [3, 4, 5]:
            label = c_count - 3
            if len(class_to_peptides[label]) < num_per_class:
                class_to_peptides[label].append(seq)
        if all(len(v) >= num_per_class for v in class_to_peptides.values()):
            break
    return class_to_peptides

def encode_peptides(model, tokenizer, peptides, device, sequences, embeddings):
    model.eval()
    z_means, z_logvars = [], []
    for seq in peptides:
        idx = sequences.index(seq)
        esm_emb = embeddings[idx]
        if isinstance(esm_emb, np.ndarray):
            esm_emb = torch.tensor(esm_emb)
        esm_emb = esm_emb.unsqueeze(0).to(device)  # [1, seq_len, input_dim]
        mu, logvar = model.encode(esm_emb)
        z_means.append(mu.squeeze(0).detach().cpu())
        z_logvars.append(logvar.squeeze(0).detach().cpu())
    return z_means, z_logvars

def perturb_and_decode(model, tokenizer, z_mean, ring_label, epsilon_std, temperature=1.0, device='cuda'):
    z_perturbed = z_mean + torch.randn_like(z_mean) * epsilon_std
    z_perturbed = z_perturbed.unsqueeze(0).to(device)
    ring_info = torch.tensor([ring_label], dtype=torch.long, device=device)
    with torch.no_grad():
        generated_ids = model.sample_from_z(z_perturbed, ring_info, temperature=temperature)
    ids = generated_ids[0].detach().cpu().tolist()
    seq = tokenizer.decode(ids, skip_special_tokens=True)
    return seq, z_perturbed.cpu().numpy()

def local_perturbation_experiment(model, tokenizer, dataset, device, num_per_class=3):
    perturbation_stds = [0.0, 0.0005, 0.001, 0.005, 0.01]
    temperatures = [0.0, 0.1, 0.2, 0.3]
    class_to_peptides = select_representative_peptides(dataset, num_per_class)
    # 获取sequences和embeddings
    sequences = dataset.sequences
    embeddings = dataset.embeddings
    for label, peptides in class_to_peptides.items():
        print(f"\n=== 环数类别 {label}（C数={label+3}） ===")
        z_means, z_logvars = encode_peptides(model, tokenizer, peptides, device, sequences, embeddings)
        for i, seq in enumerate(peptides):
            print(f"\n原始肽{i+1}: {seq}")
            for eps in perturbation_stds:
                for temp in temperatures:
                    gen_seq, z_pert = perturb_and_decode(model, tokenizer, z_means[i], label, eps, temp, device)
                    c_count = gen_seq.count('C')
                    ed = edit_distance(seq, gen_seq)
                    print(f"  [扰动std={eps:.4f}, temp={temp:.2f}] 生成: {gen_seq} | C数={c_count} | 编辑距离={ed}")

def generate_perturbed_for_all_train(model, tokenizer, dataset, device):
    """对训练集每个样本z_mean多组扰动、不同温度各生成20条新序列，输出原始与扰动序列及编辑距离统计，并保存到文件"""
    sequences = dataset.sequences
    embeddings = dataset.embeddings
    num_per_sample = 20  # 修改为每种情况生成20条
    epsilon_stds = [0.001, 0.005, 0.01]
    temperatures = [0.7, 1.0, 1.3]
    print(f"共{len(sequences)}条训练集样本，每条每扰动强度每温度生成{num_per_sample}条扰动序列...")

    # 新增：收集所有生成的序列及相关信息
    all_results = []

    for epsilon_std in epsilon_stds:
        print(f"\n==== 当前扰动强度 epsilon_std={epsilon_std} ====")
        for temp in temperatures:
            print(f"\n---- 当前temperature={temp} ----")
            all_edit_distances = []
            for idx, seq in enumerate(sequences):
                esm_emb = embeddings[idx]
                if isinstance(esm_emb, np.ndarray):
                    esm_emb = torch.tensor(esm_emb)
                esm_emb = esm_emb.unsqueeze(0).to(device)
                mu, _ = model.encode(esm_emb)
                mu = mu.squeeze(0)
                c_count = seq.count('C')
                ring_label = c_count - 3 if 3 <= c_count <= 5 else 0
                print(f"\n样本{idx+1}: {seq}")
                for j in range(num_per_sample):
                    z_perturbed = mu + torch.randn_like(mu) * epsilon_std
                    z_perturbed = z_perturbed.unsqueeze(0).to(device)
                    ring_info = torch.tensor([ring_label], dtype=torch.long, device=device)
                    with torch.no_grad():
                        gen_ids = model.decode(z_perturbed, target_ids=None, temperature=temp)
                    ids = gen_ids[0].detach().cpu().tolist()
                    gen_seq = tokenizer.decode(ids, skip_special_tokens=True)
                    ed = Levenshtein.distance(seq, gen_seq)
                    all_edit_distances.append(ed)
                    print(f"  [扰动{j+1}] 生成: {gen_seq} | 编辑距离: {ed}")
                    # 新增：保存本条结果
                    all_results.append({
                        'original_seq': seq,
                        'epsilon_std': epsilon_std,
                        'temperature': temp,
                        'perturb_id': j+1,
                        'generated_seq': gen_seq,
                        'edit_distance': ed
                    })
            # 输出本组扰动的编辑距离统计
            if all_edit_distances:
                avg_ed = np.mean(all_edit_distances)
                max_ed = np.max(all_edit_distances)
                min_ed = np.min(all_edit_distances)
                print(f"\n[扰动std={epsilon_std}, temp={temp}] 全部扰动样本的编辑距离: 平均={avg_ed:.2f}，最小={min_ed}，最大={max_ed}")

    # 新增：保存所有结果到TSV文件，路径固定为./results/perturbed_sequences.tsv
    save_path = './results/perturbed_sequences.tsv'
    output_dir = os.path.dirname(save_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('original_seq\tepsilon_std\ttemperature\tperturb_id\tgenerated_seq\tedit_distance\n')
        for item in all_results:
            f.write(f"{item['original_seq']}\t{item['epsilon_std']}\t{item['temperature']}\t{item['perturb_id']}\t{item['generated_seq']}\t{item['edit_distance']}\n")
    print(f"\n所有扰动生成的序列已保存到: {save_path}")

def vae_reconstruction_selfcheck(model, tokenizer, dataset, device):
    """
    取一条训练集embedding，encode后直接decode，检查能否还原原始序列
    """
    import numpy as np
    from Levenshtein import distance as levenshtein_distance
    print("========== VAE闭环自检 ==========")
    seq = dataset.sequences[0]
    esm_emb = dataset.embeddings[0]
    if isinstance(esm_emb, np.ndarray):
        esm_emb = torch.tensor(esm_emb)
    esm_emb = esm_emb.unsqueeze(0).to(device)
    with torch.no_grad():
        mu, _ = model.encode(esm_emb)
        gen_ids = model.decode(mu)
        ids = gen_ids[0].detach().cpu().tolist()
        recon_seq = tokenizer.decode(ids, skip_special_tokens=True)
    print("原始序列:", seq)
    print("重构序列:", recon_seq)
    ed = levenshtein_distance(seq, recon_seq)
    print(f"编辑距离: {ed}")
    print("=================================")

def visualize_z_space(model, dataset, device, method='pca', max_points=1000, save_path='z_space.png'):
    """
    可视化训练集z_mean分布，支持PCA和t-SNE
    """
    print(f"\n========== 可视化z_mean分布（{method.upper()}） ==========")
    zs = []
    labels = []
    for i in range(min(len(dataset.sequences), max_points)):
        seq = dataset.sequences[i]
        esm_emb = dataset.embeddings[i]
        if isinstance(esm_emb, np.ndarray):
            esm_emb = torch.tensor(esm_emb)
        esm_emb = esm_emb.unsqueeze(0).to(device)
        with torch.no_grad():
            mu, _ = model.encode(esm_emb)
        zs.append(mu.squeeze(0).cpu().numpy())
        # 以C数量为标签
        labels.append(seq.count('C'))
    zs = np.stack(zs)
    labels = np.array(labels)
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:
        raise ValueError('method必须为pca或tsne')
    zs_2d = reducer.fit_transform(zs)
    plt.figure(figsize=(7,6))
    scatter = plt.scatter(zs_2d[:,0], zs_2d[:,1], c=labels, cmap='rainbow', alpha=0.7, s=18)
    plt.colorbar(scatter, label='Cysteine Count')
    plt.title(f'z_mean分布 ({method.upper()})')
    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"z_mean二维分布图已保存到: {save_path}")
    plt.close()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="蛋白质VAE训练集扰动生成")
    parser.add_argument('--model_path', type=str, required=True, help='模型权重路径')
    args = parser.parse_args()

    logger = setup_logging()
    logger.info("开始训练集扰动生成...")
    model, tokenizer = load_model(args.model_path)

    # 加载数据集
    data = torch.load(EMBEDDING_FILE)
    if isinstance(data, dict) and 'sequences' in data and 'embeddings' in data:
        sequences = data['sequences']
        embeddings = data['embeddings']
    elif isinstance(data, list):
        sequences = [item['sequence'] for item in data]
        embeddings = [item['embeddings'] for item in data]
        embeddings = torch.stack(embeddings)
    dataset = ProteinDataset(sequences=sequences, embeddings=embeddings, tokenizer=tokenizer, max_length=MAX_SEQUENCE_LENGTH)

    # 直接调用，无需外部参数
    generate_perturbed_for_all_train(
        model, tokenizer, dataset, DEVICE
    )

if __name__ == "__main__":
    main() 