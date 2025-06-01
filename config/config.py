import torch
import os
import logging
import numpy as np

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据相关配置
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
SEQUENCE_FILE = os.path.join(RAW_DATA_DIR, "sequences_filtered_3to5C.txt")
EMBEDDING_FILE = os.path.join(PROCESSED_DATA_DIR, "protein_data.pt")

# 模型保存目录
SAVE_DIR = os.path.join(ROOT_DIR, "results")
MODEL_SAVE_DIR = os.path.join(SAVE_DIR, "vae")
LOG_DIR = os.path.join(SAVE_DIR, "logs")
ESM_OUTPUT_DIR = os.path.join(SAVE_DIR, "esm_finetuned")

# 确保目录存在
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(ESM_OUTPUT_DIR, exist_ok=True)

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型配置
ESM_MODEL_PATH = os.environ.get("ESM_MODEL_PATH", "/content/drive/MyDrive/esm_model")  # 默认Google Drive路径，可通过环境变量覆盖
ESM_MODEL_NAME = "esm2-fine-tune"
ESM_EMBEDDING_DIM = 1280  # ESM嵌入维度，需与ESM模型输出一致
ESM_FINETUNE_EPOCHS = 6

# 序列配置
MAX_SEQUENCE_LENGTH = 64

# 训练配置
BATCH_SIZE = 8
NUM_EPOCHS = 400  # 修改为400轮
LEARNING_RATE = 1e-4
ESM_LEARNING_RATE = 1e-5

# VAE训练参数
MAX_BETA = 0.0  # 最大beta值，训练全程无KL正则，退化为AE
WARMUP_EPOCHS = 100  # beta预热轮数
BETA_FREEZE_EPOCHS = 60  # 前60轮冻结beta为0
KLD_TARGET = 2.0  # KL散度目标值
KLD_TARGET_WEIGHT = 1.0  # KL散度目标惩罚权重
# KLD_FLOOR = 0.0  # KL散度下限，防止过度压缩 # 暂时去掉

def get_beta(epoch: int, max_beta: float = MAX_BETA, warmup_epochs: int = 300, freeze_epochs: int = BETA_FREEZE_EPOCHS) -> float:
    """计算当前epoch的beta值，支持前freeze_epochs轮beta为0
    
    Args:
        epoch: 当前epoch（从0开始）
        max_beta: 最大beta值
        warmup_epochs: 预热期epoch数
        freeze_epochs: 冻结期epoch数（前N轮beta为0）
        
    Returns:
        beta值，范围从0到max_beta
    """
    if warmup_epochs == 0:
        return max_beta
    if epoch < freeze_epochs:
        return 0.0
    elif epoch < freeze_epochs + warmup_epochs:
        # 线性增长到max_beta
        current_progress = (epoch - freeze_epochs + 1) / warmup_epochs
        beta = current_progress * max_beta
        return min(beta, max_beta)
    else:
        return max_beta

BETA = 0.0  # 初始beta值，实际值会在训练过程中通过get_beta函数计算
TRAIN_TEST_SPLIT = 0.15
NUM_WORKERS = 4
PIN_MEMORY = True
MAX_GRAD_NORM = 1.0
GRADIENT_CLIPPING = 1.0

# 模型架构配置
HIDDEN_DIMS = [1024, 512]
LATENT_DIM = 512  # 潜在空间维度，适合高重构能力需求
RNN_HIDDEN_DIM = 512
NUM_RNN_LAYERS = 1
RING_EMBEDDING_DIM = 64  # 条件嵌入维度，需与训练和推断保持一致

# 随机种子
RANDOM_SEED = 42

# 早停配置
EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_MIN_DELTA = 0.01

# 保存配置
SAVE_EVERY = 5  # 每5个epoch保存一次模型
MODEL_SAVE_FREQUENCY = 5

# 生成配置
NUM_SAMPLES = 10
GENERATED_SEQUENCES_PATH = os.path.join(SAVE_DIR, 'generated', 'sequences.txt')

# 日志配置
LOG_LEVEL = logging.INFO
PROGRESS_BAR = True

# 验证配置
def validate_config():
    """验证配置参数的一致性"""
    # 验证序列长度
    assert MAX_SEQUENCE_LENGTH > 0, "MAX_SEQUENCE_LENGTH 必须大于 0"
    
    # 验证学习率
    assert LEARNING_RATE > 0, "LEARNING_RATE 必须大于 0"
    assert ESM_LEARNING_RATE > 0, "ESM_LEARNING_RATE 必须大于 0"
    
    # 验证KL权重


# 运行配置验证
validate_config() 