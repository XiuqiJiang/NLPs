import torch
import os
import logging

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据相关配置
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
SEQUENCE_FILE = os.path.join(RAW_DATA_DIR, "sequences.txt")
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
ESM_EMBEDDING_DIM = 640
ESM_FINETUNE_EPOCHS = 6

# 序列配置
MAX_SEQUENCE_LENGTH = 64

# 训练配置
BATCH_SIZE = 32
NUM_EPOCHS = 400  # 修改为400轮
LEARNING_RATE = 1e-4
ESM_LEARNING_RATE = 1e-5

# VAE训练参数
MAX_BETA = 0.1
WARMUP_EPOCHS = 30
ANNEALING_EPOCHS = 100
KLD_TARGET = 10.0  # KL target loss目标值
KLD_TARGET_WEIGHT = 0.1  # KL target loss权重
KLD_FLOOR = 10.0  # KL loss的下限

def get_beta(epoch: int, max_beta: float = MAX_BETA, annealing_epochs: int = ANNEALING_EPOCHS, warmup_epochs: int = WARMUP_EPOCHS) -> float:
    """计算当前epoch的beta值
    
    Args:
        epoch: 当前epoch
        max_beta: 最大beta值
        annealing_epochs: beta退火的周期数
        warmup_epochs: 预热期epoch数
    Returns:
        beta值，范围从0到max_beta
    """
    if epoch < warmup_epochs:
        return 0.0
    else:
        current_annealing_epoch = epoch - warmup_epochs
        annealing_epochs_after_warmup = annealing_epochs - warmup_epochs
        beta = min(1.0, current_annealing_epoch / annealing_epochs_after_warmup) * max_beta
        return beta

BETA = 0.0  # 初始beta值，实际值会在训练过程中通过get_beta函数计算
TRAIN_TEST_SPLIT = 0.15
NUM_WORKERS = 4
PIN_MEMORY = True
MAX_GRAD_NORM = 1.0
GRADIENT_CLIPPING = 1.0

# 模型架构配置
HIDDEN_DIMS = [512, 256]
LATENT_DIM = 256  # 环数类别数

# 随机种子
RANDOM_SEED = 42

# 早停配置
EARLY_STOPPING_PATIENCE = 10
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