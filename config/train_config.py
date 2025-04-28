# 训练参数
TRAIN_TEST_SPLIT = 0.15
RANDOM_SEED = 42
EARLY_STOPPING_PATIENCE = 10
BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GRADIENT_CLIPPING = 1.0
METRICS = ['loss', 'recon_loss', 'kl_loss']

# 学习率调度器
LEARNING_RATE_SCHEDULER = {
    'type': 'ReduceLROnPlateau',
    'patience': 5,
    'factor': 0.5,
    'min_lr': 1e-6
}

# 模型保存参数
MODEL_SAVE_DIR = 'results/models/vae/weights'
MODEL_SAVE_FREQUENCY = 5  # 每5个epoch保存一次

# 日志参数
LOG_DIR = 'results/logs'
LOG_LEVEL = 'INFO'

# 进度条设置
PROGRESS_BAR = True

# 使用ESM embeddings
def prepare_VAE(data, csv=False):
    if csv == True:
        seq = pd.read_csv(data)
    else:
        seq = data
    seq = chopping(seq)
    seq = padding(seq)
    # 使用ESM模型获取embeddings
    embeddings = get_esm_embeddings(seq)
    return embeddings

