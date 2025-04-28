# 训练参数
TRAIN_TEST_SPLIT = 0.15
RANDOM_SEED = 42
EARLY_STOPPING_PATIENCE = 10
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