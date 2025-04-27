import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
from models.vae.encoder import Encoder
from models.vae.decoder import Decoder
from models.vae.vae import VAE
from utils.data_utils import chopping, padding, onehot_encoding, onehot_decoding
from utils.model_utils import kl_anneal
import config

# 加载数据
data_path = config.DATA_PATH
with open(data_path, 'r') as f:
    sequences = f.readlines()
sequences = [seq.strip() for seq in sequences]

# 数据预处理
sequences = chopping(sequences, lim=config.MAX_SEQUENCE_LENGTH)
sequences = padding(sequences, lim=config.MAX_SEQUENCE_LENGTH)
sequences = onehot_encoding(sequences, alphabet=config.ALPHABET)
sequences = np.asarray(sequences)

# 定义模型参数
latent_dim = config.LATENT_DIM
hidden_dim = config.HIDDEN_DIM
input_shape = sequences.shape[1:]
output_shape = sequences.shape[1:]

# 创建模型
encoder = Encoder(latent_dim, hidden_dim, input_shape)
decoder = Decoder(latent_dim, hidden_dim, output_shape)
vae = VAE(encoder, decoder)

# 编译模型
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE))

# 训练模型
epochs = config.EPOCHS
batch_size = config.BATCH_SIZE
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    for step in range(0, len(sequences), batch_size):
        batch = sequences[step:step + batch_size]
        vae.train_step(batch)
    # 保存模型
    if (epoch + 1) % 10 == 0:
        vae.save_weights(f'models/vae/weights/epoch_{epoch + 1}.h5') 