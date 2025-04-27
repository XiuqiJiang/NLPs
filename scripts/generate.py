import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
from models.vae.encoder import Encoder
from models.vae.decoder import Decoder
from models.vae.vae import VAE
from utils.data_utils import onehot_decoding

# 加载模型
latent_dim = 50
hidden_dim = 256
input_shape = (48, 21)  # 根据实际情况调整
output_shape = (48, 21)  # 根据实际情况调整

encoder = Encoder(latent_dim, hidden_dim, input_shape)
decoder = Decoder(latent_dim, hidden_dim, output_shape)
vae = VAE(encoder, decoder)

# 加载权重
vae.load_weights('models/vae/weights/epoch_100.h5')  # 根据实际情况调整

# 生成肽
num_samples = 10
z = np.random.normal(0, 1, (num_samples, latent_dim))
generated_sequences = vae.decoder(z)
generated_sequences = onehot_decoding(generated_sequences)

# 保存生成的肽
with open('data/processed/generated_sequences.txt', 'w') as f:
    for seq in generated_sequences:
        f.write(seq + '\n') 