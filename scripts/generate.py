import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
from models.vae.encoder import Encoder
from models.vae.decoder import Decoder
from models.vae.vae import VAE
from utils.data_utils import onehot_decoding
import config

# 加载模型
latent_dim = config.LATENT_DIM
hidden_dim = config.HIDDEN_DIM
input_shape = config.INPUT_SHAPE
output_shape = config.OUTPUT_SHAPE

encoder = Encoder(latent_dim, hidden_dim, input_shape)
decoder = Decoder(latent_dim, hidden_dim, output_shape)
vae = VAE(encoder, decoder)

# 加载权重
vae.load_weights(config.MODEL_WEIGHTS_PATH)

# 生成肽
num_samples = config.NUM_SAMPLES
z = np.random.normal(0, 1, (num_samples, latent_dim))
generated_sequences = vae.decoder(z)
generated_sequences = onehot_decoding(generated_sequences, alphabet=config.ALPHABET)

# 保存生成的肽
with open(config.GENERATED_SEQUENCES_PATH, 'w') as f:
    for seq in generated_sequences:
        f.write(seq + '\n') 