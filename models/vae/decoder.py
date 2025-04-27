import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Decoder(keras.Model):
    def __init__(self, latent_dim, hidden_dim, output_shape, name='Decoder'):
        super(Decoder, self).__init__(name=name)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_shape = output_shape

        # 定义解码器网络层
        self.dense1 = layers.Dense(hidden_dim, activation='relu')
        self.dense2 = layers.Dense(hidden_dim, activation='relu')
        self.dense_output = layers.Dense(output_shape, activation='sigmoid')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense_output(x) 