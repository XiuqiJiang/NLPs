import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Encoder(keras.Model):
    def __init__(self, latent_dim, hidden_dim, input_shape, name='Encoder'):
        super(Encoder, self).__init__(name=name)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.input_shape = input_shape

        # 定义编码器网络层
        self.dense1 = layers.Dense(hidden_dim, activation='relu')
        self.dense2 = layers.Dense(hidden_dim, activation='relu')
        self.z_mean = layers.Dense(latent_dim)
        self.z_log_var = layers.Dense(latent_dim)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        return z_mean, z_log_var

    def sampling(self, z_mean, z_log_var):
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon 