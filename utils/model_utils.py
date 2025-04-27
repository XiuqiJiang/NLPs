import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def kl_anneal(step, type='logistic', k=0.001, x0=100):
    """Function to dynamically change the KL term weight during model training

    Args:
        step (int): Current step in training
        type (str, optional): KL weight schedule; either logistic or linear. Defaults to 'logistic'.
        k (float, optional): k in Logisitc function. Defaults to 0.001.
        x0 (int, optional): x0 in logitstic function. Defaults to 100.

    Returns:
        float: KL term weight between 0-1
    """
    # KL term annealing
    if type == 'normal':
        pass
    if type == 'logistic':
        return float(1/(1+np.exp(-k*(step-x0))))
    elif type == 'linear':
        return float(min(1, step/x0))

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector in latent space

    Args:
        keras (_type_): 

    Returns:
        _type_: latent tensor
    """

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon 