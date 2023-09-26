import tensorflow as tf
import numpy as np
from tensorflow.keras.initializers import GlorotUniform


class DenseNet(tf.keras.Sequential):
    
    def __init__(self,
                 input_shape=None, 
                 last_units=1,
                 last_activation=None,
                 layers=3,
                 activation="relu",
                 units=100,
                 scale=1.,
                 offset=0.):

        state = np.random.randint(2**16)
        super().__init__()
        self.add(tf.keras.layers.Flatten(input_shape=input_shape))
        for _ in range(layers):
            self.add(tf.keras.layers.Dense(units, activation=None,
                                           kernel_initializer=GlorotUniform(seed=state)))
            self.add(tf.keras.layers.Activation(activation))
        self.add(tf.keras.layers.Dense(last_units, activation=None,
                                       kernel_initializer=GlorotUniform(seed=state)))
        self.add(tf.keras.layers.Activation(last_activation))
        self.add(tf.keras.layers.Rescaling(scale=scale, offset=offset))
        return None