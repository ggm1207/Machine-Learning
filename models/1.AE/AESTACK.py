import tensorflow as tf
from AutoEncoder import AE, dense_layer


class AESTACK(AE):
    def __init__(self, input_dim, units):
        super().__init__("AESTACK")
        self.Encoder = tf.keras.Model(*[dense_layer(unit) for unit in units])
        self.Decoder = tf.keras.Model(*[dense_layer(input_dim) for unit in units[::-1]])

    def encode(self, x):
        return self.Encoder(x)

    def decode(self, x):
        return self.Decoder(x)
