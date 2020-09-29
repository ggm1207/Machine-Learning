import tensorflow as tf
from AutoEncoder import AE, dense_layer


class AEUC(AE):
    def __init__(self, input_dim, units):
        super().__init__("AEUC")
        self.Encoder = tf.keras.Model(dense_layer(units[0]))
        self.Decoder = tf.keras.Model(dense_layer(input_dim, activation="sigmoid"))

    def encode(self, x):
        return self.Encoder(x)

    def decode(self, x):
        return self.Decoder(x)

    def call(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
