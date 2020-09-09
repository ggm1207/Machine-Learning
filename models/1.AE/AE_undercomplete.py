import tensorflow as tf

from AE import AE


n_inputs = 784
n_hidden = 100
n_outputs = n_inputs


class AE_UC(AE):
    def __init__(self):
        super().__init__("AE-undercomplete")

        self.hidden = tf.keras.layers.Dense(n_hidden)
        self.outputs = tf.keras.layers.Dense(n_outputs)

    def call(self, x):
        x = self.hidden(x)
        return self.outputs(x)
