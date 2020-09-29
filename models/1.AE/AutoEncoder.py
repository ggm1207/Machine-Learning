import os
from functools import partial

import tensorflow as tf


dense_layer = partial(
    tf.keras.layers.Dense,
    activation="relu",
    kernel_regularizer=tf.keras.regularizers.l2(0.001),
)


class AE(tf.keras.Model):
    def __init__(self, name):
        super(AE, self).__init__(name=name)
        self.cur_dir = os.path.abspath(".")
        self.file_dir = os.path.join(self.cur_dir, f"weights/{name}")

    @classmethod
    def preprocess_input(self, img):
        img = tf.cast(img, tf.float32)
        return img / 255.0
