import os
import tensorflow as tf


class AE(tf.keras.Model):
    def __init__(self, name):
        super(AE, self).__init__(name=name)
        self.cur_dir = os.path.abspath(".")
        self.filepath = os.path.join(self.cur_dir, f"weights/{name}")

    def save_weight(self, epoch, loss):
        filepath = os.path.join(self.filepath, f"epoch:{epoch}_loss:{loss}")
        try:
            self.save_weights(filepath, overwrite=True, save_format="tf")
        except ImportError as e:
            print(
                "h5py is not available and the weight file is in HDF5 format.",
                e,
                sep="\n",
            )

    # TODO: fix {filepath}
    def load_weight(self):
        self.load_weights(self.filepath, by_name=False)
