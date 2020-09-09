import tensorflow as tf
from tensorflow.keras import datasets

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 1000


def normalization(images, labels):
    images = images / 255
    return images, labels


def load_mnist(flatten=True) -> (tf.data.Dataset, tf.data.Dataset):
    """ return train_ds, test_ds """
    (
        (train_images, train_labels),
        (test_images, test_labels),
    ) = datasets.mnist.load_data()

    if flatten:
        train_images = train_images.reshape((60000, 28 * 28))
    else:
        train_images = train_images.reshape((60000, 28, 28, 1))

    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

    train_ds = (
        train_ds.map(normalization)
        .shuffle(SHUFFLE_BUFFER_SIZE)
        .batch(BATCH_SIZE)
    )
    test_ds = test_ds.map(normalization).batch(BATCH_SIZE)

    return train_ds, test_ds
