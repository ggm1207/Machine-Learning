import tensorflow as tf
import tensorflow_datasets as tfds

# data module이 담당하는 것은 데이터 로드
# 또는 이미지 데이터 증강 및 전처리 레이어


def flower_load():
    data_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file("flower_photos", origin=data_url, untar=True)
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        seed=1207,
        subset="training",
    )  # shape=(32, 180, 180, 3), dtype=float32,

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        seed=1207,
        subset="validation",
    )  # shape=(32, 180, 180, 3), dtype=float32,
    return train_ds, val_ds


def mnist_load():
    dataset, info = tfds.load("mnist", with_info=True, as_supervised=True)
    train_ds, val_ds = dataset["train"], dataset["test"]
    return train_ds, val_ds, info


def augmentataion_layer(IMG_HEIGHT, IMG_WIDTH):
    augmentataion_layers = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomFlip(
                "horizontal", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
            ),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
            tf.keras.layers.experimental.preprocessing.RandomZoom(0.3),
        ]
    )
    return augmentataion_layers


def normalization_layer():
    return tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)
