import tensorflow as tf
import matplotlib.pyplot as plt

from AEUC import AEUC
from AutoEncoder import AE
from data import mnist_load, flower_load

EPOCHS = 30
BATCH_SIZE = 32
template = "EPOCH: {}, LOSS: {}"
loss_object = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam()
units = [512, 256, 128]


def get_shape(dataset):
    batch, *shape = next(iter(dataset.take(1)))[0].shape
    return shape


def autoencoder_train_dense_model(train_ds, val_ds):
    """Train autoencoder model constructed by dense layer"""

    for model_class in AE.__subclasses__():
        w, h, channel = get_shape(train_ds)
        input_dim = w * h * channel
        model = model_class(input_dim, units)
        # model.file_dir += "_Dense"

        training(train_ds, val_ds, model)


def prepare(image, label):
    image = AE.preprocess_input(image)
    image = tf.reshape(image, [BATCH_SIZE, -1])
    return image, label


@tf.function
def train(y_true, model):
    with tf.GradientTape() as tape:
        y_pred = model(y_true)
        loss = loss_object(y_true, y_pred)

    trainable_variables = model.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss


def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ["Input Image", "Predicted Image"]

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(
            tf.keras.preprocessing.image.array_to_img(display_list[i]), cmap="gray"
        )
        plt.axis("off")
    plt.show()


def show_predictions(model, sample_image, w, h, channel):
    sample_pred = model.predict(sample_image)
    sample_image = tf.reshape(sample_image, [w, h, channel])
    sample_pred = tf.reshape(sample_pred, [w, h, channel])
    display([sample_image, sample_pred])


def training(train_ds, val_ds, model, visual):
    train_ds = train_ds.map(prepare)
    val_ds = val_ds.map(prepare)

    w, h, channel = get_shape(train_ds)
    sample_image = next(iter(train_ds.take(1)))[0][0]
    sample_image = tf.reshape(sample_image, [1, w * h * channel])
    sample_image = tf.cast(sample_image, dtype=tf.float32)
    sample_image /= 255

    for epoch in range(1, EPOCHS + 1):
        epoch_loss, cnt = 0, 0

        for image, label in train_ds:
            batch_loss = train(image, model)
            batch_loss /= BATCH_SIZE
            epoch_loss += batch_loss
            cnt += 1

        epoch_loss /= cnt
        if visual:
            clear_output(wait=True)
            show_predictions(model, sample_image, w, h, channel)
        print(template.format(epoch, epoch_loss))
