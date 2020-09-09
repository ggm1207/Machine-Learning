import tensorflow as tf

from AE import AE
from AE_undercomplete import AE_UC
from data import load_mnist

EPOCHS = 15
lr = 0.01


def loss(model, x, y):
    y_ = model(x)
    return loss_object(y_true=y, y_pred=y_)


def grad(model, x, y):
    with tf.GradientTape() as tape:
        loss_value = loss(model, x, y)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def train(model, train_ds, optimizer):
    for epoch in range(EPOCHS):
        epoch_loss_avg = tf.keras.metrics.Mean()

        for x, _ in train_ds:
            loss_value, grads = grad(model, x, x)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            epoch_loss_avg(loss_value)

        print(
            "Epoch {:03d}: Loss: {:.3f}".format(epoch, epoch_loss_avg.result())
        )


if __name__ == "__main__":

    loss_object = tf.keras.losses.MSE
    train_ds, test_ds = load_mnist(flatten=True)
    for m_cls in AE.__subclasses__():
        model = m_cls()
        optimizer = tf.keras.optimizers.Adam(lr)
        train(model, train_ds, optimizer)
        del model
