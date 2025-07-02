import tensorflow as tf

class TemporalConvNet(tf.keras.Model):
    """A lightweight temporal convolutional network used for tests.

    This implementation is intentionally simple and only provides the
    functionality required by the unit tests.
    """

    def __init__(self, mode: str, num_classes: int, num_channels: list[int], kernel_size: int = 3):
        super().__init__()
        self.mode = mode
        self.convs = [
            tf.keras.layers.Conv1D(filters, kernel_size, padding="same", activation="relu")
            for filters in num_channels
        ]
        self.pool = tf.keras.layers.GlobalAveragePooling1D()
        activation = "softmax" if mode == "classification" else None
        self.out = tf.keras.layers.Dense(num_classes, activation=activation)

    def call(self, inputs, training=False):  # noqa: D401
        x = inputs
        for conv in self.convs:
            x = conv(x)
        x = self.pool(x)
        return self.out(x)

