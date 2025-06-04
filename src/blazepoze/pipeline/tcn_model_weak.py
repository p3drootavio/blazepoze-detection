# Third-party libraries
import tensorflow as tf
from keras import Input
from tensorflow.keras.layers import Layer, Conv1D, Activation, Dropout, Add, LayerNormalization
from tensorflow.keras.models import Model


class ResidualBlock(Layer):
    def __init__(self, dilation_rate: int, nb_filters: int, kernel_size: int, padding: str = "causal", activation: str = 'relu', dropout_rate: float = 0.2, **kwargs):

        # Residual Block 1
        self.conv1 = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding)
        self.activation1 = Activation(activation)
        self.dropout1 = Dropout(dropout_rate)
        self.norm1 = LayerNormalization()

        # Residual Block 2
        self.conv2 = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding)
        self.activation2 = Activation(activation)
        self.dropout2 = Dropout(dropout_rate)
        self.norm2 = LayerNormalization()

        self.downsample = None
        self.add = Add()

        super(ResidualBlock, self).__init__(**kwargs)


    def build(self, input_shape):
        if input_shape[-1] != self.conv1.filters:
            self.downsample = Conv1D(filters=self.conv1.filters, kernel_size=1, padding="same")
        else:
            self.downsample = lambda x: x


    def call(self, inputs, training=False):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.activation1(x)
        x = self.dropout1(x, training=training)
        x = self.norm1(x)

        x = self.conv2(x)
        x = self.activation2(x)
        x = self.dropout2(x, training=training)
        x = self.norm2(x)

        return self.add([x, residual])


class TemporalConvNet(Model):
    def __init__(self, model_type, num_classes, num_channels, kernel_size=3, dropout_rate=0.2, **kwargs):
        super(TemporalConvNet, self).__init__(**kwargs)

        if model_type not in ("classification", "regression"):
            raise ValueError(f"Unknown model type: {model_type}. It must be either 'classification' or 'regression'")
        self.model_type = model_type

        self.tcn_blocks = []
        for i, out_channels in enumerate(num_channels):
            dilation_rate = 2 ** i
            self.tcn_blocks.append(ResidualBlock(dilation_rate, out_channels, kernel_size, dropout_rate=dropout_rate, **kwargs))

        if self.model_type == "classification":
            self.final_layer = tf.keras.layers.Dense(num_classes, activation="softmax")
        elif self.model_type == "regression":
            self.final_layer = tf.keras.layers.Dense(1, activation="linear")


    def call(self, inputs, training=False):
        x = inputs
        for block in self.tcn_blocks:
            x = block(x, training=training)
        return self.final_layer(x[:, -1, :]) # Only use the last time step's features
