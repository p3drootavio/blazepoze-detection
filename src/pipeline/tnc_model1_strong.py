import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv1D, Activation, Dropout, Add, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import Input

@tf.keras.utils.register_keras_serializable()
class TemporalBlock(Layer):
    def __init__(self, filters, kernel_size, dilation_rate, padding="causal", dropout=0.2, activation="relu", **kwargs):
        super(TemporalBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout = dropout

        self.conv1 = Conv1D(filters[0], kernel_size, padding=padding, dilation_rate=dilation_rate)
        self.norm1 = LayerNormalization()
        self.act1 = Activation(activation)
        self.drop1 = Dropout(dropout)

        self.conv2 = Conv1D(filters[1], kernel_size, padding=padding, dilation_rate=dilation_rate)
        self.norm2 = LayerNormalization()
        self.act2 = Activation(activation)
        self.drop2 = Dropout(dropout)

        self.downsample = None
        self.add = Add()

    def get_config(self):
        config = super(TemporalBlock, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dilation_rate': self.dilation_rate,
            'dropout': self.dropout
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


    def build(self, input_shape):
        if input_shape[-1] != self.conv2.filters:
            self.downsample = Conv1D(self.conv2.filters, 1, strides=self.conv2.strides, padding="same")


    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.drop2(x)

        res = inputs
        if self.downsample:
            res = self.downsample(res)

        return self.add([x, res])


def build_tcn(input_shape, filters, kernel_size, dilations, num_blocks, base_rate=0.0, output_units=1):
    inputs = Input(shape=input_shape, dtype=tf.float32)
    x = inputs

    for i in range(num_blocks):
        dropout_rate = base_rate * (i / num_blocks)
        x = TemporalBlock(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilations[i],
            dropout=dropout_rate
        )(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(output_units, activation='softmax')(x)

    return Model(inputs=inputs, outputs=x, name="TCN")
