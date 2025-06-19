import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv1D, Activation, Dropout, Add, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import Input


@tf.keras.utils.register_keras_serializable()
class TemporalBlock(Layer):
    def __init__(self, filters, kernel_size, dilation_rate, padding="causal", dropout=0.2, activation="relu", **kwargs):
        """A Temporal Convolutional Network (TCN) block implementation.

        This block consists of two dilated causal convolution layers with normalization,
        activation, and dropout, followed by a residual connection. If the input and output
        dimensions don't match, a 1x1 convolution is used to match dimensions.

        Args:
            filters (int): Number of filters in the first convolution layer. The second layer uses twice this number.
            kernel_size (int): Size of the convolutional kernel.
            dilation_rate (int): Dilation rate for the temporal convolutions.
            padding (str, optional): Padding method for convolutions. Defaults to "causal".
            dropout (float, optional): Dropout rate. Defaults to 0.2.
            activation (str, optional): Activation function to use. Defaults to "relu".
            **kwargs: Additional keyword arguments passed to the parent Layer class.
        """
        super(TemporalBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout = dropout

        self.conv1 = Conv1D(filters, kernel_size, padding=padding, dilation_rate=dilation_rate)
        self.norm1 = LayerNormalization()
        self.act1 = Activation(activation)
        self.drop1 = Dropout(dropout)

        self.conv2 = Conv1D(filters * 2, kernel_size, padding=padding, dilation_rate=dilation_rate)
        self.norm2 = LayerNormalization()
        self.act2 = Activation(activation)
        self.drop2 = Dropout(dropout)

        self.downsample = None
        self.add = Add()

    def get_config(self):
        """Returns the configuration of the layer.

        Returns:
            dict: Configuration dictionary containing the layer's parameters.
        """
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
        """Creates a layer from its configuration.

        Args:
            config (dict): Layer configuration dictionary.

        Returns:
            TemporalBlock: A new instance of the layer.
        """
        return cls(**config)


    def build(self, input_shape):
        """Builds the layer based on input shape.

        Creates a downsample convolution if input channels don't match output channels.

        Args:
            input_shape (tuple): Shape of the input tensor.
        """
        if input_shape[-1] != self.conv2.filters:
            self.downsample = Conv1D(self.conv2.filters, 1, strides=self.conv2.strides, padding="same")


    def call(self, inputs):
        """Forward pass of the layer.

        Applies the temporal block operations: two sets of convolution, normalization,
        activation, and dropout, followed by a residual connection.

        Args:
            inputs (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Output tensor after applying the temporal block operations.
        """
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
    """Builds a Temporal Convolutional Network (TCN) model.

    Args:
        input_shape (tuple): Shape of the input data (sequence_length, features).
        filters (int): Number of filters in each temporal block.
        kernel_size (int): Size of the convolutional kernel.
        dilations (list): List of dilation rates for each temporal block.
        num_blocks (int): Number of temporal blocks in the network.
        base_rate (float, optional): Base dropout rate. The actual dropout rate increases
            linearly with block depth. Defaults to 0.0.
        output_units (int, optional): Number of output units in the final dense layer.
            Defaults to 1.

    Returns:
        tf.keras.Model: Compiled TCN model with the specified architecture.
    """
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


def build_tcn_for_oak(input_shape_fake, real_shape, filters, kernel_size, dilations, num_blocks, base_rate=0.0, output_units=1):
    """
    Builds a TCN model compatible with OAK blob conversion.
    This function fakes a 3-channel image input and reshapes it internally to the real input shape.

    Args:
        input_shape_fake (tuple): Fake image-like input shape, e.g., (3, 10, 165)
        real_shape (tuple): Actual shape expected by the TCN, e.g., (50, 99)
        filters, kernel_size, dilations, num_blocks, base_rate, output_units: Same as original build_tcn()

    Returns:
        tf.keras.Model: Wrapped model that is blob-compatible.
    """
    inputs = Input(shape=input_shape_fake, name="oak_input")
    x = tf.keras.layers.Reshape(real_shape, name="reshape_to_sequence")(inputs)

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

    return Model(inputs=inputs, outputs=x, name="TCN_OAK_WRAPPED")
