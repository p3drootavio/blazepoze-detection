import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Layer, Conv1D, Activation, Dropout, Add, UpSampling1D, MaxPooling1D, LayerNormalization

@tf.keras.utils.register_keras_serializable()
class EDTCNBlock(Layer):
    def __init__(self, filters, kernel_size, dropout=0.2, activation="relu", causal=True, **kwargs):
        """
        Encoder-Decoder TCN Block:
        One temporal convolution layer with normalization, activation, dropout, and optional causal padding.
        """
        super(EDTCNBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.activation = activation
        self.causal = causal

        self.conv = Conv1D(filters, kernel_size, padding='causal' if causal else 'same')
        self.norm = LayerNormalization()
        self.act = Activation(activation)
        self.drop = Dropout(dropout)

    def call(self, inputs):
        """Applies Conv1D → Norm → Activation → Dropout"""
        x = self.conv(inputs)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "dropout": self.dropout,
            "activation": self.activation,
            "causal": self.causal,
        })
        return config


def build_ed_tcn(input_shape, filters, kernel_size, num_layers, base_dropout=0.2, output_units=1, causal=True):
    """
    Builds an Encoder-Decoder TCN model.

    Args:
        input_shape (tuple): Input shape (timesteps, features)
        filters (int): Base number of filters per layer
        kernel_size (int): Size of convolutional filters
        num_layers (int): Number of encoder (and decoder) layers
        base_dropout (float): Base dropout rate
        output_units (int): Number of output classes
        causal (bool): Use causal convolutions for real-time inference

    Returns:
        tf.keras.Model: The ED-TCN model
    """
    inputs = Input(shape=input_shape, name="ed_input")
    x = inputs
    skips = []

    # ----- Encoder -----
    for i in range(num_layers):
        f = filters * (i + 1)
        x = EDTCNBlock(filters=f, kernel_size=kernel_size, dropout=base_dropout, causal=causal)(x)
        skips.append(x)  # Save for skip connection if needed
        x = MaxPooling1D(pool_size=2)(x)

    # ----- Decoder -----
    for i in reversed(range(num_layers)):
        f = filters * (i + 1)
        x = UpSampling1D(size=2)(x)
        x = EDTCNBlock(filters=f, kernel_size=kernel_size, dropout=base_dropout, causal=False)(x)

    # ----- Output Layer -----
    # Apply a final temporal Conv1D followed by softmax to predict per timestep class probabilities
    x = Conv1D(output_units, 1, activation='softmax', padding='same')(x)

    return Model(inputs=inputs, outputs=x, name="ED_TCN")


def build_ed_tcn_for_oak(input_shape_fake, real_shape, filters, kernel_size, num_layers, base_dropout=0.2, output_units=1):
    """
    Wraps ED-TCN for OAK-D deployment with reshaping from image-like to sequence data.

    Args:
        input_shape_fake (tuple): Shape from OAK blob (e.g. (3, 10, 165))
        real_shape (tuple): Actual shape of time series input (e.g. (50, 99))
        filters, kernel_size, num_layers, base_dropout, output_units: Same as base model

    Returns:
        tf.keras.Model: Wrapped ED-TCN model
    """
    inputs = Input(shape=input_shape_fake, name="oak_input")
    x = tf.keras.layers.Reshape(real_shape, name="reshape_to_sequence")(inputs)

    # Apply ED-TCN
    for i in range(num_layers):
        f = filters * (i + 1)
        x = EDTCNBlock(filters=f, kernel_size=kernel_size, dropout=base_dropout, causal=True)(x)
        x = MaxPooling1D(pool_size=2)(x)

    for i in reversed(range(num_layers)):
        f = filters * (i + 1)
        x = UpSampling1D(size=2)(x)
        x = EDTCNBlock(filters=f, kernel_size=kernel_size, dropout=base_dropout, causal=False)(x)

    # Predict one class per time step
    x = Conv1D(output_units, 1, activation='softmax', padding='same')(x)

    return Model(inputs=inputs, outputs=x, name="ED_TCN_OAK_WRAPPED")
