import tensorflow as tf
from tensorflow.keras import Input, Model


def build_rnn(
    input_shape: tuple,
    units: int,
    num_layers: int,
    dropout: float = 0.0,
    rnn_type: str = "gru",
    output_units: int = 1,
) -> tf.keras.Model:
    """Build a recurrent neural network for sequence classification.

    Parameters
    ----------
    input_shape : tuple
        Shape of the time series input ``(timesteps, features)``.
    units : int
        Number of units in each recurrent layer.
    num_layers : int
        How many recurrent layers to stack.
    dropout : float, optional
        Dropout rate applied to each recurrent layer, by default ``0.0``.
    rnn_type : str, optional
        Type of RNN cell to use (``"gru"`` or ``"lstm"``), by default ``"gru"``.
    output_units : int, optional
        Number of output classes, by default ``1``.

    Returns
    -------
    tf.keras.Model
        The compiled RNN model.
    """
    inputs = Input(shape=input_shape, dtype=tf.float32)
    x = inputs
    cell = tf.keras.layers.GRU if rnn_type.lower() == "gru" else tf.keras.layers.LSTM

    for _ in range(max(0, num_layers - 1)):
        x = cell(units, return_sequences=True, dropout=dropout)(x)

    x = cell(units, return_sequences=False, dropout=dropout)(x)
    x = tf.keras.layers.Dense(output_units, activation="softmax")(x)
    return Model(inputs=inputs, outputs=x, name="RNN")


def build_rnn_for_oak(
    input_shape_fake: tuple,
    real_shape: tuple,
    units: int,
    num_layers: int,
    dropout: float = 0.0,
    rnn_type: str = "gru",
    output_units: int = 1,
) -> tf.keras.Model:
    """Wrap :func:`build_rnn` for OAK-D deployment.

    Parameters are the same as :func:`build_rnn` with additional
    ``input_shape_fake`` and ``real_shape`` arguments used to reshape the
    input from an image-like blob into a sequence.
    """
    inputs = Input(shape=input_shape_fake, name="oak_input")
    x = tf.keras.layers.Reshape(real_shape, name="reshape_to_sequence")(inputs)

    cell = tf.keras.layers.GRU if rnn_type.lower() == "gru" else tf.keras.layers.LSTM
    for _ in range(max(0, num_layers - 1)):
        x = cell(units, return_sequences=True, dropout=dropout)(x)

    x = cell(units, return_sequences=False, dropout=dropout)(x)
    x = tf.keras.layers.Dense(output_units, activation="softmax")(x)
    return Model(inputs=inputs, outputs=x, name="RNN_OAK_WRAPPED")

