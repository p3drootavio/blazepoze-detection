import tensorflow as tf
from src.blazepoze.pipeline.tnc_model_strong import build_tcn
from src.blazepoze.pipeline.tcn_model_weak import TemporalConvNet


def test_build_tcn():
    model = build_tcn(
        input_shape=(50, 99),
        filters=16,
        kernel_size=3,
        dilations=[1, 2],
        num_blocks=2,
        base_rate=0.1,
        output_units=3,
    )
    assert isinstance(model, tf.keras.Model)
    assert model.output_shape[-1] == 3


def test_temporal_conv_net():
    model = TemporalConvNet("classification", num_classes=4, num_channels=[8, 8])
    dummy = tf.zeros([1, 50, 16], dtype=tf.float32)
    out = model(dummy, training=False)
    assert out.shape == (1, 4)
