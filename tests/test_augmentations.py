import tensorflow as tf
from src.blazepoze.utils import augment


def test_gaussian_noise():
    aug = augment.adding_gaussian_noise(prob=1.0, stddev=0.1)
    x = tf.zeros([1, 50, 99], dtype=tf.float32)
    y = tf.constant([0])
    nx, ny = aug(x, y)
    assert nx.shape == x.shape
    assert int(ny.numpy()[0]) == 0


def test_scaling():
    aug = augment.adding_scaling(prob=1.0, scale_range=(0.5, 0.5))
    x = tf.ones([1, 50, 99], dtype=tf.float32)
    y = tf.constant([1])
    sx, sy = aug(x, y)
    assert sx.shape == x.shape
    assert float(tf.reduce_max(sx).numpy()) <= 1.0
    assert int(sy.numpy()[0]) == 1


def test_shifts():
    aug = augment.adding_shifts(prob=1.0, shift_range=(-0.1, 0.1))
    x = tf.zeros([1, 50, 99], dtype=tf.float32)
    y = tf.constant([2])
    sx, sy = aug(x, y)
    assert sx.shape == x.shape
    assert int(sy.numpy()[0]) == 2
