import tensorflow as tf


def adding_gaussian_noise(prob=0.5, mean=0.0, stddev=0.01, clip_values=True):
    """Apply Gaussian noise to the input with given probability.

    Args:
        prob (float): Probability of applying the augmentation (0 to 1)
        mean (float): Mean of the Gaussian noise
        stddev (float): Standard deviation of the Gaussian noise
        clip_values (bool): Whether to clip values to [0, 1] range

    Returns:
        A function that performs the augmentation
    """

    def _validate_inputs():
        tf.debugging.assert_greater_equal(prob, 0.0, message="`prob` must be greater or equal than 0.")
        tf.debugging.assert_less_equal(prob, 1.0, message="`prob` must be less or equal than 1.")
        tf.debugging.assert_greater_equal(stddev, 0.0, message="`stddev` must be greater or equal than 0.")

    @tf.function
    def augment(x, y):
        _validate_inputs()
        x = tf.cast(x, tf.float32)

        def apply_noise():
            noise = tf.random.normal(tf.shape(x), mean=mean, stddev=stddev, dtype=tf.float32)
            noised = x + noise
            if clip_values:
                noised = tf.clip_by_value(noised, 0.0, 1.0)
            return noised, y

        return tf.cond(tf.random.uniform([]) < prob, apply_noise, lambda: (x, y))

    return augment


def adding_scaling(prob=0.5, scale_range=(0.9, 1.1), clip_values=True):
    """Apply random scaling to the input with given probability.

    Args:
        prob (float): Probability of applying the augmentation (0 to 1)
        scale_range (tuple): Range for random scaling (min, max)
        clip_values (bool): Whether to clip values to [0, 1] range

    Returns:
        A function that performs the augmentation
    """

    def _validate_inputs():
        tf.debugging.assert_greater_equal(prob, 0.0, message="`prob` must be greater or equal than 0.")
        tf.debugging.assert_less_equal(prob, 1.0, message="`prob` must be less or equal than 1.")
        tf.debugging.assert_greater_equal(scale_range[0], 0.0, message="`scale_range[0]` must be greater or equal than 0.")
        tf.debugging.assert_greater_equal(scale_range[1], 0.0, message="`scale_range[1]` must be greater or equal than 0.")
        tf.debugging.assert_less_equal(scale_range[0], scale_range[1], message="`scale_range[0]` must be less than `scale_range[1]`")

    @tf.function
    def augment(x, y):
        _validate_inputs()
        x = tf.cast(x, tf.float32)

        def apply_scaling():
            scale = tf.random.uniform([1], scale_range[0], scale_range[1])
            scaled = x * scale
            if clip_values:
                scaled = tf.clip_by_value(scaled, 0.0, 1.0)
            return scaled, y

        return tf.cond(tf.random.uniform([]) < prob, apply_scaling, lambda: (x, y))

    return augment


def adding_shifts(prob=0.5, shift_range=(-0.05, 0.05), clip_values=True):
    """Apply random shifts to the input with given probability.

    Args:
        prob (float): Probability of applying the augmentation (0 to 1)
        shift_range (tuple): Range for random shifts (min, max)
        clip_values (bool): Whether to clip values to [0, 1] range

    Returns:
        A function that performs the augmentation
    """
    def _validate_inputs():
        tf.debugging.assert_greater_equal(prob, 0.0, message="`prob` must be greater or equal than 0.")
        tf.debugging.assert_less_equal(prob, 1.0, message="`prob` must be less or equal than 1.")
        tf.debugging.assert_less(shift_range[0], shift_range[1], message="`shift_range[0]` must be less than `shift_range[1]`")

    @tf.function
    def augment(x, y):
        _validate_inputs()
        x = tf.cast(x, tf.float32)

        def apply_shift():
            shift = tf.random.uniform(tf.shape(x), shift_range[0], shift_range[1])
            shifted = x + shift
            if clip_values:
                shifted = tf.clip_by_value(shifted, 0.0, 1.0)
            return shifted, y

        return tf.cond(tf.random.uniform([]) < prob, apply_shift, lambda: (x, y))

    return augment
