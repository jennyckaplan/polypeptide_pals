import tensorflow as tf


def deserialize_secondary_structure(example):
    context = {
        'id': tf.io.FixedLenFeature([], tf.string),
        'protein_length': tf.io.FixedLenFeature([], tf.int64)
    }

    features = {
        'primary': tf.io.FixedLenSequenceFeature([], tf.int64),
        'evolutionary': tf.io.FixedLenSequenceFeature([30], tf.float32),
        'ss3': tf.io.FixedLenSequenceFeature([], tf.int64),
        'ss8': tf.io.FixedLenSequenceFeature([], tf.int64),
        'disorder': tf.io.FixedLenSequenceFeature([], tf.int64),
        'interface': tf.io.FixedLenSequenceFeature([], tf.int64),
        'phi': tf.io.FixedLenSequenceFeature([], tf.float32),
        'psi': tf.io.FixedLenSequenceFeature([], tf.float32),
        'rsa': tf.io.FixedLenSequenceFeature([], tf.float32),
        'asa_max': tf.io.FixedLenSequenceFeature([], tf.float32),
        'valid_mask': tf.io.FixedLenSequenceFeature([], tf.float32)
    }

    context, features = tf.io.parse_single_sequence_example(
        example,
        context_features=context,
        sequence_features=features
    )

    features.update(context)

    for name, feature in features.items():
        if feature.dtype == tf.int64:
            features[name] = tf.cast(feature, tf.int32)

    return features
