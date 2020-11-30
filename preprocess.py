import tensorflow as tf

def deserialize_secondary_structure(example):
    context = {
        'id': tf.FixedLenFeature([], tf.string),
        'protein_length': tf.FixedLenFeature([], tf.int64)
    }

    features = {
        'primary': tf.FixedLenSequenceFeature([], tf.int64),
        'evolutionary': tf.FixedLenSequenceFeature([30], tf.float32),
        'ss3': tf.FixedLenSequenceFeature([], tf.int64),
        'ss8': tf.FixedLenSequenceFeature([], tf.int64),
        'disorder': tf.FixedLenSequenceFeature([], tf.int64),
        'interface': tf.FixedLenSequenceFeature([], tf.int64),
        'phi': tf.FixedLenSequenceFeature([], tf.float32),
        'psi': tf.FixedLenSequenceFeature([], tf.float32),
        'rsa': tf.FixedLenSequenceFeature([], tf.float32),
        'asa_max': tf.FixedLenSequenceFeature([], tf.float32),
        'valid_mask': tf.FixedLenSequenceFeature([], tf.float32)
    }

    context, features = tf.parse_single_sequence_example(
        example,
        context_features=context,
        sequence_features=features
    )

    features.update(context)

    for name, feature in features.items():
        if feature.dtype == tf.int64:
            features[name] = tf.cast(feature, tf.int32)

    return features