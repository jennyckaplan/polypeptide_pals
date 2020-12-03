import tensorflow as tf

def classification_loss_and_accuracy(labels, logits, weights=None):
    weights = 1 if weights is None else tf.cast(weights, logits.dtype)

    predictions = tf.argmax(logits, -1, output_type=labels.dtype)
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels, logits, weights)
    # TODO: decipher/change this (https://github.com/CannyLab/rinokeras/blob/c570ba8704c6d79934246732940186c1e007294c/rinokeras/core/v1x/utils/metrics/accuracy.py)
    correct = tf.equal(labels, predictions)
    weights = tf.ones_like(correct) * weights
    accuracy = tf.reduce_sum(correct * weights) / (tf.reduce_sum(weights) + 1e-10)
    return loss, accuracy