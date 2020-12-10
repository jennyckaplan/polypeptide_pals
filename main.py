import os
import tensorflow as tf
import numpy as np
from preprocess import *
from transformer_model import Transformer_Seq2Seq
from rnn_model import RNN_Seq2Seq
import sys
import random

from attenvis import AttentionVis
av = AttentionVis()


def train(model, train_primary, train_ss, ss_padding_index):
    """
    Runs through one epoch - all training examples.

    :param model: the initialized model to use for forward and backward pass
    :param train_primary: primary (amino acid seq) train data (all data for training) of shape (num_sentences, window_size)
    :param train_ss: secondary structure train data (all data for training) of shape (num_sentences, window_size)
    :param ss_padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
    :return: None
    """
    num_examples = train_primary.shape[0]
    num_batches = (int)(np.ceil(num_examples / model.batch_size))

    primary_batch = np.asarray(np.array_split(train_primary, num_batches))
    ss_batch = np.asarray(np.array_split(train_ss, num_batches))

    for i in range(num_batches):
        curr_primary = primary_batch[i]
        curr_SS = ss_batch[i]
        ss_batch_inputs = curr_SS[:, 0:-1]
        ss_batch_labels = curr_SS[:, 1:]

        mask = np.where(ss_batch_labels == ss_padding_index, 0, 1)
        with tf.GradientTape() as tape:
            probs = model(curr_primary, ss_batch_inputs)

            loss = model.loss_function(probs, ss_batch_labels, mask)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(
            zip(gradients, model.trainable_variables))


@av.test_func
def test(model, test_primary, test_ss, ss_padding_index):
    """
    Runs through one epoch - all testing examples.

    :param model: the initialized model to use for forward and backward pass
    :param test_primary: primary (amino acid seq) test data (all data for testing) of shape (num_proteins, window_size)
    :param test_ss: secondary structure (ss) test data (all data for testing) of shape (num_proteins, window_size)
    :param ss_padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
    :returns: a tuple containing at index 0 the perplexity of the test set and at index 1 the per symbol accuracy on test set,
    e.g. (my_perplexity, my_accuracy)
    """
    num_examples = test_primary.shape[0]
    num_batches = (int)(np.ceil(num_examples / model.batch_size))

    primary_batch = np.asarray(np.array_split(test_primary, num_batches))
    ss_batch = np.asarray(np.array_split(test_ss, num_batches))

    losses = []
    accuracies = []

    sum_accuracy = 0
    sum_loss = 0
    total_tokens = 0
    for i in range(num_batches):  # going thru all the batches
        curr_primary = primary_batch[i]
        curr_SS = ss_batch[i]
        ss_batch_inputs = curr_SS[:, 0:-1]
        ss_batch_labels = curr_SS[:, 1:]

        probs = model(curr_primary, ss_batch_inputs)

        mask = np.where(ss_batch_labels == ss_padding_index, 0, 1)

        loss = model.loss_function(probs, ss_batch_labels, mask)
        accuracy = model.accuracy_function(probs, ss_batch_labels, mask)

        tokens = np.sum(mask)
        total_tokens = total_tokens + tokens
        sum_loss = sum_loss + loss
        sum_accuracy = sum_accuracy + accuracy*tokens

    perplexity = np.exp(sum_loss / total_tokens)
    accuracy = (sum_accuracy + 0.0) / total_tokens

    print("perplexity: " + str(perplexity))
    print("accuracy: " + str(accuracy))
    return perplexity, accuracy


def main():
    if len(sys.argv) != 4 or sys.argv[1] not in {"RNN", "TRANSFORMER", "LSTM", "GRU"} or sys.argv[2] not in {"ss3", "ss8"} or sys.argv[3] not in {"valid", "casp12", "cb513", "ts115"}:
        print("USAGE: python main.py <Model Type> <Data Type> <Dataset>")
        print("<Model Type>: [RNN/TRANSFORMER/LSTM/GRU]")
        print("<Data Type>: [ss3/ss8]")
        print("<Dataset>: [valid, casp12, cb513, ts115]")
        exit()

    if sys.argv[1] == "TRANSFORMER":
        av.setup_visualization(enable=False)

    data_types = {'ss3': 2, 'ss8': 3}
    data_index = data_types[sys.argv[2]]

    print("Running preprocessing...")
    primary_train, primary_test, ss_train, ss_test, primary_vocab, ss_vocab, ss_pad_tokenID = get_data(
        "data/pickle/secondary_structure_train.p", "data/pickle/secondary_structure_" + sys.argv[3] + ".p", data_index)
    print("Preprocessing complete.")

    model_args = (WINDOW_SIZE, len(primary_vocab), len(ss_vocab))

    if sys.argv[1] == "RNN":
        model = RNN_Seq2Seq(*model_args)
    elif sys.argv[1] == "TRANSFORMER":
        model = Transformer_Seq2Seq(*model_args)

    print("start training")
    train(model, primary_train, ss_train, ss_pad_tokenID)

    print("start testing")
    test(model, primary_test, ss_test, ss_pad_tokenID)

    # Visualize a sample attention matrix from the test set
    # Only takes effect if you enabled visualizations above
    # av.show_atten_heatmap()


if __name__ == '__main__':
    main()
