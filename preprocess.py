import numpy as np
import tensorflow as tf
import numpy as np
import pickle

from attenvis import AttentionVis
av = AttentionVis()

PAD_TOKEN = "*PAD*"
STOP_TOKEN = "*STOP*"
START_TOKEN = "*START*"
UNK_TOKEN = "*UNK*"
WINDOW_SIZE = 1633


def pad_corpus(primary, secondary_structure):
    """
    Arguments are lists of primary, secondary_structure sequences/labels. The
    text is given an initial "*STOP*".  All sentences are padded with "*STOP*" at
    the end.

    :param primary: list of primary sequences
    :param english: list of secondary structure sequences
    :return: A tuple of: (list of padded sequences for ss, list of padded sequences for primary)
    """
    primary_padded_lines = []
    for line in primary:
        padded_primary_item = line[:WINDOW_SIZE]
        padded_primary_item += [STOP_TOKEN] + [PAD_TOKEN] * \
            (WINDOW_SIZE - len(padded_primary_item)-1)
        primary_padded_lines.append(padded_primary_item)

    ss_padded_lines = []
    for line in secondary_structure:
        padded_ss_item = line[:WINDOW_SIZE]
        padded_ss_item = [START_TOKEN] + padded_ss_item + [STOP_TOKEN] + \
            [PAD_TOKEN] * (WINDOW_SIZE - len(padded_ss_item)-1)
        ss_padded_lines.append(padded_ss_item)

    return np.array(primary_padded_lines), np.array(ss_padded_lines)


def build_vocab(sentences):
    """

    Builds vocab from list of sequences

    :param sentences:  list of sequences, each a list of tokens
    :return: tuple of (dictionary: word --> unique index, pad_token_idx)
    """
    tokens = []
    for s in sentences:
        tokens.extend(str(s))
    all_tokens = sorted(
        list(set([STOP_TOKEN, PAD_TOKEN, UNK_TOKEN] + tokens)))

    vocab = {token: i for i, token in enumerate(all_tokens)}

    return vocab, vocab[PAD_TOKEN]


def build_lstm_vocab(tokens):
    """

    Builds vocab from list of sequences

    :param sentences:  list of sequences, each a list of tokens
    :return: tuple of (dictionary: word --> unique index, pad_token_idx)
    """
    all_tokens = list(set(tokens))

    vocab = {token: i for i, token in enumerate(all_tokens)}

    return vocab


def convert_to_id(vocab, sentences):
    """
    Convert sentences to indexed

    :param vocab:  dictionary, word --> unique index
    :param sentences:  list of lists of words, each representing padded sentence
    :return: numpy array of integers, with each row representing the word indeces in the corresponding sentences
    """
    return np.stack([[vocab[word] if word in vocab else vocab[UNK_TOKEN] for word in sentence] for sentence in sentences])


@av.get_data_func
def get_data(training_pickle, testing_pickle, data_idx: int):
    """
    Reads and parses training and test data, then pad the corpus.
    Then vectorize your train and test data based on your vocabulary dictionaries.

    :param training_file: Path to the training data file.
    :param testing_file: Path to the testing data file.

    :return: Tuple of train containing:
    (2-d list or array with english training sentences in vectorized/id form [num_sentences x 15] ),
    (2-d list or array with english test sentences in vectorized/id form [num_sentences x 15]),
    (2-d list or array with french training sentences in vectorized/id form [num_sentences x 14]),
    (2-d list or array with french test sentences in vectorized/id form [num_sentences x 14]),
    english vocab (Dict containg word->index mapping),
    french vocab (Dict containg word->index mapping),
    english padding ID (the ID used for *PAD* in the English vocab. This will be used for masking loss)
    """

    # Read primary and ss sequence data for training and testing
    training_data = pickle.load(open(training_pickle, "rb"))
    testing_data = pickle.load(open(testing_pickle, "rb"))

    training_primary = training_data[:, 1]
    training_ss3 = training_data[:, data_idx]

    testing_primary = testing_data[:, 1]
    testing_ss3 = testing_data[:, data_idx]

    # Pad training data
    padded_primary_train, padded_ss_train = pad_corpus(
        training_primary, training_ss3)

    # Pad testing data
    padded_primary_test, padded_ss_test = pad_corpus(
        testing_primary, testing_ss3)

    # Build vocab for ss
    primary_vocab, primary_pad_tokenID = build_vocab(padded_primary_train)

    # Build vocab for primary sequences
    ss_vocab, ss_pad_token_id = build_vocab(padded_ss_train)

    # Convert training and testing primary sequences to list of IDS
    primary_train_vec = np.array(convert_to_id(
        primary_vocab, padded_primary_train))
    primary_test_vec = np.array(convert_to_id(
        primary_vocab, padded_primary_test))

    # Convert training and testing secondary structures to list of IDS
    ss_train_vec = np.array(convert_to_id(ss_vocab, padded_ss_train))
    ss_test_vec = np.array(convert_to_id(ss_vocab, padded_ss_test))

    return primary_train_vec, primary_test_vec, ss_train_vec, ss_test_vec, primary_vocab, ss_vocab, ss_pad_token_id


def get_lstm_data(train_file, test_file, data_idx):
    """
    Read and parse the train and test file line by line, then tokenize the sentences to build the train and test data separately.
    Create a vocabulary dictionary that maps all the unique tokens from your train and test data as keys to a unique integer value.
    Then vectorize your train and test data based on your vocabulary dictionary.

    :param train_file: Path to the training file.
    :param test_file: Path to the test file.
    :return: Tuple of train (1-d list or array with training words in vectorized/id form), test (1-d list or array with testing words in vectorized/id form), vocabulary (Dict containg index->word mapping)
    """
    training_data = pickle.load(open(train_file, "rb"))
    testing_data = pickle.load(open(test_file, "rb"))

    training_primary = training_data[:, 1]
    training_ss3 = training_data[:, data_idx]

    testing_primary = testing_data[:, 1]
    testing_ss3 = testing_data[:, data_idx]

    # Build vocab
    def flatten(t): return [item for sublist in t for item in sublist]
    training_primary = flatten(training_primary)
    training_ss3 = flatten(training_ss3)
    testing_primary = flatten(testing_primary)
    testing_ss3 = flatten(testing_ss3)

    tokens = np.concatenate((training_primary, training_ss3))
    tokens = np.concatenate((tokens, testing_primary))
    tokens = np.concatenate((tokens, testing_ss3))
    vocab = build_lstm_vocab(tokens)

    train_inputs = np.array([vocab[token] for token in training_primary])
    train_labels = np.array([vocab[token] for token in training_ss3])

    test_inputs = np.array([vocab[token] for token in testing_primary])
    test_labels = np.array([vocab[token] for token in testing_ss3])

    return (train_inputs, train_labels, test_inputs, test_labels, vocab)


def get_next_batch(inputs, labels, start, batch_size):
    """
    Helper function for batching
    Returns a slice of inputs and slice of corresponding labels
    :param inputs: NumPy inputs array
    :param labels: NumPy labels array
    :param start: starting index for the slice
    :param batch_size: number of examples desired for the batch
    :return: NumPy array of batched inputs and labels
    """
    end = start + batch_size
    return (inputs[start:end], labels[start:end])
