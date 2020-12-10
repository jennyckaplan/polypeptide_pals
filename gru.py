#==============================================================================#
#                                  DEPENDENCIES                                #
#==============================================================================#

import tensorflow as tf
import numpy as np

from tensorflow.keras import Model
from preprocess       import get_lstm_data, get_next_batch

#==============================================================================#
#                                 GRU DEFINITION                               #
#==============================================================================#

class Model(tf.keras.Model):
    def __init__(self, vocab_size):
        """
        The Model class predicts the next tokens in a sequence.

        :param vocab_size: The number of unique tokens in the data
        """
        super(Model, self).__init__()

        # HYPERPARAMETERS =====================================================#
        
        self.vocab_size     = vocab_size
        self.window_size    = 1633
        self.embedding_size = 64
        self.rnn_size       = 150
        self.batch_size     = 128
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
        
        # PARAMETERS/LAYERS ===================================================#
        
        # Embedding layer
        self.E = tf.Variable(tf.random.truncated_normal(
            [self.vocab_size, self.embedding_size], stddev=0.1))
        
        # GRU Cell
        self.gru = tf.keras.layers.GRU(
            self.rnn_size, return_sequences=True, return_state=True)
        
        # Linear Layers
        self.dense1 = tf.keras.layers.Dense(self.rnn_size, activation='relu')
        self.dense2 = tf.keras.layers.Dense(self.vocab_size, activation='softmax')

    def call(self, inputs, initial_state):
        """
        The embedding layer is the fist layer of this network and the next is a LSTM

        :param inputs: token ids of shape (batch_size, window_size)
        :param initial_state: 2-d array of shape (batch_size, rnn_size) as a tensor
        :return: the batch element probabilities as a tensor, a final_state
        The final_state will be the last two RNN outputs, and we only need to 
        use the initial state during generation
        """
        
        # Embed input sequence
        embedding = tf.nn.embedding_lookup(self.E, inputs)
        
        # Pass through RNN cell
        gru_output, final_memory_state, final_carry_state = self.gru(
            embedding, initial_state=initial_state)
        
        # LINEAR LAYERS =======================================================#
        
        layer1 = self.dense1(gru_output)
        probs  = self.dense2(dense_output)
        return probs, (final_memory_state, final_carry_state)

    def loss(self, probs, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction

        :param probs: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the loss of the model as a tensor of size 1
        """
        losses = tf.keras.losses.sparse_categorical_crossentropy(labels, probs)
        
        return tf.reduce_mean(losses)


def train(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: None
    """
    num_inputs = train_inputs.shape[0]

    # separate into window size groups
    num_windows = num_inputs // model.window_size
    input_groups = []
    label_groups = []

    leftover = num_inputs % model.window_size
    
    for i in range(0, num_inputs - leftover, model.window_size):
        input_groups.append(train_inputs[i:i + model.window_size])
        label_groups.append(train_labels[i:i + model.window_size])

    input_groups = np.array(input_groups)
    label_groups = np.array(label_groups)

    for i in range(0, num_windows, model.batch_size):
        inputs_batch, labels_batch = get_next_batch(input_groups, label_groups, i, model.batch_size)

        # Backpropagation - Add loss to gradient tape
        with tf.GradientTape() as tape:
            probs, _ = model.call(inputs_batch, initial_state=None)
            loss     = model.loss(probs, labels_batch)
        
        # Compute and apply gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, test_inputs, test_labels):
    """
    Runs through one epoch - all testing examples

    :param model: the trained model to use for prediction
    :param test_inputs: train inputs (all inputs for testing) of shape (num_inputs,)
    :param test_labels: train labels (all labels for testing) of shape (num_labels,)
    :returns: perplexity of the test set
    """
    batch_losses = 0
    num_inputs = test_inputs.shape[0]
    # separate into window size groups
    num_windows = num_inputs // model.window_size
    input_groups = []
    label_groups = []

    leftover = num_inputs % model.window_size

    for i in range(0, num_inputs - leftover, model.window_size):
        input_groups.append(test_inputs[i:i+model.window_size])
        label_groups.append(test_labels[i:i+model.window_size])

    input_groups = np.array(input_groups)
    label_groups = np.array(label_groups)
    batch_accuracies = 0

    for i in range(0, num_windows, model.batch_size):
        (inputs_batch, labels_batch) = get_next_batch(
            input_groups, label_groups, i, model.batch_size)
        (probs, _) = model.call(inputs_batch, initial_state=None)
        loss = model.loss(probs, labels_batch)
        batch_losses += loss

        predicted_labels = tf.argmax(input=probs, axis=2)
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(predicted_labels, labels_batch), dtype=tf.float32))
        batch_accuracies += accuracy

    num_batches = num_windows // model.batch_size
    average_loss = batch_losses / num_batches
    perplexity = np.exp(np.mean(average_loss))
    accuracy = batch_accuracies / num_batches

    return perplexity, accuracy


def generate_sentence(word1, length, vocab, model, sample_n=10):
    """
    Takes a model, vocab, selects from the most likely next word from the model's distribution

    :param model: trained RNN model
    :param vocab: dictionary, word to id mapping
    :return: None
    """

    reverse_vocab = {idx: word for word, idx in vocab.items()}
    previous_state = None

    first_string = word1
    first_word_index = vocab[word1]
    next_input = [[first_word_index]]
    text = [first_string]

    for i in range(length):
        logits, previous_state = model.call(next_input, previous_state)
        logits = np.array(logits[0, 0, :])
        top_n = np.argsort(logits)[-sample_n:]
        n_logits = np.exp(logits[top_n])/np.exp(logits[top_n]).sum()
        out_index = np.random.choice(top_n, p=n_logits)

        text.append(reverse_vocab[out_index])
        next_input = [[out_index]]

    print(" ".join(text))


def main():
    # Pre-process and vectorize the data
    print("Begin preprocessing...")
    (train_inputs, train_labels, test_inputs, test_labels, vocab_dict) = get_lstm_data(
        "data/pickle/secondary_structure_train.p", "data/pickle/secondary_structure_valid.p")
    print("Preprocessing complete.")

    # make train inputs/labels and test inputs/labels numpy arrays
    train_inputs = np.array(train_inputs, dtype=np.int32)
    train_labels = np.array(train_labels, dtype=np.int32)
    test_inputs = np.array(test_inputs, dtype=np.int32)
    test_labels = np.array(test_labels, dtype=np.int32)

    # initialize model and tensorflow variables
    model = Model(len(vocab_dict))

    print("training")
    # Set-up the training step
    train(model, train_inputs, train_labels)

    print("testing")
    # Set up the testing steps
    perplexity, accuracy = test(model, test_inputs, test_labels)

    # Print out perplexity
    print("Perplexity: {}".format(perplexity))
    # Print accuracy
    print("Accuracy: {}".format(accuracy))


if __name__ == '__main__':
    main()
