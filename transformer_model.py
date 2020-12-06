import numpy as np
import tensorflow as tf
import transformer_funcs as transformer

from attenvis import AttentionVis
av = AttentionVis()


class Transformer_Seq2Seq(tf.keras.Model):
    def __init__(self, window_size, primary_vocab_size, ss_vocab_size):
        super(Transformer_Seq2Seq, self).__init__()

        self.primary_vocab_size = primary_vocab_size  # The size of the primary vocab
        self.ss_vocab_size = ss_vocab_size  # The size of the ss vocab

        self.window_size = window_size  # The window size

        self.learning_rate = 1e-2
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        self.batch_size = 10
        self.embedding_size = 32

        # Define primary and ss embedding layers:
        self.E_primary = tf.Variable(tf.random.normal(
            [self.primary_vocab_size, self.embedding_size], stddev=.1))
        self.E_ss = tf.Variable(tf.random.normal(
            [self.ss_vocab_size, self.embedding_size], stddev=.1))

        # Create positional encoder layers
        self.pos_encoder_primary = transformer.Position_Encoding_Layer(
            self.window_size, self.embedding_size)
        self.pos_encoder_ss = transformer.Position_Encoding_Layer(
            self.window_size, self.embedding_size)

        # Define encoder and decoder layers:
        self.encoder_1 = transformer.Transformer_Block(
            self.embedding_size, False)
        self.decoder_1 = transformer.Transformer_Block(
            self.embedding_size, True)
        self.decoder_2 = transformer.Transformer_Block(
            self.embedding_size, True)

        # Define dense layer(s)
        self.dense = tf.keras.layers.Dense(
            self.ss_vocab_size, activation='softmax')

    @tf.function
    def call(self, encoder_input, decoder_input):
        """
        :param encoder_input: batched ids corresponding to primary (amino acid seqs)
        :param decoder_input: batched ids corresponding to ss
        :return prbs: The 3d probabilities as a tensor, [batch_size x window_size x ss_vocab_size]
        """
        # Adds the positional embeddings to primary seq (ammino acid seq)
        primary_embeddings = tf.nn.embedding_lookup(
            self.E_primary, encoder_input)
        primary_embeddings = self.pos_encoder_primary(primary_embeddings)

        # Passes the primary seq (amino acid seq) embeddings to the encoder
        encoder_output = self.encoder_1(primary_embeddings)

        # Adds positional embeddings to the secondary structure (ss) embeddings
        ss_embeddings = tf.nn.embedding_lookup(self.E_ss, decoder_input)
        ss_embeddings = self.pos_encoder_ss(ss_embeddings)

        # Passes the secondary structure (ss) embeddings and output of your encoder, to the decoder
        decoder_output_1 = self.decoder_1(ss_embeddings, encoder_output)
        decoder_output_2 = self.decoder_2(decoder_output_1, encoder_output)

        probs = self.dense(decoder_output_2)

        return probs

    def accuracy_function(self, prbs, labels, mask):
        """
        Computes the batch accuracy

        :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x ss_vocab_size]
        :param labels:  integer tensor, word prediction labels [batch_size x window_size]
        :param mask:  tensor that acts as a padding mask [batch_size x window_size]
        :return: scalar tensor of accuracy of the batch between 0 and 1
        """

        decoded_symbols = tf.argmax(input=prbs, axis=2)
        accuracy = tf.reduce_mean(tf.boolean_mask(
            tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32), mask))
        return accuracy

    def loss_function(self, prbs, labels, mask):
        """
        Calculates the model cross-entropy loss after one forward pass

        :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x ss_vocab_size]
        :param labels:  integer tensor, word prediction labels [batch_size x window_size]
        :param mask:  tensor that acts as a padding mask [batch_size x window_size]
        :return: the loss of the model as a tensor
        """
        loss = tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(
            labels, prbs)*mask)
        return loss

    @av.call_func
    def __call__(self, *args, **kwargs):
        return super(Transformer_Seq2Seq, self).__call__(*args, **kwargs)
