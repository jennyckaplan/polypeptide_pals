import numpy as np
import tensorflow as tf
import pdb

#french = primary
#english = ss

class RNN_Seq2Seq(tf.keras.Model):
	def __init__(self, WINDOW_SIZE, primary_vocab_size, ss_vocab_size):
		###### DO NOT CHANGE ##############
		super(RNN_Seq2Seq, self).__init__()
		self.primary_vocab_size = primary_vocab_size # The size of the primary (amino acid)
		self.ss_vocab_size = ss_vocab_size # The size of the secondary structure vocab

		self.window_size = WINDOW_SIZE # The window size (same for primary and secondary)
		# self.english_window_size = english_window_size # The english window size
		######^^^ DO NOT CHANGE ^^^##################

		# TODO:
		# 1) Define any hyperparameter
		# Define batch size and optimizer/learning rate
		self.batch_size = 100 # You can change this
		self.embedding_size = 50 # You should change this

		self.optimizer = tf.keras.optimizers.Adam(0.01)
		self.rnn_size = 128

		# 2) Define embeddings, encoder, decoder, and feed forward layers
		#embedding matrix (one for each language)
		self.E_primary = tf.Variable(tf.random.normal([self.primary_vocab_size, self.embedding_size], stddev=.1, dtype=tf.float32))
		self.E_ss = tf.Variable(tf.random.normal([self.ss_vocab_size, self.embedding_size], stddev=.1, dtype=tf.float32))

		self.encoder = tf.keras.layers.GRU(self.rnn_size, return_sequences=True, return_state=True)
		self.decoder = tf.keras.layers.GRU(self.rnn_size, return_sequences=True, return_state=True)
		self.dense = tf.keras.layers.Dense(self.ss_vocab_size, activation='softmax')

	@tf.function
	def call(self, encoder_input, decoder_input):
		"""
		:param encoder_input: batched ids corresponding to primary (amino acid seq)
		:param decoder_input: batched ids corresponding to secondary structure sentences
		:return prbs: The 3d probabilities as a tensor, [batch_size x window_size x ss_vocab_size]
		"""


		# TODO:
		#1) Pass your french sentence embeddings to your encoder
		#2) Pass your english sentence embeddings, and final state of your encoder, to your decoder
		#3) Apply dense layer(s) to the decoder out to generate probabilities
		primary_embeddings = tf.nn.embedding_lookup(self.E_primary, encoder_input)
		ss_embeddings = tf.nn.embedding_lookup(self.E_ss, decoder_input)


		encoder_out, encoder_final_state = self.encoder(primary_embeddings)
		decoder_output_seq, decoder_state = self.decoder(ss_embeddings, initial_state=encoder_final_state)

		probs = self.dense(decoder_output_seq)

		return probs




	def accuracy_function(self, prbs, labels, mask):
		"""
		DO NOT CHANGE

		Computes the batch accuracy

		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: scalar tensor of accuracy of the batch between 0 and 1
		"""

		decoded_symbols = tf.argmax(input=prbs, axis=2)
		accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32),mask))
		return accuracy


	def loss_function(self, prbs, labels, mask):
		"""
		Calculates the total model cross-entropy loss after one forward pass.
		Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.

		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: the loss of the model as a tensor
		"""
		loss = tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(labels, prbs)*mask) #how to mask????
		return loss
