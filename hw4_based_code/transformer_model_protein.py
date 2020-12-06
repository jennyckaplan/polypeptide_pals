import numpy as np
import tensorflow as tf
import transformer_funcs_protein as transformer

from attenvis import AttentionVis

av = AttentionVis()
#1 encoder, 2 decoder, embeddings for primary and secondary, then 2 positional embedding for encoder/decoder

#french = primary
#english = ss

class Transformer_Seq2Seq(tf.keras.Model):
	def __init__(self, window_size, primary_vocab_size, ss_vocab_size):

		######vvv DO NOT CHANGE vvv##################
		super(Transformer_Seq2Seq, self).__init__()

		self.primary_vocab_size = primary_vocab_size # The size of the primary vocab
		self.ss_vocab_size = ss_vocab_size # The size of the ss vocab

		self.window_size = window_size # The window size
		######^^^ DO NOT CHANGE ^^^##################


		# TODO:
		# 1) Define any hyperparameters
		# 2) Define embeddings, encoder, decoder, and feed forward layers

		self.optimizer = tf.keras.optimizers.Adam(0.001)

		# Define batch size and optimizer/learning rate
		self.batch_size = 100
		self.embedding_size = 32

		# Define primary and ss embedding layers:
		self.E_primary = tf.Variable(tf.random.normal([self.primary_vocab_size, self.embedding_size], stddev=.1))
		self.E_ss = tf.Variable(tf.random.normal([self.ss_vocab_size, self.embedding_size], stddev=.1))

		# Create positional encoder layers
		self.pos_encoder_primary = transformer.Position_Encoding_Layer(self.window_size, self.embedding_size)
		self.pos_encoder_ss = transformer.Position_Encoding_Layer(self.window_size, self.embedding_size)

		# Define encoder and decoder layers:
		self.encoder_1 = transformer.Transformer_Block(self.embedding_size, False)
		self.decoder_1 = transformer.Transformer_Block(self.embedding_size, True)
		self.decoder_2 = transformer.Transformer_Block(self.embedding_size, True)

		# Define dense layer(s)
		self.dense = tf.keras.layers.Dense(self.ss_vocab_size, activation='softmax')


	@tf.function
	def call(self, encoder_input, decoder_input):
		"""
		:param encoder_input: batched ids corresponding to primary (amino acid seqs)
		:param decoder_input: batched ids corresponding to ss
		:return prbs: The 3d probabilities as a tensor, [batch_size x window_size x ss_vocab_size]
		"""
		# TODO:
		#1) Add the positional embeddings to primary seq (ammino acid seq)
		primary_embeddings = tf.nn.embedding_lookup(self.E_primary, encoder_input)
		primary_embeddings = self.pos_encoder_primary(primary_embeddings)

		#2) Pass the primary seq (amino acid seq) embeddings to the encoder
		encoder_output = self.encoder_1(primary_embeddings)

		#3) Add positional embeddings to the secondary structure (ss) embeddings
		ss_embeddings = tf.nn.embedding_lookup(self.E_ss, decoder_input)
		ss_embeddings = self.pos_encoder_ss(ss_embeddings)

		#4) Pass the secondary structure (ss) embeddings and output of your encoder, to the decoder

		decoder_output_1 = self.decoder_1(ss_embeddings, encoder_output)
		decoder_output_2 = self.decoder_2(decoder_output_1, encoder_output)

		#5) Apply dense layer(s) to the decoder out to generate probabilities
		probs = self.dense(decoder_output_2)

		return probs

	def accuracy_function(self, prbs, labels, mask):
		"""
		DO NOT CHANGE

		Computes the batch accuracy

		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x ss_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: scalar tensor of accuracy of the batch between 0 and 1
		"""

		decoded_symbols = tf.argmax(input=prbs, axis=2)
		accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32),mask))
		return accuracy


	def loss_function(self, prbs, labels, mask):
		"""
		Calculates the model cross-entropy loss after one forward pass
		Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.

		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x ss_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: the loss of the model as a tensor
		"""
		# Note: you can reuse this from rnn_model.
		loss = tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(labels, prbs)*mask) #how to mask????
		return loss

	@av.call_func
	def __call__(self, *args, **kwargs):
		return super(Transformer_Seq2Seq, self).__call__(*args, **kwargs)
