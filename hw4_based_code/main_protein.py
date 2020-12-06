import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess_proteins import *
from transformer_model_protein import Transformer_Seq2Seq
from rnn_model_protein import RNN_Seq2Seq
import sys
import random

from attenvis import AttentionVis
av = AttentionVis()
#french = primary
#english = ss

def train(model, train_primary, train_ss, ss_padding_index):
	"""
	Runs through one epoch - all training examples.

	:param model: the initialized model to use for forward and backward pass
	:param train_primary: primary (amino acid seq) train data (all data for training) of shape (num_sentences, window_size)
	:param train_ss: secondary structure train data (all data for training) of shape (num_sentences, window_size)
	:param ss_padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
	:return: None
	"""
	numExamples = train_primary.shape[0]
	numBatches = (int) (np.ceil(numExamples / model.batch_size))

	primaryBatch = np.asarray(np.array_split(train_primary, numBatches))
	ssBatch = np.asarray(np.array_split(train_ss, numBatches))

	for i in range(numBatches):
		currPrimary = primaryBatch[i]
		currSS = ssBatch[i]
		ssBatch_Inputs = currSS[:, 0:-1]
		ssBatch_Labels = currSS[:, 1:]

		mask = np.where(ssBatch_Labels == ss_padding_index, 0, 1)
		with tf.GradientTape() as tape:
			probs = model(currPrimary, ssBatch_Inputs)

			loss = model.loss_function(probs, ssBatch_Labels, mask)

		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


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
	numExamples = test_primary.shape[0]
	numBatches = (int) (np.ceil(numExamples / model.batch_size))

	primaryBatch = np.asarray(np.array_split(test_primary, numBatches))
	ssBatch = np.asarray(np.array_split(test_ss, numBatches))


	losses = []
	accuracies = []

	sum_accuracy = 0
	sum_loss = 0
	totalTokens = 0
	for i in range(numBatches): #going thru all the batches
		currPrimary = primaryBatch[i]
		currSS = ssBatch[i]
		ssBatch_Inputs = currSS[:, 0:-1]
		ssBatch_Labels = currSS[:, 1:]

		probs = model(currPrimary, ssBatch_Inputs)

		mask = np.where(ssBatch_Labels == ss_padding_index, 0, 1)

		loss = model.loss_function(probs, ssBatch_Labels, mask)
		accuracy = model.accuracy_function(probs, ssBatch_Labels, mask)

		tokens = np.sum(mask)
		totalTokens = totalTokens + tokens
		sum_loss = sum_loss + loss
		sum_accuracy = sum_accuracy + accuracy*tokens

	perplexity = np.exp(sum_loss / totalTokens)
	accuracy = (sum_accuracy + 0.0) / totalTokens

	print ("perplexity: " + str(perplexity))
	print ("accuracy: " + str(accuracy))
	return perplexity, accuracy

	# Note: Follow the same procedure as in train() to construct batches of data!

	#perplexity = np.exp(sum of the losses / # of ALL non-PAD tokens)

def main():
	if len(sys.argv) != 2 or sys.argv[1] not in {"RNN","TRANSFORMER"}:
			print("USAGE: python assignment.py <Model Type>")
			print("<Model Type>: [RNN/TRANSFORMER]")
			exit()

	# Change this to "True" to turn on the attention matrix visualization.
	# You should turn this on once you feel your code is working.
	# Note that it is designed to work with transformers that have single attention heads.
	if sys.argv[1] == "TRANSFORMER":
		av.setup_visualization(enable=False)

	print("Running preprocessing...")
	primary_train, primary_test, ss_train, ss_test, primary_vocab, ss_vocab, ss_pad_tokenID = get_data("../train_secondary_structure.p", "../valid_secondary_structure.p")
	print("Preprocessing complete.")

	model_args = (WINDOW_SIZE, len(primary_vocab), len(ss_vocab))
	print (len(model_args))
	if sys.argv[1] == "RNN":
		model = RNN_Seq2Seq(*model_args)
	elif sys.argv[1] == "TRANSFORMER":
		model = Transformer_Seq2Seq(*model_args)

	# TODO:

	# Train and Test Model for 1 epoch.

	print ("start training")
	train(model, primary_train, ss_train, ss_pad_tokenID)

	print ("start testing")
	test(model, primary_test, ss_test, ss_pad_tokenID)


	# Visualize a sample attention matrix from the test set
	# Only takes effect if you enabled visualizations above
	#av.show_atten_heatmap()

if __name__ == '__main__':
	main()
