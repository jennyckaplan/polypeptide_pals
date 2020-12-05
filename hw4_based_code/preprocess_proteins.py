import numpy as np
import tensorflow as tf
import numpy as np

from attenvis import AttentionVis
av = AttentionVis()

##########DO NOT CHANGE#####################
PAD_TOKEN = "*PAD*"
STOP_TOKEN = "*STOP*"
START_TOKEN = "*START*"
UNK_TOKEN = "*UNK*"
WINDOW_SIZE = 1632
##########DO NOT CHANGE#####################

def pad_corpus(primary, secondary_structure):
	"""
	DO NOT CHANGE:

	arguments are lists of FRENCH, ENGLISH sentences. Returns [FRENCH-sents, ENGLISH-sents]. The
	text is given an initial "*STOP*".  All sentences are padded with "*STOP*" at
	the end.

	:param french: list of French sentences
	:param english: list of English sentences
	:return: A tuple of: (list of padded sentences for French, list of padded sentences for English)
	"""
	PRIMARY_padded_lines = []
	for line in primary:
		padded_primary_item = line[:WINDOW_SIZE]
		padded_primary_item += [STOP_TOKEN] + [PAD_TOKEN] * (WINDOW_SIZE - len(padded_primary_item)-1)
		PRIMARY_padded_lines.append(padded_primary_item)

	SS_padded_lines = []
	for line in secondary_structure:
		padded_ss_item = line[:WINDOW_SIZE]
		padded_ss_item = [START_TOKEN] + padded_ss_item + [STOP_TOKEN] + [PAD_TOKEN] * (WINDOW_SIZE - len(padded_ss_item)-1)
		SS_padded_lines.append(padded_ss_item)

	return PRIMARY_padded_lines, SS_padded_lines

def build_vocab(sentences):
	"""
	DO NOT CHANGE

  Builds vocab from list of sentences

	:param sentences:  list of sentences, each a list of words
	:return: tuple of (dictionary: word --> unique index, pad_token_idx)
  """
	tokens = []
	for s in sentences: tokens.extend(s)
	all_words = sorted(list(set([STOP_TOKEN,PAD_TOKEN,UNK_TOKEN] + tokens)))

	vocab =  {word:i for i,word in enumerate(all_words)}

	return vocab,vocab[PAD_TOKEN]

def convert_to_id(vocab, sentences):
	"""
	DO NOT CHANGE

  Convert sentences to indexed

	:param vocab:  dictionary, word --> unique index
	:param sentences:  list of lists of words, each representing padded sentence
	:return: numpy array of integers, with each row representing the word indeces in the corresponding sentences
  """
	return np.stack([[vocab[word] if word in vocab else vocab[UNK_TOKEN] for word in sentence] for sentence in sentences])


def read_data(file_name):
	"""
	DO NOT CHANGE

  Load text data from file

	:param file_name:  string, name of data file
	:return: list of sentences, each a list of words split on whitespace
  """
	text = []
	with open(file_name, 'rt', encoding='latin') as data_file:
		for line in data_file: text.append(line.split())
	return text

@av.get_data_func
def get_data(french_training_file, english_training_file, french_test_file, english_test_file):
	"""
	Use the helper functions in this file to read and parse training and test data, then pad the corpus.
	Then vectorize your train and test data based on your vocabulary dictionaries.

	:param french_training_file: Path to the french training file.
	:param english_training_file: Path to the english training file.
	:param french_test_file: Path to the french test file.
	:param english_test_file: Path to the english test file.

	:return: Tuple of train containing:
	(2-d list or array with english training sentences in vectorized/id form [num_sentences x 15] ),
	(2-d list or array with english test sentences in vectorized/id form [num_sentences x 15]),
	(2-d list or array with french training sentences in vectorized/id form [num_sentences x 14]),
	(2-d list or array with french test sentences in vectorized/id form [num_sentences x 14]),
	english vocab (Dict containg word->index mapping),
	french vocab (Dict containg word->index mapping),
	english padding ID (the ID used for *PAD* in the English vocab. This will be used for masking loss)
	"""
	# MAKE SURE YOU RETURN SOMETHING IN THIS PARTICULAR ORDER: train_english, test_english, train_french, test_french, english_vocab, french_vocab, eng_padding_index

	#TODO:


	#1) Read English and French Data for training and testing (see read_data)
	french_training_data = read_data(french_training_file)
	english_training_data = read_data(english_training_file)

	french_test_data = read_data(french_test_file)
	english_test_data = read_data(english_test_file)

	#2) Pad training data (see pad_corpus)
	padded_french_train, padded_english_train = pad_corpus(french_training_data, english_training_data)

	#3) Pad testing data (see pad_corpus)
	padded_french_test, padded_english_test = pad_corpus(french_test_data, english_test_data)

	#4) Build vocab for french (see build_vocab)
	#french_sentences = padded_french_test + padded_french_train
	french_vocab, french_pad_tokenID = build_vocab(padded_french_train)

	#5) Build vocab for english (see build_vocab)
	#english_sentences = padded_english_test + padded_english_train
	english_vocab, english_pad_tokenID = build_vocab(padded_english_train)

	#6) Convert training and testing english sentences to list of IDS (see convert_to_id)

	english_train_vec = np.array(convert_to_id(english_vocab, padded_english_train))
	english_test_vec = np.array(convert_to_id(english_vocab, padded_english_test))

	#7) Convert training and testing french sentences to list of IDS (see convert_to_id)
	french_train_vec = np.array(convert_to_id(french_vocab, padded_french_train))
	french_test_vec = np.array(convert_to_id(french_vocab, padded_french_test))

	return english_train_vec, english_test_vec, french_train_vec, french_test_vec, english_vocab, french_vocab, english_pad_tokenID
