from typing import Optional, Tuple, List

import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from transformer_input_embedding import TransformerInputEmbedding
from transformer_encoder import TransformerEncoder


class Transformer(Model):

    def __init__(self,
                 n_symbols: int,
                 n_layers: int = 12,
                 n_heads: int = 8,
                 d_model: int = 512,
                 d_filter: int = 2048,
                 dropout: Optional[float] = 0.1,
                 layer_dropout: Optional[float] = None,
                 kernel_regularizer: Optional[str] = None) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_filter = d_filter
        self.kernel_regularizer = kernel_regularizer
        self.dropout = dropout

        input_embedding = TransformerInputEmbedding(
            d_model, discrete=True, n_symbols=n_symbols, dropout=dropout,
            concat_position_encoding=True, reproject_position_encoding=True)

        self.encoder = TransformerEncoder(
            input_embedding, n_layers, n_heads, d_model, d_filter, dropout, layer_dropout)

    def convert_sequence_mask_to_attention_mask(self, sequence, sequence_mask):
        """Given a padded input tensor of sequences and a boolean mask for each position
        in the sequence, returns a 3D boolean mask for use in attention.
         Args:
        sequence (tf.Tensor): Tensor of shape [batch_size, sequence_length_1, ndim]
          padding_mask (tf.Tensor[bool]): Tensor of shape [batch_size, sequence_length_2]
        Returns:
            tf.Tensor[bool]: Tensor of shape [batch_size, sequence_length_1, sequence_length_2]
        """
        batch_assert = tf.assert_equal(tf.shape(sequence_mask)[0], tf.shape(sequence)[0],
                                       message='batch size mismatch between input sequence and  \
                                            sequence_mask')
        rank_assert = tf.assert_equal(tf.rank(sequence_mask), 2,
                                      message='Can only convert 2D position mask to 3D attention mask')

        with tf.control_dependencies([batch_assert, rank_assert]):
            attention_mask = tf.tile(
                sequence_mask[:, None, :], (1, tf.shape(sequence)[1], 1))

            return attention_mask

    def convert_sequence_length_to_sequence_mask(self, sequence, sequence_lengths):
        """Given a padded input tensor of sequences and a tensor of lengths, returns
        a boolean mask for each position in the sequence indicating whether or not
        that position is padding.
        Args:
            sequence (tf.Tensor): Tensor of shape [batch_size, sequence_length, ndim]
            sequence_lengths (tf.Tensor[int]): Tensor of shape [batch_size]
        Returns:
            tf.Tensor[bool]: Tensor of shape [batch_size, sequence_length]
        """
        batch_assert = tf.assert_equal(tf.shape(sequence_lengths)[0], tf.shape(sequence)[0],
                                       message='batch size mismatch between input sequence and  \
                                            sequence_lengths')
        rank_assert = tf.assert_equal(tf.rank(sequence_lengths), 1,
                                      message='Can only convert 1D sequence_lengths to 2D mask')

        dtype = sequence_lengths.dtype
        with tf.control_dependencies([batch_assert, rank_assert]):
            array_shape = tf.shape(sequence, out_type=dtype)
            batch_size = array_shape[0]
            seqlen = array_shape[1]

            indices = tf.tile(tf.range(seqlen, dtype=dtype)
                              [None, :], (batch_size, 1))
            mask = indices < sequence_lengths[:, None]

            return mask

    def convert_to_attention_mask(self, sequence, mask):
        """Given a padded input tensor of sequences and a tensor of lengths, returns
        a boolean mask for each position in the sequence indicating whether or not
        that position is padding.
        Args:
            sequence (tf.Tensor): Tensor of shape [batch_size, sequence_length, ndim]
            sequence_lengths (tf.Tensor[int]): Tensor of shape [batch_size]
        Returns:
            tf.Tensor[bool]: Tensor of shape [batch_size, sequence_length]
        """
        if mask is None:
            return None
        if len(mask.shape) == 1:
            mask = self.convert_sequence_length_to_sequence_mask(
                sequence, mask)
        if len(mask.shape) == 2:
            mask = self.convert_sequence_mask_to_attention_mask(
                sequence, mask)
        if mask.dtype != tf.bool:
            mask = tf.cast(mask, tf.bool)
        return mask

    def call(self, inputs):
        """
        Args:
            sequence: tf.Tensor[int32] - Amino acid sequence,
                a padded tensor with shape [batch_size, MAX_PROTEIN_LENGTH]
            protein_length: tf.Tensor[int32] - Length of each protein in the sequence, a tensor with shape [batch_size]
        Output:
            encoder_output: tf.Tensor[float32] - embedding of each amino acid
                a tensor with shape [batch_size, MAX_PROTEIN_LENGTH, d_model]
        """

        sequence = inputs['primary']
        protein_length = inputs['protein_length']

        attention_mask = self.convert_to_attention_mask(
            sequence, protein_length)

        encoder_output = self.encoder(sequence, mask=attention_mask)
        inputs['encoder_output'] = encoder_output
        return inputs

    def get_optimal_batch_sizes(self) -> Tuple[List[int], List[int]]:
        bucket_sizes = np.array(
            [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
             1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 2000])
        batch_sizes = np.array(
            [4, 3, 2, 1.5, 1, 0.9, 0.9, 0.8, 0.65, 0.6,
             0.5, 0.5, 0.4, 0.3, 0.3, 0.2, 0.2, 0.1, 0, 0])

        batch_sizes = np.asarray(batch_sizes, np.int32)
        batch_sizes[batch_sizes <= 0] = 1

        return bucket_sizes, batch_sizes
