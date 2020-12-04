from typing import Optional, Tuple, List

import numpy as np

import tensorflow as tf
from transformer_input_embedding import TransformerInputEmbedding
from transformer_encoder import TransformerEncoder


class Transformer:

    def __init__(self,
                 n_symbols: int,
                 n_layers: int = 12,
                 n_heads: int = 8,
                 d_model: int = 512,
                 d_filter: int = 2048,
                 dropout: Optional[float] = 0.1,
                 layer_dropout: Optional[float] = None,
                 kernel_regularizer: Optional[str] = None) -> None:

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

        print(self)

    def convert_to_attention_mask(self, sequence, sequence_lengths):
        """Given a padded input tensor of sequences and a tensor of lengths, returns
        a boolean mask for each position in the sequence indicating whether or not
        that position is padding.
        Args:
            sequence (tf.Tensor): Tensor of shape [batch_size, sequence_length, ndim]
            sequence_lengths (tf.Tensor[int]): Tensor of shape [batch_size]
        Returns:
            tf.Tensor[bool]: Tensor of shape [batch_size, sequence_length]
        """
        indices = tf.tile(tf.range(tf.shape(sequence)[1])[
                          None, :], (tf.shape(sequence_lengths)[0], 1))
        mask = indices < sequence_lengths[:, None]
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
