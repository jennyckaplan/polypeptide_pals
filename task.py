import os
import tensorflow as tf
import numpy as np
from typing import Dict, Tuple, List, Callable

"""
This is the generic Task class used to prepare the datasets
and get train and test data for a task.
"""
class Task:

    """
    The constructor for the Task class takes in the key metric
    and the deserialization function (for converting TFRecord to features)
    for the task.
    """
    def __init__(self,
                 key_metric: str,
                 deserialization_func: Callable[[bytes], Dict[str, tf.Tensor]]):
        self._key_metric = key_metric
        self._deserialization_func = deserialization_func

    """
    This function gets the length of a data element for batching
    according to length.
    """
    def protein_length_function(self, data):
        return data['protein_length']

    """
    This function prepares the dataset by shuffling the data,
    and batching it.
    """
    def prepare_dataset(self,
                        dataset: tf.data.Dataset,
                        buckets: List[int],
                        batch_sizes: List[int],
                        shuffle: bool = False) -> tf.data.Dataset:

        dataset = dataset.map(self._deserialization_func,
                              num_parallel_calls=128)

        buckets_array = np.array(buckets)
        batch_sizes_array = np.array(batch_sizes)

        if np.any(batch_sizes_array == 0) and shuffle:
            iszero = np.where(batch_sizes_array == 0)[0][0]
            filterlen = buckets_array[iszero - 1]
            print("Filtering sequences of length {}".format(filterlen))
            dataset = dataset.filter(
                lambda example: example['protein_length'] < filterlen)
        else:
            batch_sizes_array[batch_sizes_array <= 0] = 1

        dataset = dataset.shuffle(1024) if shuffle else dataset.prefetch(1024)
        batch_fun = tf.data.experimental.bucket_by_sequence_length(
            self.protein_length_function,
            buckets_array,
            batch_sizes_array)
        dataset = dataset.apply(batch_fun)
        return dataset

    """
    This function gets the training and valid data,
    initializes the TFRecordDatasets and prepares the train
    and valid data.
    """
    def get_train_data(self,
                       boundaries: Tuple[List[int], List[int]],
                       train_file: str,
                       valid_file: str,
                       max_sequence_length: int = 100000,
                       add_cls_token: bool = False) -> Tuple[tf.data.Dataset, tf.data.Dataset]:

        if not os.path.exists(train_file):
            raise FileNotFoundError(train_file)

        if not os.path.exists(valid_file):
            raise FileNotFoundError(valid_file)

        train_data = tf.data.TFRecordDataset(train_file)
        valid_data = tf.data.TFRecordDataset(valid_file)

        buckets, batch_sizes = boundaries
        train_data = self.prepare_dataset(
            train_data, buckets, batch_sizes, shuffle=True)
        valid_data = self.prepare_dataset(
            valid_data, buckets, batch_sizes, shuffle=False)

        return train_data, valid_data

    """
    This function gets the test data,
    initializes the TFRecordDataset and prepares the test
    data.
    """
    def get_test_data(self,
                      boundaries: Tuple[List[int], List[int]],
                      test_file: str) -> tf.data.Dataset:

        if not os.path.exists(test_file):
            raise FileNotFoundError(test_file)

        test_data = tf.data.TFRecordDataset(test_file)

        buckets, batch_sizes = boundaries
        test_data = self.prepare_dataset(
            test_data, buckets, batch_sizes, shuffle=False)

        return test_data

    @property
    def key_metric(self) -> str:
        return self._key_metric

    @property
    def deserialization_func(self) -> Callable:
        return self._deserialization_func
