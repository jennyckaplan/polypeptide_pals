import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import contextlib
import argparse
import pickle as pkl

from typing import Optional
from pathlib import Path

from vocab import PFAM_VOCAB
from transformer_model import Transformer
from task_builder import TaskBuilder
from secondary_structure import SecondaryStructureTask


def run_embed(datafile: str,
              model_name: str,
              load_from: str):

    datapath = Path(datafile)
    if not datapath.exists():
        raise FileNotFoundError(datapath)
    elif datapath.suffix not in ['.tfrecord']:
        raise Exception(
            f"Unknown file type: {datapath.suffix}, must be .tfrecord")

    load_path: Optional[Path] = None
    if load_from is not None:
        load_path = Path(load_from)
        if not load_path.exists():
            raise FileNotFoundError(load_path)

    sess = tf.compat.v1.InteractiveSession()
    K.set_learning_phase(0)
    n_symbols = len(PFAM_VOCAB)
    embedding_model = Transformer(n_symbols)

    task = SecondaryStructureTask()
    deserialization_func = task.deserialization_func

    data = tf.data.TFRecordDataset(str(datapath)).map(deserialization_func)
    data = data.batch(1)
    iterator = data.make_one_shot_iterator()
    batch = iterator.get_next()
    output = embedding_model(batch)
    if load_path is not None:
        embedding_model.load_weights(str(load_path))

    embeddings = []
    with contextlib.suppress(tf.errors.OutOfRangeError):
        while True:
            output_batch = sess.run(output['encoder_output'])
            for encoder_output in output_batch:
                embeddings.append(encoder_output)

    return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile', type=str, help='sequences to embed')
    parser.add_argument('model', type=str, help='which model to use')
    parser.add_argument('--load-from', type=str, default=None,
                        help='file from which to load pretrained weights')
    parser.add_argument(
        '--task', default=None,
        help='If running a forward pass through existing task datasets, refer to the task with this flag')
    parser.add_argument('--output', default='outputs.pkl',
                        type=str, help='file to output results to')
    args = parser.parse_args()

    embeddings = run_embed(args.datafile, args.model,
                           args.load_from)

    with open(args.output, 'wb') as f:
        pkl.dump(embeddings, f)


if __name__ == '__main__':
    main()
