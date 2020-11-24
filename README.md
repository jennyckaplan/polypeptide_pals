## Developing Unprecedentedly Competent Transformers for TAPE (Tasks Assessing Protein Embeddings)
---

### Overview

This is a reimplementation of the Transformer model from the paper [Evaluating Protein Transfer Learning with TAPE by Rao et al.](https://arxiv.org/pdf/1906.08230.pdf). We focus on a single biological task: Secondary Structure (SS) Prediction. The GitHub for the paper with the model architectures implemented in PyTorch and data is [here](https://github.com/songlab-cal/tape). 

### Dependencies

- wget (for MacOS: run `brew install wget`)

### Data

To download the dataset, run `download_data.sh`. This should create a /data folder in the root directory that contains two folders for the different types of data: TFRecord and raw JSON. This dataset is for the Secondary Structure task only.

### Public Implementations

- https://github.com/songlab-cal/tape
- (Deprecated TensorFlow) https://github.com/songlab-cal/tape-neurips2019
