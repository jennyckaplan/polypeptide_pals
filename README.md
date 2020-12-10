## Developing Unprecedentedly Competent Transformers for TAPE

## (Tasks Assessing Protein Embeddings)

### Overview

This is a reimplementation of the paper [Evaluating Protein Transfer Learning with TAPE by Rao et al](https://arxiv.org/pdf/1906.08230.pdf). The GitHub for the paper with the model architectures implemented in PyTorch and data is [here](https://github.com/songlab-cal/tape). Due to time and resource constraints, we focused on reimplementing the downstream tasks, conditionally passing in the unsupervised pretrained model (trained on the very large Pfam protein family dataset) weights to these downstream models. We reimplemented multiple downstream models each trained on the biological task of Secondary Structure (SS) Prediction. The models we reimplemented were a Transformer, RNN, and LSTM.

### Results

Secondary Structure Prediction (ss3)
| Model       | Accuracy | Perplexity |
| ----------- | :------: | ---------: |
| RNN         |  0.8403  |      1.649 |
| Transformer |  0.8402  |      1.653 |
| LSTM        |  0.6325  |      5.246 |

Secondary Structure Prediction (ss8)
| Model       | Accuracy | Perplexity |
| ----------- | :------: | ---------: |
| RNN         |  0.7474  |      2.241 |
| Transformer |  0.7275  |      2.377 |
| LSTM        |  0.2959  |     12.707 |


### Dependencies

- wget (for MacOS: run `brew install wget`)
- tensorflow 2.3.0
- numpy 1.17.3+

### Data

To download the dataset, run `download_data.sh`. This should create a /data folder in the root directory that contains the data for each task. This dataset is for the Secondary Structure task only.

Then, run `python3 json_to_pickle.py` to create the pickle files in `data/pickle/` with the relevant data for training, validation, and testing.

### Running

`python3 main.py <model>` where model is "RNN", "TRANSFORMER" or "LSTM"

### Public Implementations

- https://github.com/songlab-cal/tape
- (Deprecated TensorFlow) https://github.com/songlab-cal/tape-neurips2019
