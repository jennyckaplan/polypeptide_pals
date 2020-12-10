import tensorflow as tf
import numpy as np
import json
import pickle


def json_to_arr(datafile_to_json):
    with open("data/secondary_structure/" + datafile_to_json + ".json") as f:
        data = json.load(f)

    data_arr = []
    for i in range(len(data)):  # for item in data
        features = []  # primary, ss3, ss8, id, protein_length
        example = data[i]
        features.append(example['id'])
        features.append(example['primary'])
        features.append(example['ss3'])
        features.append(example['ss8'])
        features.append(example['protein_length'])

        data_arr.append(features)
    data_arr = np.array(data_arr)

    print(np.mean(data_arr[:, 4]))
    print(type(data_arr))

    # now pickle the data array!
    pickle.dump(data_arr, open(
        "data/pickle/" + datafile_to_json + ".p", "wb"))


if __name__ == '__main__':
    json_to_arr('secondary_structure_train')
    json_to_arr('secondary_structure_valid')
    json_to_arr('secondary_structure_cb513')
    json_to_arr('secondary_structure_ts115')
    json_to_arr('secondary_structure_casp12')
