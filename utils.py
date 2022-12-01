import json
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

def get_params():
    
    file = open('params.yaml', 'r', encoding='utf-8')
    params = yaml.load(file, Loader=yaml.FullLoader)

    return params

def load_data(data_path, return_mapping=False):
    """Loads training dataset from json file.
        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    params = get_params()
    random_seed = params["random_seed"]
    split_ratio = params["split_ratio"]

    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["pitch"])
    y = np.array(data["labels"])
    m = np.array(data["mapping"])

    if not return_mapping:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=random_seed)
    
        return X_train, X_test, y_train, y_test
    else:
        return X, y, m

