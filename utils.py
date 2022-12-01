import json
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(data_path, seed=123):
    """Loads training dataset from json file.
        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["pitch"])
    y = np.array(data["labels"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
    
    return X_train, X_test, y_train, y_test



