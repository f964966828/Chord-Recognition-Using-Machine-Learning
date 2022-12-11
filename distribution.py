import os
import json
import numpy as np
import matplotlib.pyplot as plt
from utils import get_params, load_data

params = get_params()
data_path = params["train_data_path"]
json_name = params["train_json_name"]
json_path = os.path.join(data_path, json_name)
image_path = params["image_path"]

if __name__ == "__main__":

    with open(json_path, "r") as fp:
        data = json.load(fp)

    y = np.array(data["labels"])
    map = np.array(data["mapping"])
    freq = np.array(data["frequency"])
    length = np.array(data["len"])
    new = np.array(data["new"])
    labels = np.array([map[i] for i in y])


    bins = np.arange(len(map)) - 0.5
    # distribution of chords
    plt.figure()
    plt.hist(x=labels, bins=bins, alpha=0.8, rwidth=0.85, label='new')
    plt.hist(x=labels[new==0], bins=bins, alpha=0.8, rwidth=0.85, label='origin')
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Chord', fontsize=18)
    plt.ylabel('Count', fontsize=18)
    plt.title('Chord Distribution', fontsize=20)
    plt.legend()
    plt.savefig(os.path.join(image_path, 'chord_distribution.png'))

    new_seconds = np.clip(length[new==1] / freq[new==1], 0, 10)
    old_seconds = np.clip(length[new==0] / freq[new==0], 0, 10)
    # distribution of .wav length
    plt.figure()
    plt.hist(x=new_seconds, bins=30, alpha=0.8, label='new')
    plt.hist(x=old_seconds, bins=30, alpha=0.8, label='origin')
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Seconds', fontsize=18)
    plt.ylabel('Count', fontsize=18)
    plt.title('Length Distribution', fontsize=20)
    plt.legend()
    plt.savefig(os.path.join(image_path, 'length_distribution.png'))
