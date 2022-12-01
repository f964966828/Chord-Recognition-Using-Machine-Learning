import os
import time
import json
import math
from pitch_class_profiling import PitchClassProfiler
from scipy.io import wavfile
from scipy.io.wavfile import write
from utils import get_params

params = get_params()
train_data_path = params["train_data_path"]
train_json_name = params["train_json_name"]
test_data_path = params["test_data_path"]
test_json_name = params["test_json_name"]
test_file_name = params["test_file_name"]
mode = params["mode"]
unit = params["unit"]

train_json_path = os.path.join(train_data_path, train_json_name)
test_json_path = os.path.join(test_data_path, test_json_name)
test_file_path = os.path.join(test_data_path, test_file_name)
test_wav_path = os.path.join(test_data_path, "wave")

def save_pitch():
    """Extracts pitch class from music dataset and saves them into a json file along witgh genre labels.
        :param dataset_path (str): Path to dataset
        :param json_path (str): Path to json file used to save pitchs
        :return:
        """
    if(mode == "train"):        
        # dictionary to store mapping, labels, and pitch classes
        data = {
            "mapping": [],
            "labels": [],
            "pitch": []
        }

        # loop through all chord sub-folder
        for i, label_name in enumerate(os.listdir(train_data_path)):
            
            if 'json' in label_name or 'wav' in label_name:
                continue
            
            train_label_path = os.path.join(train_data_path, label_name)
            data["mapping"].append(label_name)
            print("\nProcessing: {}".format(label_name))

            for file_name in os.listdir(train_label_path):
                file_path = os.path.join(train_label_path, file_name)

                # process all segments of audio file
                ptc=PitchClassProfiler(file_path)
                data["pitch"].append(ptc.get_profile())
                data["labels"].append(i)
                print("{} \t label:{}".format(file_path, i))

        # save pitch classes to json file
        with open(train_json_path, "w") as fp:
            json.dump(data, fp, indent=4)
    else:
        print("Start preprocess on %s" % test_file_name)

        # splitting input test file
        frecuency, samples = wavfile.read(test_file_path)
        N = len(samples) // unit
        for i in range(N):
            write(os.path.join(test_wav_path, f"{i}.wav"), frecuency,samples[ i * unit : (i+1) * unit ])
        
        data = {
            "mapping": [],
            "labels": [],
            "pitch": [],
            "order":[]
        }

        label_name = "unknown"
        label = -1

        data["mapping"].append(label_name)
        print("\nProcessing: {}".format(label_name))

        for file_name in os.listdir(test_wav_path):

            # load audio file
            file_path = os.path.join(test_wav_path, file_name)
            order = file_name.replace(".wav", "")
            data["order"].append(order)

            # process all segments of audio file
            ptc=PitchClassProfiler(file_path)
            data["pitch"].append(ptc.get_profile())
            data["labels"].append(label)
            print("{} \t label:{}".format(file_path, label))
        
        #Sorting the dictionary
        n = len(data["order"]) 
        for i in range(n-1): 
        # range(n) also work but outer loop will repeat one time more than needed. 
    
            # Last i elements are already in place 
            for j in range(0, n-i-1): 
    
                # traverse the array from 0 to n-i-1 
                # Swap if the element found is greater 
                # than the next element 
                if int(data["order"][j]) > int(data["order"][j+1]) : 
                    data["order"][j],data["order"][j+1]=data["order"][j+1],data["order"][j]
                    data["pitch"][j],data["pitch"][j+1]=data["pitch"][j+1],data["pitch"][j]
                    data["labels"][j],data["labels"][j+1]=data["labels"][j+1],data["labels"][j]

        # save pitch classes to json file
        with open(test_json_path, "w") as fp:
            json.dump(data, fp, indent=4)

if __name__ == "__main__":
    start_time = time.time()
    save_pitch()
    print("%s preprocessing finished, spending %.2f sec" % (mode, time.time() - start_time))
