import os
import time
import json
import math
from scipy.io import wavfile
from scipy.io.wavfile import write
from utils import get_params, PitchClassProfiler

params = get_params()
train_data_path = params["train_data_path"]
train_json_name = params["train_json_name"]
test_data_path = params["test_data_path"]
test_json_name = params["test_json_name"]
test_file_name = params["test_file_name"]
image_path = params["image_path"]
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
            "len" : [],
            "new": [],
            "frequency":[],
            "pitch": []
        }        
        plot_flag = True
        # loop through all chord sub-folder
        idx = 0
        for label_name in os.listdir(train_data_path):
            
            if 'json' in label_name or 'wav' in label_name:
                continue
            
            train_label_path = os.path.join(train_data_path, label_name)
            data["mapping"].append(label_name)
            print("\nProcessing: {}".format(label_name))

            for file_name in os.listdir(train_label_path):
                file_path = os.path.join(train_label_path, file_name)
                ptc=PitchClassProfiler(file_path)    

                if(plot_flag):
                    ptc.plot_signal(image_path)
                    ptc.plot_fourier(image_path)
                    ptc.plot_profile(image_path)
                    plot_flag = False        

                if(file_name[0].islower()):                    
                    data['new'].append(0)
                else:                    
                    data['new'].append(1)
                    
                data['frequency'].append(ptc.frequency())
                data['len'].append(ptc.get_len())
                data["pitch"].append(ptc.get_profile())
                data["labels"].append(idx)
                print("{} \t label:{}".format(file_path, idx))

            idx += 1

        # save pitch classes to json file
        with open(train_json_path, "w") as fp:
            json.dump(data, fp, indent=4)
    else:
        print("Start preprocess on %s" % test_file_name)

        for file_name in os.listdir(test_wav_path):
            os.remove(os.path.join(test_wav_path, file_name))

        # splitting input test file
        frecuency, samples = wavfile.read(test_file_path)
        N = len(samples) // unit
        for i in range(N):
            write(os.path.join(test_wav_path, "{:03d}.wav".format(i)), frecuency,samples[ i * unit : (i+1) * unit ])
        
        data = {
            "mapping": [],
            "labels": [],
            "len" : [],
            "pitch": []
        }

        label_name = "unknown"
        label = -1

        data["mapping"].append(label_name)
        print("\nProcessing: {}".format(label_name))

        files = os.listdir(test_wav_path)
        files.sort()
        for file_name in files[:110]:

            # load audio file
            file_path = os.path.join(test_wav_path, file_name)

            # process all segments of audio file
            ptc=PitchClassProfiler(file_path)
            data["pitch"].append(ptc.get_profile())
            data["labels"].append(label)
            print("{} \t label:{}".format(file_path, label))
            #print(ptc.get_profile())
        

        # save pitch classes to json file
        with open(test_json_path, "w") as fp:
            json.dump(data, fp, indent=4)

if __name__ == "__main__":
    start_time = time.time()
    save_pitch()
    print("%s preprocessing finished, spending %.2f sec" % (mode, time.time() - start_time))
