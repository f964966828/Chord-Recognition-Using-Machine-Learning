import json
import yaml
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.fftpack import fft
from math import log2
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


class PitchClassProfiler():
    def __init__(self, file_name):
        self.file_name = file_name
        self.read = False

    def _read_file(self):
        self._frequency, samples = wavfile.read(self.file_name)   
        #print(samples.shape)
        try:
            self._samples = samples[:, 0]
        except:
            self._samples = samples
        #print(self._samples.shape)
        self.read = True

    def frequency(self):
        if not self.read:
            self._read_file()        
        return self._frequency

    def samples(self):
        if not self.read:
            self._read_file()
        return self._samples

    def fourier(self):
        return fft(self.samples())

    def plot_signal(self, path):
        plt.plot(self.samples())
        plt.savefig(os.path.join(path,'signal.png'))
        #plt.show()

    def plot_fourier(self,path):
        plt.plot(self.fourier())
        plt.savefig(os.path.join(path,'fourier.png'))
        #plt.show()
    def get_len(self):
        if not self.read:
            self._read_file()        
        return len(self._samples)
    def pcp(self, X):
        #The algorithm here is implemented using
        #the names of the math formula as shown in the paper
        fs = self.frequency()

        #fref = [16.35, 17.32, 18.35, 19.45, 20.60, 21.83, 23.12, 24.50, 25.96, 27.50, 29.14, 30.87]
        #fref = [130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 185.00, 196.00, 207.65, 220.00, 233.08, 246.94]
        fref = 130.81  #???

        N = len(X)
        #assert(N % 2 == 0)

        def M(l, p):
            if l == 0:
                return -1
            return round(12 * log2( (fs * l)/(N * fref )  ) ) % 12

        pcp = [0 for p in range(12)]
        
        #print("Computing pcp...")
        for p in range(12):
            for l in range(N//2):
                if p == M(l, p):
                    pcp[p] += abs(X[l])**2
        
        #Normalize pcp
        pcp_norm = [0 for p in range(12)]
        for p in range(12):
            pcp_norm[p] = (pcp[p] / sum(pcp))
        return list(pcp_norm)

    def get_profile(self):
        X = self.fourier()        
        return self.pcp(X)
        
    def plot_profile(self,path):
        objects = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')
        y_pos = np.arange(len(objects))
        performance = self.get_profile()
        
        plt.figure()
        plt.bar(y_pos, performance, align='center', alpha=0.5)
        plt.xticks(y_pos, objects)
        plt.ylabel('Energy')
        plt.title('PCP results')        
        plt.savefig(os.path.join(path,'profile.png'))
        #plt.show()

class LongFileProfiler(PitchClassProfiler):
    def __init__(self, file_name):
        super().__init__(file_name)
        self.current_pointer = 0
        self.window = self.frequency() // 2
        print(self.window)

    def get_profile(self):
        profiles_list = []
        samples_count = len( self.samples() )

        while self.current_pointer < samples_count:
            rigth_bound =  self.current_pointer + self.window
            
            if rigth_bound >= samples_count:
                rigth_bound = samples_count - 1

            window_samples = self.samples()[self.current_pointer: rigth_bound]
            X = fft(window_samples)
            profiles_list.append( self.pcp(X) )

            self.current_pointer += self.window
        return profiles_list
