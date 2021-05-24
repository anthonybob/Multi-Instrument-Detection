import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy
import numpy as np
import os
class Spectogram():
    #Window length is the size of the window
    #Window padding is number of zeros to pad audio after window is applied
    #Hop length is the number of audio samples between each STFT output
    def __init__(self, window_length= 512, window_padding=0, hop_length=256, window=scipy.signal.windows.hann):
        self.window_length = window_length
        self.window_padding = window_padding
        self.window = window
        if hop_length != None:
            self.hop_length = hop_length
        else:
            self.hop_length = window_length // 4

    #Code based off examples from https://librosa.org/doc/latest/index.html
    def signal_to_spectogram(self, wav_file):
        signal, sampling_rate = librosa.load(wav_file)
        spectogram_matrix = librosa.stft(signal, n_fft = self.window_length + self.window_padding, hop_length=self.hop_length)
        return spectogram_matrix

    @staticmethod
    def float_to_string(real_num):
        if abs(real_num) < 0.01:
            return "0"
        return "{:.2f}".format(real_num)

    #The spectogram file output is as follows:
    #The first line has two integers. The first number is the number of time samples, and the second number is the number of 
    #frequency bins in the STFT. 
    #Following that each line has four integers that represent the data of the STFT starting with the frequency bins for the first time sample,
    #all the way to the frequency bins to the last time sample. 
    #Each line represents one frequency bin for one time sample.
    #The two integers are the real and complex part of the complex number associated with the frequency bin for the given time in the STFT.
    #The last two integers are respectively the magnitude and phase of the frequency bins.
    def signal_to_spectogram_file(self, wav_file, spectogram_file):
        spectogram_matrix = self.signal_to_spectogram(wav_file)
        np.savetxt(spectogram_file,  np.abs(spectogram_matrix), fmt = '%.2f')
    @staticmethod
    def load_spectogram_from_file(spectogram_file):
        return np.loadtxt(spectogram_file)

    @staticmethod
    def real_spectogram_to_db_spectogram(spectogram):
        return librosa.amplitude_to_db(spectogram, ref=np.max)

    @staticmethod
    def display_complex_spectogram(spectogram):
        real_spectogram_db = librosa.amplitude_to_db(np.abs(spectogram), ref=np.max)
        plt.figure()
        librosa.display.specshow(real_spectogram_db)
        plt.colorbar()
        plt.show()

    @staticmethod
    def display_real_spectogram(spectogram):
        real_spectogram_db = Spectogram.real_spectogram_to_db_spectogram(spectogram)
        plt.figure()
        librosa.display.specshow(real_spectogram_db)
        plt.colorbar()
        plt.show()

def file_test():
    spectogram = Spectogram()
    spec=spectogram.signal_to_spectogram('test_data/0/Medley-solos-DB_test-0_7f9c729c-396b-5cd7-f2ea-b137d8ee7222.wav')
    print(spec.shape[0], spec.shape[1])
    Spectogram.display_complex_spectogram(spec)
    spectogram.signal_to_spectogram_file('test_data/0/Medley-solos-DB_test-0_7f9c729c-396b-5cd7-f2ea-b137d8ee7222.wav', 'spec')
    spec=Spectogram.load_spectogram_from_file('spec')
    Spectogram.display_real_spectogram(spec)

def convert_signals_to_spectograms():
    spectogram = Spectogram()
    for data_dir in ['test_data/', 'training_data/', 'validation_data/']:
        print('Converting' + ' ' + data_dir)
        for label_dir in os.listdir(data_dir):
            os.mkdir(data_dir + 'spec' + label_dir)
            path = data_dir + label_dir + '/'
            for wav_file in os.listdir(path):
                print('Converting ' + wav_file)
                spec = wav_file + '.spec'
                spec_path = data_dir + 'spec' + label_dir + '/'
                spectogram.signal_to_spectogram_file(path + wav_file, spec_path + spec)
    print('Done Converting!')
        
if __name__ == '__main__':
    convert_signals_to_spectograms()