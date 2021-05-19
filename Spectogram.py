import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy
import numpy as np
class Spectogram():
    #Window length is the size of the window
    #Window padding is number of zeros to pad audio after window is applied
    #Hop length is the number of audio samples between each STFT output
    def __init__(self, window_length=2048, window_padding=0, hop_length=None, window=scipy.signal.windows.hann):
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
        with open(spectogram_file, 'w') as f:
            f.write(str(spectogram_matrix.shape[1]) + ' ' + str(spectogram_matrix.shape[0]) + '\n')
            for time_val in range(spectogram_matrix.shape[1]):
                for frequency_bin in range(spectogram_matrix.shape[0]):
                    complex_val = spectogram_matrix[frequency_bin][time_val]
                    f.write(str(complex_val.real) + ' ' + str(complex_val.imag) + ' ' + 
                    str(np.abs(complex_val)) + ' ' + str(np.angle(complex_val)) + '\n')



    def display_spectogram(self, spectogram):
        real_spectogram_db = librosa.amplitude_to_db(np.abs(spectogram), ref=np.max)
        plt.figure()
        librosa.display.specshow(real_spectogram_db)
        plt.colorbar()
        plt.show()
