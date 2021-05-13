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

    def display_spectogram(self, spectogram):
        real_spectogram_db = librosa.amplitude_to_db(np.abs(spectogram), ref=np.max)
        plt.figure()
        librosa.display.specshow(real_spectogram_db)
        plt.colorbar()
        plt.show()
