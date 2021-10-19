import numpy as np
import librosa
import matplotlib.pyplot as plt
import os
from librosa.feature.spectral import mfcc

def read_audio():
    audio_data = {
        'y': [],
        'sr': []
    }
    for person in range(3):
        for times in range(10):
            for sentence in range(9):
                y, sr = librosa.load('./src/'+str(person)+'-'+str(times)+'-'+str(sentence)+'.wav')
                audio_data['y'].append(y)
                audio_data['sr'].append(sr)   
    return audio_data

def padding(audio_data, padding_len=291500):
    padding_data = []
    for y in audio_data['y']:
        zero_len = (padding_len - len(y)) // 2
        padding_data.append(np.hstack((np.zeros(zero_len), y, np.zeros(padding_len-zero_len-len(y)))))
    return padding_data

if __name__ == '__main__':
    audio_data = read_audio()
    padding_data = padding(audio_data)
    print(mfcc(padding_data[0]).shape)