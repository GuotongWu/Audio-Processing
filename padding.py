from librosa.core import audio
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pickle as pkl
from librosa.feature.spectral import mfcc

def padding(audio_data, padding_len=291500):
    padding_data = []
    for y in audio_data:
        zero_len = (padding_len - len(y)) // 2
        padding_data.append(np.hstack((np.zeros(zero_len), y, np.zeros(padding_len-zero_len-len(y)))))
    return padding_data

def read_audio():
    audio_data = {
        'feature': [],
        'label': [], #(sentence, person)
    }
    pre_data = []
    for person in range(3):
        for times in range(10):
            for sentence in range(9):
                y, _ = librosa.load('./src/'+str(person)+'-'+str(times)+'-'+str(sentence)+'.wav')
                pre_data.append(y)
                audio_data['label'].append((sentence, person))   
    pre_data = padding(pre_data)
    for y in pre_data:
        audio_data['feature'].append(librosa.feature.mfcc(y))
    return audio_data

def save_obj(obj, name):
    with open(name+'.pkl', 'wb') as f:
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)
 

if __name__ == '__main__':
    audio_data = read_audio()
    save_obj(audio_data, './src/audio')