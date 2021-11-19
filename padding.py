import numpy as np
import librosa
import matplotlib.pyplot as plt
import pickle as pkl
from librosa.feature.spectral import mfcc
from mpl_toolkits.mplot3d import Axes3D
import os

def padding(audio_data, padding_len=32500):
    padding_data = []
    for y in audio_data:
        zero_len = (padding_len - len(y)) // 2
        padding_data.append(np.hstack((np.zeros(zero_len), y, np.zeros(padding_len-zero_len-len(y)))))
    return padding_data

def read_audio(path:str, need_padding=False):
    audio_data = {
        'feature': [],
        'label': [], #(sentence, person)
    }
    pre_data = []
    for item in os.listdir(path):
        person, times, sentence = [int(i) for i in item.split('.')[0].split('-')]
        y, _ = librosa.load(os.path.join(path, item))
        # print(librosa.feature.mfcc(y).shape)
        pre_data.append(y)
        audio_data['label'].append((sentence, person))
    # exit()
    if need_padding:   
        pre_data = padding(pre_data)
    for y in pre_data:
        audio_data['feature'].append(librosa.feature.mfcc(y))
    return audio_data

def save_obj(obj, name):
    with open(name+'.pkl', 'wb') as f:
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)

def read_obj(address):
    with open(address, 'rb') as f:
        audio_data = pkl.load(f)
    return audio_data

def draw_feature(audio_data):
    fig = plt.figure() 
    ax = Axes3D(fig)

    num = audio_data['feature'][0].shape[1]
    
    x, y = np.meshgrid(np.arange(0, num), np.arange(0, 20))

    ax.scatter3D(x, y, audio_data['feature'][0])
    ax.scatter3D(x, y, audio_data['feature'][1])
    plt.show()



if True:
    audio_data = read_audio('./temp/test', need_padding=False)
    save_obj(audio_data, './pkl/all_test_nopadding')
audio_data = read_obj('./pkl/all_test_nopadding.pkl')
print(audio_data['feature'][0].shape)
print(audio_data['feature'][1].shape)
# draw_feature(audio_data)