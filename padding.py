from librosa.core import audio
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pickle as pkl
from librosa.feature.spectral import mfcc
from mpl_toolkits.mplot3d import Axes3D

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


def draw_feature(audio_data):
    fig = plt.figure() 
    ax = Axes3D(fig)
    
    xy = [[],[]]
    for i in range(50):
        xy[0] += [i]*20
    xy[1] = list(range(20))*50

    # print(audio_data['feature'][0][:, :100].reshape((-1,)))
    ax.scatter3D(xy[0], xy[1], audio_data['feature'][0][:, 270:320].reshape((-1,)))
    plt.show()

 
if __name__ == '__main__':
    audio_data = read_audio()
    save_obj(audio_data, './src/audio')
    draw_feature(audio_data)