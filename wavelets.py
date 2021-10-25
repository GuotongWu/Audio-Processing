import librosa
import numpy as np
from skimage.restoration import denoise_wavelet
import matplotlib.pyplot as plt
import soundfile as sf

def read_audio():
    audio_data = {
        'denoise': [],
        'feature': [],
        'label': [], #(sentence, person)
    }
    pre_data = []
    for person in range(3):
        for times in range(10):
            for sentence in range(9):
                y, sr = librosa.load('./src/'+str(person)+'-'+str(times)+'-'+str(sentence)+'.wav')
                # pre_data.append(DenoisingAudio(y, sr))
                pre_data.append(y)
                audio_data['label'].append((sentence, person))
                audio_data['denoise'].append(DenoisingAudio(y, sr))
    for y in pre_data:
        audio_data['feature'].append(librosa.feature.mfcc(y))
    return audio_data, sr


def DenoisingAudio(y, sr):
    # Fs, x = wavfile.read(flute.wav)  # Reading Audio Wave File
    y = y / max(y)  # Normalizing Amplitude

    sigma = 0.05  # Noise Variance
    # print(np.random.randn(y.size))
    # y_noisy = y
    y_noisy = y + sigma * np.random.randn(y.size)  # Adding Noise to Signal

    # Wavelet Denoising
    y_denoise = denoise_wavelet(y_noisy, method='BayesShrink', mode='soft', wavelet_levels=3, wavelet='haar',rescale_sigma=True)

    # plt.figure(figsize=(10, 5), dpi=100)
    plt.plot(y_noisy)
    plt.plot(y_denoise)

    sf.write('./src/denoise/test_noisy.wav', np.array(y_noisy,dtype=np.float32), sr)
    sf.write('./src/denoise/test_denoisy.wav', np.array(y_denoise,dtype=np.float32), sr)

    # plt.legend([signal1, signal2], loc='best', labels=[y_noisy, y_denoise])
    # plt.show()
    return y_noisy, y_denoise

def audio_output(audio_data, sr):
    # print(len(audio_data['denoise']))
    for each_audio in audio_data.items():
        # print('\n' + each_audio[0])
        for person in range(3):
            for times in range(10):
                for sentence in range(9):
                    sf.write('./src/denoise/'+str(person)+'-'+str(times)+'-'+str(sentence)+'.wav', np.array(each_audio[1][person+times+sentence],dtype=np.float32), sr)
                    # sf.write('stereo_file1.wav', each_audio[0], 48000)


if __name__ == '__main__':
    audio_data, sr = read_audio()
    audio_output(audio_data, sr)