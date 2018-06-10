## KMEANS

import numpy, scipy, matplotlib.pyplot as plt, sklearn, librosa, mir_eval, IPython.display
import librosa.display
plt.rcParams['figure.figsize'] = (14, 4)

filename = './results/dataset4/ep10-s1.wav'
x, fs = librosa.load(filename)

onset_frames = librosa.onset.onset_detect(x, sr=fs, delta=0.04, wait=4)
onset_times = librosa.frames_to_time(onset_frames, sr=fs)
onset_samples = librosa.frames_to_samples(onset_frames)

def extract_features(x, fs):
    zcr = librosa.zero_crossings(x).sum()
    energy = scipy.linalg.norm(x)
    return [zcr, energy]



frame_sz = fs*0.090
features = numpy.array([extract_features(x[i:i+frame_sz], fs) for i in onset_samples])
print features.shape
