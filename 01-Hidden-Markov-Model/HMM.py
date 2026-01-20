import numpy as np
import librosa
import os
from hmmlearn import hmm

def extract_features(audio_file):
    y, sr = librosa.load(audio_file)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfccs.T

data_dir = './audio_samples'

digit_samples = {}

for digit in range(10):
    digit_dir = os.path.join(data_dir, str(digit))
    digit_samples[digit] = [extract_features(os.path.join(digit_dir, filename)) for filename in os.listdir(digit_dir)]

digit_models = {}

for digit, samples in digit_samples.items():
    model = hmm.GaussianHMM(n_components = 3, covariance_type="diag", n_iter= 100)
    X = np.concatenate(samples)
    lengths = [sample.shape[0] for sample in samples]
    model.fit(X, lengths = lengths)
    digit_models[digit] = model

def recognize_digit(audio_file):
    features = extract_features(audio_file)
    scores = {digit: model.score(features) for digit, model in digit_models.items()}
    recognized_digit = max(scores, key=scores.get)
    return recognized_digit

test_audio_file = "text_audio.wav"
recognized_digit = recognize_digit(test_audio_file)
print(f"The recognized digit is: {recognized_digit}")