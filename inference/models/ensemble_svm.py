import numpy as np
import librosa
import noisereduce as nr
from scipy.signal import butter, lfilter
import os
import joblib
import warnings

warnings.filterwarnings('ignore')

# --- Preprocessing and feature extraction functions from EnsembleSVM.ipynb ---

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs 
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=300.0, highcut=3400.0, fs=16000, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def reduce_noise(y, sr):
    noise_sample = y[:int(0.5*sr)]
    reduced = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample)
    return reduced

def segment_audio(y, sr=24000, segment_length=3):
    total_length = segment_length * sr
    if len(y) < total_length:
        y = librosa.util.fix_length(y, total_length)
    elif len(y) > total_length:
        y = y[:total_length]
    return y

def preprocess_audio(file_path, sr=24000):
    y, _ = librosa.load(file_path, sr=sr)
    y = bandpass_filter(y, fs=sr)
    y = reduce_noise(y, sr=sr)
    y = segment_audio(y, sr=sr, segment_length=3)

    return y

def extract_features(y, sr=24000):
    # 1. MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    mfcc_median = np.median(mfcc, axis=1)

    # 2. Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)

    # 3. Spectral Contrast
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_contrast_mean = np.mean(spec_contrast, axis=1)
    spec_contrast_std = np.std(spec_contrast, axis=1)

    # 4. Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_bandwidth_std = np.std(spectral_bandwidth)

    # 5. Zero-Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_median = np.median(zcr)

    # 6. Tonnetz
    y_harmonic = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
    tonnetz_mean = np.mean(tonnetz, axis=1)
    tonnetz_std = np.std(tonnetz, axis=1)

    feature_vector = np.concatenate([
        mfcc_mean, mfcc_std, mfcc_median,
        chroma_mean, chroma_std,
        spec_contrast_mean, spec_contrast_std,
        [spectral_bandwidth_std],
        [zcr_median],
        tonnetz_mean, tonnetz_std
    ])

    return feature_vector

class EnsembleSVMInfer:
    def __init__(self, model_dir='weights/saved_model_SVMs'):
        self.svm_models = []
        for i in range(1, 51):
            model_path = os.path.join(model_dir, f'svm_model_{i}.pth')
            model = joblib.load(model_path)  
            self.svm_models.append(model)
        self.subsets = joblib.load(os.path.join(model_dir, 'subsets.pkl'))

    def forward(self, file_path):
        """
        file_path: path to audio file
        Returns True for fake (class 1), False for real (class 0)
        """
        y = preprocess_audio(file_path)
        features = extract_features(y).reshape(1, -1)
        
        proba_list = []
        for i in range(50):
            model = self.svm_models[i]
            subset = self.subsets[i]
            x_sub = features[:, subset]
            prob = model.predict_proba(x_sub)[0]  # [prob_class_0, prob_class_1]
            proba_list.append(prob)
        avg_proba = np.mean(proba_list, axis=0)  # [mean_prob_0, mean_prob_1]
        final_label = np.argmax(avg_proba)
        return final_label == 1

        print(f"Error processing {file_path}: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    test_file = "dataset/fake/26_496_000021_000003_gen.wav"  # Change to your test file
    infer = EnsembleSVMInfer()
    result = infer.forward(test_file)
    if result is None:
        print("Classification failed.")
    elif result:
        print("The audio is classified as FAKE.")
    else:
        print("The audio is classified as REAL.")