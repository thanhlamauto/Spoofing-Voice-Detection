import numpy as np
import xgboost as xgb
import os
import librosa
import warnings

# print(xgb.__version__)


warnings.filterwarnings('ignore')

# --- Preprocessing and feature extraction functions from xgboost_trial2.ipynb ---

def remove_silence(y, sr, top_db=20):
    yt, index = librosa.effects.trim(y, top_db=top_db)
    if len(yt) == 0:
        return y
    return yt

def bandpass_filter(data, sr, lowcut=300.0, highcut=3400.0, order=5):
    from scipy.signal import butter, lfilter
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    if low >= 1.0: low = 0.99
    if high >= 1.0: high = 0.99
    if low >= high:
        low = high / 2
        if low == 0: low = 0.01
    try:
        b, a = butter(order, [low, high], btype='band')
        y = lfilter(b, a, data)
        return y
    except Exception:
        return data

def normalize_volume(y, desired_rms=0.05):
    current_rms = np.sqrt(np.mean(y**2))
    if current_rms == 0:
        return y
    y_normalized = y * (desired_rms / current_rms)
    return y_normalized

def extract_features(y, sr, n_mfcc=13, n_mels=128, hop_length=512, frame_length=2048):
    features = {}
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=frame_length, hop_length=hop_length)
    for i in range(n_mfcc):
        features[f'mfcc_mean_{i}'] = np.mean(mfccs[i,:])
        features[f'mfcc_std_{i}'] = np.std(mfccs[i,:])
        features[f'mfcc_median_{i}'] = np.median(mfccs[i,:])
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)
    for i in range(chroma.shape[0]):
        features[f'chroma_mean_{i}'] = np.mean(chroma[i,:])
        features[f'chroma_std_{i}'] = np.std(chroma[i,:])
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)
    for i in range(spec_contrast.shape[0]):
        features[f'spec_contrast_mean_{i}'] = np.mean(spec_contrast[i,:])
        features[f'spec_contrast_std_{i}'] = np.std(spec_contrast[i,:])
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)
    features['zcr_median'] = np.median(zcr)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)
    features['rms_mean'] = np.mean(rms)
    features['rms_std'] = np.std(rms)
    features['rms_median'] = np.median(rms)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    features['mel_spec_db_mean_overall'] = np.mean(mel_spec_db)
    features['mel_spec_db_std_overall'] = np.std(mel_spec_db)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    for i in range(tonnetz.shape[0]):
        features[f'tonnetz_mean_{i}'] = np.mean(tonnetz[i,:])
        features[f'tonnetz_std_{i}'] = np.std(tonnetz[i,:])
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)
    features['spectral_centroid_mean'] = np.mean(spectral_centroid)
    features['spectral_centroid_std'] = np.std(spectral_centroid)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)
    features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
    features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)
    features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
    features['spectral_rolloff_std'] = np.std(spectral_rolloff)
    return features

class XGBoostInfer:
    def __init__(self, model_path='weights/xgboost_model.json'):
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        # Optionally: you may want to store the feature order if needed

    def preprocess(self, file_path):
        y, sr = librosa.load(file_path, sr=None)
        if len(y) == 0:
            raise ValueError("Empty audio file.")
        y = remove_silence(y, sr, top_db=25)
        if len(y) == 0:
            raise ValueError("Audio empty after silence removal.")
        y = bandpass_filter(y, sr, lowcut=200.0, highcut=4000.0, order=4)
        y = normalize_volume(y, desired_rms=0.05)
        features = extract_features(y, sr)
        # Convert to numpy array in the order of keys
        feature_vector = np.array([features[k] for k in sorted(features.keys())], dtype=np.float32)
        return feature_vector

    def forward(self, file_path):
        """
        file_path: path to audio file
        Returns True for fake (class 1), False for real (class 0)
        """
        try:
            features = self.preprocess(file_path)
            features = features.reshape(1, -1)
            pred = self.model.predict(features)
            return bool(pred[0] == 1)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

if __name__ == "__main__":
    # Example usage
    test_file = "dataset/fake/26_496_000021_000003_gen.wav"  # Change to your test file
    infer = XGBoostInfer(model_path='weights/xgboost_model.json')
    result = infer.forward(test_file)
    if result is None:
        print("Classification failed.")
    elif result:
        print("The audio is classified as FAKE.")
    else:
        print("The audio is classified as REAL.")