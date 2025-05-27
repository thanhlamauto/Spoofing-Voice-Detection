import numpy as np
import xgboost as xgb
import os
import librosa
import warnings
import pickle

from .preprocessing import remove_silence, bandpass_filter, normalize_volume, extract_features_xgb

class XGBoostInfer:
    def __init__(self, model_path='/kaggle/working/xgboost_model.json'):
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        # Define the feature order from training (excluding 'label')
        self.feature_order = [
            'mfcc_mean_0', 'mfcc_std_0', 'mfcc_median_0', 'mfcc_mean_1', 'mfcc_std_1', 'mfcc_median_1',
            'mfcc_mean_2', 'mfcc_std_2', 'mfcc_median_2', 'mfcc_mean_3', 'mfcc_std_3', 'mfcc_median_3',
            'mfcc_mean_4', 'mfcc_std_4', 'mfcc_median_4', 'mfcc_mean_5', 'mfcc_std_5', 'mfcc_median_5',
            'mfcc_mean_6', 'mfcc_std_6', 'mfcc_median_6', 'mfcc_mean_7', 'mfcc_std_7', 'mfcc_median_7',
            'mfcc_mean_8', 'mfcc_std_8', 'mfcc_median_8', 'mfcc_mean_9', 'mfcc_std_9', 'mfcc_median_9',
            'mfcc_mean_10', 'mfcc_std_10', 'mfcc_median_10', 'mfcc_mean_11', 'mfcc_std_11', 'mfcc_median_11',
            'mfcc_mean_12', 'mfcc_std_12', 'mfcc_median_12',
            'chroma_mean_0', 'chroma_std_0', 'chroma_mean_1', 'chroma_std_1', 'chroma_mean_2', 'chroma_std_2',
            'chroma_mean_3', 'chroma_std_3', 'chroma_mean_4', 'chroma_std_4', 'chroma_mean_5', 'chroma_std_5',
            'chroma_mean_6', 'chroma_std_6', 'chroma_mean_7', 'chroma_std_7', 'chroma_mean_8', 'chroma_std_8',
            'chroma_mean_9', 'chroma_std_9', 'chroma_mean_10', 'chroma_std_10', 'chroma_mean_11', 'chroma_std_11',
            'spec_contrast_mean_0', 'spec_contrast_std_0', 'spec_contrast_mean_1', 'spec_contrast_std_1',
            'spec_contrast_mean_2', 'spec_contrast_std_2', 'spec_contrast_mean_3', 'spec_contrast_std_3',
            'spec_contrast_mean_4', 'spec_contrast_std_4', 'spec_contrast_mean_5', 'spec_contrast_std_5',
            'spec_contrast_mean_6', 'spec_contrast_std_6',
            'zcr_mean', 'zcr_std', 'zcr_median',
            'rms_mean', 'rms_std', 'rms_median',
            'mel_spec_db_mean_overall', 'mel_spec_db_std_overall',
            'tonnetz_mean_0', 'tonnetz_std_0', 'tonnetz_mean_1', 'tonnetz_std_1',
            'tonnetz_mean_2', 'tonnetz_std_2', 'tonnetz_mean_3', 'tonnetz_std_3',
            'tonnetz_mean_4', 'tonnetz_std_4', 'tonnetz_mean_5', 'tonnetz_std_5',
            'spectral_centroid_mean', 'spectral_centroid_std',
            'spectral_bandwidth_mean', 'spectral_bandwidth_std',
            'spectral_rolloff_mean', 'spectral_rolloff_std'
        ]

    def preprocess(self, file_path):
        y, sr = librosa.load(file_path, sr=None)
        if len(y) == 0:
            raise ValueError("Empty audio file.")
        y = remove_silence(y, sr, top_db=25)
        if len(y) == 0:
            raise ValueError("Audio empty after silence removal.")
        y = bandpass_filter(y, sr, lowcut=200.0, highcut=4000.0, order=4)
        y = normalize_volume(y, desired_rms=0.05)
        features = extract_features_xgb(y, sr)
        # Convert to numpy array in the training feature order
        feature_vector = np.array([features.get(k, np.nan) for k in self.feature_order], dtype=np.float32)
        # Check for missing or NaN features
        if np.any(np.isnan(feature_vector)):
            missing_features = [k for k, v in zip(self.feature_order, feature_vector) if np.isnan(v)]
            raise ValueError(f"Missing or invalid features in {file_path}: {missing_features}")
        return feature_vector

    def forward(self, file_path):
        """
        file_path: path to audio file
        Returns True if the audio is classified as fake (class 0), False if real (class 1)
        """
        try:
            features = self.preprocess(file_path)
            features = features.reshape(1, -1)
            pred = self.model.predict(features)
            return pred[0] == 0  # True if fake (label 0), False if real (label 1)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

if __name__ == "__main__":
    # Example usage
    test_file = "dataset/real/87_121553_000200_000000.wav"  # Change to your test file
    infer = XGBoostInfer()
    result = infer.forward(test_file)
    if result is None:
        print("Classification failed.")
    elif result:
        print("The audio is classified as FAKE.")
    else:
        print("The audio is classified as REAL.")