import numpy as np
import xgboost as xgb
import os
import librosa
import warnings
import pickle
import joblib

from .preprocessing import remove_silence, bandpass_filter, normalize_volume, extract_features_xgb

class XGBoostInfer:
    def __init__(self, model_path='weights/XGBoost/xgboost_model.json', scaler_path='weights/XGBoost/standard_scaler.joblib'):
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        # with open(scaler_path, 'rb') as f:
        #     self.scaler = pickle.load(f)
        # self.scaler = joblib.load(scaler_path)  # Or pipeline.steps[0][1] if using a pipeline


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
        features = extract_features_xgb(y, sr)
        # Convert to numpy array in the order of keys
        feature_vector = np.array([features[k] for k in features.keys()], dtype=np.float32)
        # feature_vector = self.scaler.transform(feature_vector.reshape(1, -1))[0]
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
            return bool(pred[0] == 0)
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