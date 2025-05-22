import numpy as np
import librosa
import torch
import joblib
from scipy.stats import skew, kurtosis
import torchaudio
import os

from preprocessing import pre_emphasis, peak_normalize, add_noise, chunk_signal, extract_lfcc

def extract_gmm_scores(file_path, gmms, sr=24000):
    y, sr = librosa.load(file_path, sr=sr)
    y = peak_normalize(pre_emphasis(y))
    y = add_noise(y, noise_level=0.005)
    frames = chunk_signal(y, sr)

    gmm_scores = {gmm_name: [] for gmm_name in gmms.keys()}

    for frame in frames:
        if np.any(frame):
            for gmm_name, gmm in gmms.items():
                lfcc = extract_lfcc(frame, sr)
                logL = gmm.score_samples(lfcc)
                gmm_scores[gmm_name].append(logL)
    feature = []
    for gmm_name in gmms.keys():
        scores = gmm_scores[gmm_name]
        feature.extend([
            np.mean(scores),
            np.std(scores),
            skew(scores)[0],
            kurtosis(scores)[0],
            np.min(scores),
            np.max(scores),
            np.percentile(scores, 25),
            np.percentile(scores, 75)
        ])
    return np.array(feature)

# --- Inference class ---
class LFCCGMMInfer:
    def __init__(self):
        """
        gmm_paths: dict with keys as GMM names and values as paths to GMM model files
        xgb_model_path: path to trained XGBoost model (joblib .pkl)
        """
        self.gmm_paths = {
            "groundtruth": "weights/LFCC_GMM_XGBOOST/gmm_model_ground-truth.pkl",
            "autoregressive": "weights/LFCC_GMM_XGBOOST/gmm_model_autoregressive.pkl",
            "gan-based": "weights/LFCC_GMM_XGBOOST/gmm_model_gan-based.pkl",
            "diffusion-based": "weights/LFCC_GMM_XGBOOST/gmm_model_diffusion-based.pkl",
            }
        self.xgb_model_path = "weights/LFCC_GMM_XGBOOST/xgboost_best_model.pkl"
        self.GMMs = {name: joblib.load(path) for name, path in self.gmm_paths.items()}
        self.model = joblib.load(self.xgb_model_path)


    def preprocess(self, file_path):
        features = extract_gmm_scores(file_path, self.GMMs)
        return features

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
    infer = LFCCGMMInfer()
    test_file = "dataset/fake/32_4137_000006_000001_gen.wav"  # Replace with your test file path
    result = infer.forward(test_file)
    if result is None:
        print("Classification failed.")
    elif result:
        print("The audio is classified as FAKE.")
    else:
        print("The audio is classified as REAL.")