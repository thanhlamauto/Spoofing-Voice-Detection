import numpy as np
import librosa
import torch
import joblib
from scipy.stats import skew, kurtosis
import torchaudio
import os
import random
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

def extract_features_xgb(y, sr, n_mfcc=13, n_mels=128, hop_length=512, frame_length=2048):
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


# --- LFCC feature extraction and preprocessing utilities ---
frame_length = 0.025
frame_shift = 0.010
sample_rate = 24000
n_lfcc = 20

def pre_emphasis(signal, coeff=0.97):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def peak_normalize(signal):
    return signal / np.max(np.abs(signal))

def add_noise(signal, noise_level=0.005):
    noise = np.random.normal(0, noise_level, signal.shape)
    return signal + noise

def chunk_signal(signal, sr, frame_length=frame_length, frame_shift=frame_shift):
    frame_len = int(frame_length * sr)
    hop_len = int(frame_shift * sr)
    frames = librosa.util.frame(signal, frame_length=frame_len, hop_length=hop_len).T
    return frames

lfcc_transform = torchaudio.transforms.LFCC(
    sample_rate=sample_rate,
    n_lfcc=n_lfcc,
    speckwargs={"n_fft": 1048, "hop_length": int(sample_rate * frame_length) + 1, "win_length": int(sample_rate * frame_length)},
    log_lf=True
)

def extract_lfcc(frame, sr=24000):
    waveform = torch.tensor(frame).float().unsqueeze(0)
    lfcc = lfcc_transform(waveform)
    return lfcc.squeeze(0).T.numpy()

def preprocess_audio_melcnn(audio_path, sample_rate=16000, n_mels=128, segment_duration=4):
    waveform, _ = torchaudio.load(audio_path, normalize=True)
    waveform = waveform.mean(dim=0).unsqueeze(0)  # Make mono if stereo

    num_frames = waveform.size(1)
    segment_length = int(segment_duration * sample_rate)
    if num_frames > segment_length:
        start_frame = random.randint(0, num_frames - segment_length)
        waveform = waveform[:, start_frame:start_frame + segment_length]

    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=800, hop_length=160,
        win_length=400, n_mels=n_mels, power=2.0)(waveform)

    log_mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-6)
    return log_mel_spec

