import os
import torch
import torchaudio
import torchaudio.transforms as T
from scipy.signal import butter, lfilter
from tensorflow.keras import models
import numpy as np
import random


SEGMENT_LENGTH = 1
NUM_SEGMENT = 30
SR = 24000
MODEL_PATH = "MyModule\\Model\\best_model.keras"

model = models.load_model(MODEL_PATH)


# Check if MPS (Metal Performance Shaders) is available and set the device accordingly
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Function to apply a bandpass filter
def bandpass_filter(y, sr, lowcut=250, highcut=4000, order=5):
    """
    Applies a bandpass filter to an audio signal.

    Args:
        y (torch.Tensor): The audio signal as a PyTorch tensor.
        sr (int): The sample rate of the audio signal.
        lowcut (int, optional): The lower cutoff frequency. Defaults to 250.
        highcut (int, optional): The upper cutoff frequency. Defaults to 4000.
        order (int, optional): The order of the filter. Defaults to 5.

    Returns:
        torch.Tensor: The filtered audio signal as a PyTorch tensor.
    """
    # Perform the filtering (this part uses scipy and will run on the CPU)
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y_filtered = lfilter(b, a, y)  # Move to CPU for scipy

    # Move the filtered signal back to the original device
    return torch.tensor(y_filtered, dtype=y.dtype).to(device)

def decrease_low_db(y, sr, threshold_db=-50, target_db=-80):
    """
    Giảm độ lớn của các mẫu âm thanh dưới ngưỡng dB cho trước đến độ to mong muốn,
    giữ nguyên thời gian của tín hiệu âm thanh.

    :param y: Tín hiệu âm thanh (tensor)
    :param sr: Tần số lấy mẫu (Hz)
    :param threshold_db: Ngưỡng dB để xác định các mẫu cần giảm độ lớn (ví dụ: -40 dB)
    :param target_db: Độ to mong muốn cho các mẫu dưới ngưỡng (ví dụ: -80 dB)
    :return: Tín hiệu đã được điều chỉnh (tensor)
    """
    # Calculate the absolute amplitude of the signal
    abs_y = torch.abs(y)

    # Calculate the reference amplitude (maximum amplitude)
    ref_amplitude = torch.max(abs_y) if torch.max(abs_y) > 0 else torch.tensor(1.0, dtype=torch.float32).to(device)

    # Calculate the dB level of each sample relative to the reference amplitude
    y_db = 20 * torch.log10(abs_y / ref_amplitude + 1e-10)  # Add epsilon to avoid log(0)

    # Create a mask for samples below the dB thr2eshold
    mask = y_db < threshold_db

    # Calculate the desired amplitude for samples below the dB threshold
    desired_amplitude = 10 ** (target_db / 20) * ref_amplitude  # Example: -80 dB

    # Create a copy of the signal to adjust
    y_adjusted = y.clone()

    # Reduce the amplitude of samples below the dB threshold
    # Avoid division by zero by adding epsilon
    y_adjusted[mask] = y_adjusted[mask] / (abs_y[mask] + 1e-10) * desired_amplitude

    return y_adjusted  # Convert back to numpy array if needed

def segment_to_spectrogram(segment, sr=24000, n_fft=2048, hop_length=512, n_mels=128):
    """
    Extracts a Mel spectrogram from an audio segment, ensuring execution on the MPS GPU if available.

    Args:
        segment (torch.Tensor): The audio segment as a PyTorch tensor.
        sr (int, optional): The sample rate of the audio segment. Defaults to 24000.
        n_fft (int, optional): The size of the FFT. Defaults to 2048.
        hop_length (int, optional): The hop length for the STFT. Defaults to 512.
        n_mels (int, optional): The number of Mel filterbanks. Defaults to 128.

    Returns:
        torch.Tensor: The Mel spectrogram in decibels (dB).
    """

    # Create the MelSpectrogram transform
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    ).to(device)  # Ensure the transform is also on the correct device

    # Apply the MelSpectrogram transform
    mel_spectrogram = mel_spectrogram(segment)

    # Convert to decibels (dB)
    spectrogram_db = T.AmplitudeToDB().to(device)  # Move AmplitudeToDB to the device
    spectrogram_db = spectrogram_db(mel_spectrogram)

    return spectrogram_db

def extract_segments(audio_file, segment_length=SEGMENT_LENGTH, num_segments=NUM_SEGMENT):
    # Load audio file using torchaudio
    waveform, sr = torchaudio.load(audio_file)
    waveform = bandpass_filter(waveform, sr, lowcut=250, highcut=4000)
    waveform = torch.tensor(waveform, dtype=torch.float32).to(device)
    waveform = decrease_low_db(waveform, sr)

    # Resample if necessary
    if sr != SR:
        resampler = T.Resample(orig_freq=sr, new_freq=SR)
        waveform = resampler(waveform)
        sr = SR

    min_duration = 7  # Minimum duration in seconds
    max_duration = 30  # Maximum duration in seconds
    current_duration = waveform.shape[1] / sr

    if current_duration < min_duration:
        repeat_factor = int(min_duration / current_duration) + 1
        waveform = waveform.repeat(1, repeat_factor)  # Repeat along the time dimension
    elif current_duration > max_duration:
        start_sample = random.randint(0, int((current_duration - max_duration) * sr))
        end_sample = start_sample + int(max_duration * sr)
        waveform = waveform[:, start_sample:end_sample]  # Cut a random 30s segment

    # Calculate the total duration in seconds
    total_duration = waveform.shape[1] / sr

    # Calculate the overlap to ensure exactly num_segments
    overlap = (total_duration - segment_length) / (num_segments - 1)

    # Convert segment length and overlap to samples
    segment_samples = int(segment_length * sr)
    overlap_samples = int(overlap * sr)

    # Extract the segments
    segments = []
    for i in range(num_segments):
        start_sample = i * overlap_samples
        end_sample = start_sample + segment_samples
        segment = waveform[:, start_sample:end_sample]
        spectrogram = segment_to_spectrogram(segment)
        segments.append(spectrogram)

    return segments

def Make_Prediction(file_path):
    train_segments = extract_segments(file_path)
    train_segments = np.array([segment.cpu().numpy() for segment in train_segments])
    print(train_segments.shape)
    # Assuming your data is in a NumPy array
    train_segments = np.expand_dims(train_segments, axis=0)

# Step 2: Transpose to (1, 30, 128, 47, 1)
    train_segments = np.transpose(train_segments, (0, 1, 3, 4, 2))

    predictions = model.predict(train_segments)
    return predictions[0][0]


print (Make_Prediction("H:\Dev\SyntheticVoiceDetection\\Application\\real_demo.wav"))

    