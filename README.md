# Spoofing Voice Detection

This project, developed for a machine learning course, focuses on the critical task of spoofing voice detection. The primary objective is to accurately distinguish between genuine human speech and various forms of synthetically generated or replayed voice audio. This is a crucial area in cybersecurity and biometric authentication, aiming to enhance the robustness of voice-based systems against malicious attacks.

Our group has explored and implemented five distinct approaches to tackle this challenge, each contributed by a different student. This multi-faceted exploration allows for a comprehensive comparison of different methodologies and their effectiveness in identifying spoofed audio.

## Project Structure

The repository is organized to facilitate easy navigation and understanding of the project components:

* **`train/`**: This directory contains the scripts and configurations used for training each of the five detection models. Each method's training code is located here.

* **`weights/`**: This folder stores the trained model weights and checkpoints for each of the five implemented methods. These pre-trained models can be used for direct inference without retraining.

* **`infer/`**: This directory houses the inference scripts for each detection method. These scripts demonstrate how to load the trained models and use them to classify new audio samples as genuine or spoofed.

## Dataset

The dataset used for training and evaluating our spoofing voice detection models is sourced from Kaggle:

[**LibriSEVOC Dataset**](https://www.kaggle.com/datasets/trinhhaphuong/librisevoc)

This dataset provides a collection of genuine and spoofed voice samples, essential for developing and testing robust detection algorithms. Please download and extract the dataset into a designated `data/` directory (or link it appropriately) within your project environment before running the training or inference scripts.

## Methods Implemented

(Details on the five specific methods will be added here by the project team. For example, you might list:

1. **Method 1: [MFCC + SVM]**

2. **Method 2: [CNN-based approach]**

3. **Method 3: [Spectrogram analysis with RNN]**

4. **Method 4: [ Deep learning with raw audio]**

5. **Method 5: [Feature engineering with GMM]**
   )

## Getting Started

To set up and run the project locally, follow these steps:

1. **Clone the repository:**


## Inference

🔧 Command Format

python infer.py --model <model_name> --num_files <N> --batch_size <B>

✅ Example

python infer.py --model xgboost --num_files 32 --batch_size 8

🧠 Model Options

1. gmm – LFCC + GMM model

2. cnn – Mel-spectrogram CNN

3. xgboost – XGBoost-based classifier

4. wav2vec – Pretrained Wav2Vec2 model

📌 Notes

Test dataset size is 220
