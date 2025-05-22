import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import torchaudio
import random

from preprocessing import preprocess_audio_melcnn

class CNNModel(nn.Module):
    def __init__(self, in_channels=1, out_classes=2):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Linear(64, out_classes)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.classifier(x)

    def extract_feat(self, x):
        x = self.conv_block(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class MelCNNClassifier:
    def __init__(self, weight_path='weights/best_model.pth', device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNNModel()
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def forward(self, audio_path):
        try:
            mel_spec = preprocess_audio_melcnn(audio_path)
            mel_spec = mel_spec.unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(mel_spec)
                probs = F.softmax(output, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
            # Return True for fake (class 1), False for real (class 0)
            return pred_class == 1
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None

if __name__ == "__main__":
    file_path = "dataset/fake/26_496_000021_000003_gen.wav"
    print("File exists:", os.path.exists(file_path))
    classifier = MelCNNClassifier(weight_path='weights/best_model.pth')
    result = classifier.forward(file_path)
    if result is None:
        print("Classification failed.")
    elif result:
        print("The audio is classified as FAKE.")
    else:
        print("The audio is classified as REAL.")