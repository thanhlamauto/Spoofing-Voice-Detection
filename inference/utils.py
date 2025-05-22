import os
from torch.utils.data import Dataset, DataLoader, Subset
from .models import LFCCGMMClassifier, MelCNNClassifier, XGBoostClassifier, Wav2Vec2Classifier, EnsembleSVMClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import random
from tqdm import tqdm
import numpy as np

class VoiceTestDataset(Dataset):
    """
    PyTorch Dataset for loading test audio files and their labels.
    Label: 1 for fake, 0 for real.
    """
    def __init__(self, datalist, transform=None):
        self.datalist = datalist
        self.transform = transform

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        file_path, label = self.datalist[idx]
        sample = {'file_path': file_path, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample

def get_test_dataset(fake_dir="dataset/fake", real_dir="dataset/real"):
    """
    Collects all .wav files from fake and real directories and returns a VoiceTestDataset.
    """
    datalist = []
    # Fake files (label 1)
    for fname in os.listdir(fake_dir):
        if fname.endswith('.wav'):
            datalist.append((os.path.join(fake_dir, fname), 1))
    # Real files (label 0)
    for fname in os.listdir(real_dir):
        if fname.endswith('.wav'):
            datalist.append((os.path.join(real_dir, fname), 0))
    return VoiceTestDataset(datalist)

def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    # Find the threshold where FPR ~= FNR
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer, eer_threshold

def test_model(model_name, num_files, batch_size=1):
    """
    Tests the specified model on a subset of the dataset and prints a classification report.
    :param model_name: str, one of ['lfcc', 'mel', 'xgboost', 'wav2vec']
    :param num_files: int, number of files to test
    :param dataset: VoiceTestDataset instance
    """
    model_map = {
        'gmm': LFCCGMMClassifier,
        'cnn': MelCNNClassifier,
        'xgboost': XGBoostClassifier,
        'wav2vec': Wav2Vec2Classifier,
        'svm': EnsembleSVMClassifier
    }

    if model_name not in model_map:
        raise ValueError(f"Invalid model name '{model_name}'. Choose from {list(model_map.keys())}")

    model = model_map[model_name]()  # instantiate the model
    model = model.forward

    print("Model loaded. Ready to predict...")

    fake_dir = "dataset/fake"
    real_dir = "dataset/real"
    dataset = get_test_dataset(fake_dir, real_dir)


# Randomly sample num_files from dataset and create a subset
    indices = random.sample(range(len(dataset)), min(num_files, len(dataset)))
    subset = Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    y_true, y_pred, y_scores = [], [], []

    for batch in tqdm(dataloader, desc="Processing batches"):
        file_paths = batch['file_path']
        labels = batch['label']

        for file_path, label in tqdm(zip(file_paths, labels), total=len(labels), desc="Processing files", leave=False):
            score = model(file_path)  # Score can be probability or confidence
            # If model returns boolean, convert to float score
            if isinstance(score, bool):
                score = float(score)

            pred_label = 1 if score >= 0.5 else 0

            y_true.append(label)
            y_pred.append(pred_label)
            y_scores.append(score)

    print(f"\nClassification Report for model: {model_name}")
    print(classification_report(y_true, y_pred, target_names=["Real", "Fake"]))

    try:
        auc = roc_auc_score(y_true, y_scores)
        eer, threshold = compute_eer(y_true, y_scores)
        print(f"AUC: {auc:.4f}")
        print(f"EER: {eer:.4f} (Threshold: {threshold:.4f})")
    except ValueError as e:
        print(f"Could not compute AUC or EER: {e}")

if __name__ == "__main__":

    # Test the model with a sample call
    test_model(model_name="gmm", num_files=32, batch_size=32)
