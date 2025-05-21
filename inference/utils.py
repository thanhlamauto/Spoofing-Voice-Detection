import os
from torch.utils.data import Dataset, DataLoader, Subset
from .models import LFCCGMMClassifier, MelCNNClassifier, XGBoostClassifier, Wav2Vec2Classifier
from sklearn.metrics import classification_report
import random

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
        'wav2vec': Wav2Vec2Classifier
    }

    if model_name not in model_map:
        raise ValueError(f"Invalid model name '{model_name}'. Choose from {list(model_map.keys())}")

    model = model_map[model_name]()  # instantiate the model
    if model_name in ['gmm', 'xgboost']:
        model = model.forward

    fake_dir = "dataset/fake"
    real_dir = "dataset/real"
    dataset = get_test_dataset(fake_dir, real_dir)


# Randomly sample num_files from dataset and create a subset
    indices = random.sample(range(len(dataset)), min(num_files, len(dataset)))
    subset = Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    y_true, y_pred = [], []

    for batch in dataloader:
        file_paths = batch['file_path']
        labels = batch['label']

        for file_path, label in zip(file_paths, labels):
            pred = model(file_path)  # Assuming model returns True for fake, False for real
            pred_label = 1 if pred else 0
            y_true.append(label)
            y_pred.append(pred_label)

    print(f"\nClassification Report for model: {model_name}")
    print(classification_report(y_true, y_pred, target_names=["Real", "Fake"]))

if __name__ == "__main__":

    # Test the model with a sample call
    test_model(model_name="gmm", num_files=32, batch_size=32)
