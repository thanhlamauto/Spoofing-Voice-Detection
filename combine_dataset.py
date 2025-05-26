import os
import shutil
from pathlib import Path

def create_directory_if_not_exists(directory):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def combine_files_for_testing(dataset_dir, output_dir):
    """Combine real and fake files into a test directory with labels."""
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    test_dir = output_path / "test" / "librisevoc"
    
    # Create test directory
    create_directory_if_not_exists(test_dir)
    
    # Paths to real and fake directories
    real_dir = dataset_path / "real"
    fake_dir = dataset_path / "fake"
    
    # Check if directories exist
    if not os.path.exists(real_dir):
        raise ValueError(f"Real directory not found: {real_dir}")
    if not os.path.exists(fake_dir):
        raise ValueError(f"Fake directory not found: {fake_dir}")
    
    # List to store file labels
    file_labels = []
    
    # Copy real files and add to labels
    real_files = [f for f in os.listdir(real_dir) if os.path.isfile(os.path.join(real_dir, f))]
    for file in real_files:
        source_path = os.path.join(real_dir, file)
        destination_path = os.path.join(test_dir, file)
        shutil.copy2(source_path, destination_path)
        file_labels.append(f"{file} 1")  # 1 for real
    
    print(f"Copied {len(real_files)} real files to {test_dir}")
    
    # Copy fake files and add to labels
    fake_files = [f for f in os.listdir(fake_dir) if os.path.isfile(os.path.join(fake_dir, f))]
    for file in fake_files:
        source_path = os.path.join(fake_dir, file)
        destination_path = os.path.join(test_dir, file)
        shutil.copy2(source_path, destination_path)
        file_labels.append(f"{file} 0")  # 0 for fake
    
    print(f"Copied {len(fake_files)} fake files to {test_dir}")
    
    # Create labels file
    labels_file = test_dir / "labels.txt"
    with open(labels_file, 'w') as f:
        for label in file_labels:
            f.write(label + '\n')
    
    print(f"Created labels file: {labels_file}")
    print(f"Total files: {len(file_labels)} ({len(real_files)} real, {len(fake_files)} fake)")

if __name__ == "__main__":
    # Set the dataset directory path (where real and fake folders are)
    dataset_dir = "./dataset"
    output_dir = "."
    
    try:
        combine_files_for_testing(dataset_dir, output_dir)
        print("File combination completed successfully!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
