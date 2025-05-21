import os
import shutil
import random
from pathlib import Path

def create_directory_if_not_exists(directory):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_random_files(directory, num_files):
    """Get a list of random files from a directory."""
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    if len(files) < num_files:
        raise ValueError(f"Not enough files in {directory}. Required: {num_files}, Available: {len(files)}")
    return random.sample(files, num_files)

def move_files_to_destination(source_dir, destination_dir, files):
    """Move files from source to destination directory."""
    for file in files:
        source_path = os.path.join(source_dir, file)
        destination_path = os.path.join(destination_dir, file)
        shutil.move(source_path, destination_path)

def organize_librisvoc_files(base_dir):
    """Organize LibriSeVoc files into real and fake categories."""
    # Define paths
    base_path = Path(base_dir)
    real_dir = base_path / "real"
    fake_dir = base_path / "fake"
    
    # Create destination directories
    create_directory_if_not_exists(real_dir)
    create_directory_if_not_exists(fake_dir)
    
    # Move real files (from gt)
    gt_dir = base_path / "gt"
    if not os.path.exists(gt_dir):
        raise ValueError(f"Ground truth directory not found: {gt_dir}")
    
    real_files = get_random_files(gt_dir, 100)
    move_files_to_destination(gt_dir, real_dir, real_files)
    print(f"Moved {len(real_files)} files from {gt_dir} to {real_dir}")
    
    # Move fake files (from other directories)
    fake_dirs = [
        "wavernn",
        "wavenet",
        "wavegrad",
        "parallel_wave_gan",
        "melgan",
        "diffwave"
    ]
    
    for dir_name in fake_dirs:
        source_dir = base_path / dir_name
        if not os.path.exists(source_dir):
            print(f"Warning: Directory not found: {source_dir}")
            continue
            
        try:
            fake_files = get_random_files(source_dir, 20)
            move_files_to_destination(source_dir, fake_dir, fake_files)
            print(f"Moved {len(fake_files)} files from {source_dir} to {fake_dir}")
        except ValueError as e:
            print(f"Error processing {dir_name}: {str(e)}")

if __name__ == "__main__":
    # Set the LibriSeVoc-500 directory path directly
    librisvoc_dir = "/Users/ronan/Downloads/LibriSeVoc-500"
    
    try:
        organize_librisvoc_files(librisvoc_dir)
        print("File organization completed successfully!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
