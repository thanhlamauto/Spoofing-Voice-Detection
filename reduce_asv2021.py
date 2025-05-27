import os
import random
import shutil
from pathlib import Path

def reduce_asv2021_dataset(test_dir, trial_metadata_file, target_real=200, target_fake=200):
    """Reduce ASV2021 dataset to specified number of real and fake files."""
    
    # Read trial metadata to get file labels
    file_labels = {}
    real_files = []
    fake_files = []
    
    print("Reading trial metadata...")
    with open(trial_metadata_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                filename = parts[1] + ".flac"  # Add .flac extension
                
                # Check if line contains spoof or bonafide
                if "spoof" in line.lower():
                    file_labels[filename] = 0  # fake
                    fake_files.append(filename)
                elif "bonafide" in line.lower():
                    file_labels[filename] = 1  # real
                    real_files.append(filename)
    
    print(f"Found {len(real_files)} real files and {len(fake_files)} fake files in metadata")
    
    # Check which files actually exist in the test directory
    test_path = Path(test_dir)
    existing_files = set(f.name for f in test_path.glob("*.flac"))
    
    # Filter to only files that exist
    existing_real = [f for f in real_files if f in existing_files]
    existing_fake = [f for f in fake_files if f in existing_files]
    
    print(f"Found {len(existing_real)} existing real files and {len(existing_fake)} existing fake files")
    
    # Randomly select target number of files
    if len(existing_real) > target_real:
        selected_real = random.sample(existing_real, target_real)
    else:
        selected_real = existing_real
        print(f"Warning: Only {len(existing_real)} real files available, less than target {target_real}")
    
    if len(existing_fake) > target_fake:
        selected_fake = random.sample(existing_fake, target_fake)
    else:
        selected_fake = existing_fake
        print(f"Warning: Only {len(existing_fake)} fake files available, less than target {target_fake}")
    
    # Files to keep
    files_to_keep = set(selected_real + selected_fake)
    
    # Remove files not in the selection
    removed_count = 0
    for file_path in test_path.glob("*.flac"):
        if file_path.name not in files_to_keep:
            file_path.unlink()
            removed_count += 1
    
    print(f"Removed {removed_count} files")
    print(f"Kept {len(selected_real)} real files and {len(selected_fake)} fake files")
    
    # Create new labels file
    labels_file = Path("asv2021_labels.txt")
    with open(labels_file, 'w') as f:
        for filename in selected_real:
            f.write(f"{filename} 1\n")
        for filename in selected_fake:
            f.write(f"{filename} 0\n")
    
    print(f"Created new labels file: {labels_file}")
    print(f"Total files remaining: {len(files_to_keep)}")

if __name__ == "__main__":
    test_dir = "./test/asv-2021"
    trial_metadata_file = "./trial_metadata.txt"
    
    try:
        reduce_asv2021_dataset(test_dir, trial_metadata_file, target_real=200, target_fake=200)
        print("ASV2021 dataset reduction completed successfully!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
