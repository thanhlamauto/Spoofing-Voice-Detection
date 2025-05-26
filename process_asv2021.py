import os
import shutil
from pathlib import Path

def create_directory_if_not_exists(directory):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_asv2021_data(source_flac_dir, trial_metadata_file, output_dir):
    """Process ASVspoof2021 data: move FLAC files and create labels."""
    source_path = Path(source_flac_dir)
    output_path = Path(output_dir)
    test_dir = output_path / "test" / "asv-2021"
    
    # Create test directory
    create_directory_if_not_exists(test_dir)
    
    # Check if source directory exists
    if not os.path.exists(source_path):
        raise ValueError(f"Source FLAC directory not found: {source_path}")
    
    if not os.path.exists(trial_metadata_file):
        raise ValueError(f"Trial metadata file not found: {trial_metadata_file}")
    
    # Dictionary to store filename -> label mapping
    file_labels = {}
    
    # Read trial metadata file
    print("Reading trial metadata...")
    with open(trial_metadata_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 100000 == 0:  # Progress indicator for large file
                print(f"Processed {line_num} lines...")
            
            parts = line.strip().split()
            if len(parts) < 2:
                continue
                
            filename = parts[1]  # Second word is the filename
            
            # Check if line contains spoof or bonafide
            line_lower = line.lower()
            if 'spoof' in line_lower:
                file_labels[filename] = 0  # fake
            elif 'bonafide' in line_lower:
                file_labels[filename] = 1  # real
    
    print(f"Found {len(file_labels)} file labels from metadata")
    
    # Get all FLAC files from source directory
    flac_files = [f for f in os.listdir(source_path) if f.endswith('.flac')]
    print(f"Found {len(flac_files)} FLAC files in source directory")
    
    # Copy FLAC files and prepare labels
    copied_files = 0
    labels_output = []
    
    for flac_file in flac_files:
        # Extract base filename without extension
        base_filename = flac_file.replace('.flac', '')
        
        if base_filename in file_labels:
            # Copy the file
            source_file_path = source_path / flac_file
            dest_file_path = test_dir / flac_file
            shutil.copy2(source_file_path, dest_file_path)
            
            # Add to labels
            label = file_labels[base_filename]
            labels_output.append(f"{flac_file} {label}")
            
            copied_files += 1
        else:
            print(f"Warning: No metadata found for {flac_file}")
    
    print(f"Copied {copied_files} FLAC files to {test_dir}")
    
    # Create labels file in the main project directory
    labels_file = output_path / "asv2021_labels.txt"
    with open(labels_file, 'w') as f:
        for label_line in labels_output:
            f.write(label_line + '\n')
    
    print(f"Created labels file: {labels_file}")
    print(f"Total processed files: {len(labels_output)}")
    
    # Count real vs fake
    real_count = sum(1 for line in labels_output if line.endswith(' 1'))
    fake_count = sum(1 for line in labels_output if line.endswith(' 0'))
    print(f"Real files: {real_count}, Fake files: {fake_count}")

if __name__ == "__main__":
    # Set the paths
    source_flac_dir = "/Users/ronan/Downloads/ASVspoof2021_DF_eval 2/flac"
    trial_metadata_file = "/Users/ronan/Downloads/ASVspoof2021_DF_eval 2/trial_metadata.txt"
    output_dir = "/Users/ronan/Developer/Spoofing-Voice-Detection"
    
    try:
        process_asv2021_data(source_flac_dir, trial_metadata_file, output_dir)
        print("ASVspoof2021 data processing completed successfully!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
