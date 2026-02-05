import os
import pandas as pd
import soundfile as sf


def find_and_process_missing_files(audio_dir="data/genres_original", csv_path="features_full_combined.csv", output_csv="features_full_combined.csv"):
    """
    Finds missing files in the features CSV based on audio files in the directory,
    extracts their features, and appends them to the CSV.
    
    Args:
        audio_dir (str): Path to the directory containing WAV files
        csv_path (str): Path to the combined features CSV file
        output_csv (str): Path to output the appended features CSV
    """
    # Import here to avoid circular import
    from feature_extraction import extract_features
    
    # Collect all WAV files (full paths)
    all_files = []
    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            if file.lower().endswith(".wav"):
                all_files.append(os.path.join(root, file))

    print(f"Total WAV files on disk: {len(all_files)}")

    # Load merged features CSV
    df = pd.read_csv(csv_path)

    # Find missing files (compare by filename)
    files_in_csv = df['filename'].tolist()
    missing_files = [f for f in all_files if os.path.basename(f) not in files_in_csv]

    print("Missing file(s):", [os.path.basename(f) for f in missing_files])

    # Process missing files
    for file_path in missing_files:
        print(f"Processing missing file: {os.path.basename(file_path)}")
        y, sr = sf.read(file_path)
        print(len(y), sr)
        feat = extract_features(file_path)  # full path
        # Ensure the function returns a dictionary with all keys
        if feat is None or 'filename' not in feat:
            print(f"WARNING: Failed to extract features for {file_path}")
            continue

        df_missing = pd.DataFrame([feat])
        # Append to CSV
        df_missing.to_csv(output_csv, mode='a', index=False, header=False)
        print(f"Appended features for {os.path.basename(file_path)}")



