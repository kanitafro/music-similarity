import os
import librosa
import librosa.display
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import soundfile as sf
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from concurrent.futures import ProcessPoolExecutor, as_completed
from missing_files import find_and_process_missing_files
#from feature_groups import get_feature_groups, get_grouped_dataframes

# ================================================================
# CONFIG
# ================================================================
AUDIO_DIR = "data/genres_original"
USE_MULTIPROCESSING = True
MAX_WORKERS = 4
SR = None
MONO = True
N_CHUNKS = 10           # number of parts to divide files into
START_CHUNK = 0         # which chunk to start from (0-indexed)
OUTPUT_DIR = "features_chunks"  # folder to save CSVs

# Directories for spectrograms
SPECTROGRAM_NPY_DIR = "data/spectrograms_original"  # .npy format for analysis
SPECTROGRAM_PNG_DIR = "data/images_spectrograms"     # .png format for visualization

# create output folders if not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SPECTROGRAM_NPY_DIR, exist_ok=True)
os.makedirs(SPECTROGRAM_PNG_DIR, exist_ok=True)

# ================================================================
# LAYER 1: A-WEIGHTING HELPERS
# ================================================================
def a_weighting(frequencies):
    """
    Return A-weighting coefficients for an array of frequencies in Hz
    based on IEC 61672:2003 approximation.
    """
    f = np.array(frequencies)
    ra_num = (12194**2) * (f**4)
    ra_den = ((f**2 + 20.6**2) *
              (f**2 + 12194**2) *
              np.sqrt((f**2 + 107.7**2) * (f**2 + 737.9**2)))
    a = ra_num / ra_den
    # Handle zero frequencies to avoid log10(0) warning
    a = np.where(a > 0, a, 1e-10)
    a_db = 20 * np.log10(a) + 2.0  # +2 dB to match IEC 61672 A-weighting
    a_linear = 10 ** (a_db / 20)    # convert dB to linear
    return a_linear

def apply_a_weighting_to_mel(mel_spectrogram, sr, n_mels):
    """
    Apply A-weighting to a pre-computed Mel spectrogram or Mel-band energies.
    mel_spectrogram: shape (n_mels, n_frames) or 1D (n_mels,) energy vector
    sr: sample rate
    n_mels: number of Mel bands
    """
    mel_frequencies = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sr/2)
    weights = a_weighting(mel_frequencies)
    if mel_spectrogram.ndim == 2:
        return mel_spectrogram * weights[:, np.newaxis]
    else:
        return mel_spectrogram * weights

# ================================================================
# SPECTROGRAM SAVING FUNCTIONS
# ================================================================
def save_mel_spectrogram(y, sr, file_path):
    """
    Extract and save mel spectrogram in both .npy and .png formats.
    Saves to genre-specific subdirectories.
    """
    try:
        # Extract filename and genre info
        base_name = os.path.basename(file_path)
        name_without_ext = os.path.splitext(base_name)[0]
        
        # Extract genre from filename (e.g., "rock.00005.wav" -> "rock")
        genre = base_name.split(".")[0]
        
        # Create genre subdirectories if they don't exist
        npy_genre_dir = os.path.join(SPECTROGRAM_NPY_DIR, genre)
        png_genre_dir = os.path.join(SPECTROGRAM_PNG_DIR, genre)
        os.makedirs(npy_genre_dir, exist_ok=True)
        os.makedirs(png_genre_dir, exist_ok=True)
        
        # Compute mel spectrogram (using librosa defaults: n_mels=128)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Save as .npy for analysis (numpy array format)
        npy_path = os.path.join(npy_genre_dir, f"{name_without_ext}.npy")
        np.save(npy_path, mel_spec_db)
        
        # Save as .png for visualization
        png_path = os.path.join(png_genre_dir, f"{name_without_ext}.png")
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel Spectrogram: {base_name}')
        plt.tight_layout()
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error saving spectrogram for {file_path}: {e}")


# ================================================================
# FEATURE EXTRACTION FUNCTION
# ================================================================
def extract_features(file_path):
    try:
        # Load full audio
        try:
            y, sr_native = sf.read(file_path, always_2d=False)
            if y.ndim > 1:
                y = y.mean(axis=1)
            sr = sr_native
            if len(y) == 0:
                raise ValueError("Audio data is empty")
        except Exception:
            y, sr = librosa.load(file_path, sr=None, mono=True, res_type="kaiser_fast")
            if len(y) == 0:
                raise ValueError("Audio data is empty (fallback)")

        feats = {}
        feats["filename"] = os.path.basename(file_path)
        feats["full_path"] = file_path

        # ====================================================
        # LAYER 2 PREPARATION: EXTRACT track_id AND genre
        # ====================================================
        base_name = os.path.basename(file_path)
        name_parts = base_name.split(".")
        feats["genre"] = name_parts[0]
        feats["track_id"] = name_parts[1] if len(name_parts) > 1 else base_name

        feats["length"] = librosa.get_duration(y=y, sr=sr)

        # ====================================================
        # SAVE MEL SPECTROGRAMS (NPY + PNG)
        # ====================================================
        save_mel_spectrogram(y, sr, file_path)

        stft = np.abs(librosa.stft(y))

        # CHROMA
        chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
        feats["chroma_stft_mean"] = chroma.mean()
        feats["chroma_stft_var"] = chroma.var()

        # RMS
        rms = librosa.feature.rms(S=stft)[0]
        feats["rms_mean"] = rms.mean()
        feats["rms_var"] = rms.var()

        # Spectral
        spectral_centroid = librosa.feature.spectral_centroid(S=stft, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(S=stft, sr=sr)[0]
        rolloff = librosa.feature.spectral_rolloff(S=stft, sr=sr)[0]

        feats["spectral_centroid_mean"] = spectral_centroid.mean()
        feats["spectral_centroid_var"] = spectral_centroid.var()
        feats["spectral_bandwidth_mean"] = spectral_bandwidth.mean()
        feats["spectral_bandwidth_var"] = spectral_bandwidth.var()
        feats["rolloff_mean"] = rolloff.mean()
        feats["rolloff_var"] = rolloff.var()

        # Zero crossing
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        feats["zero_crossing_rate_mean"] = zcr.mean()
        feats["zero_crossing_rate_var"] = zcr.var()

        # Harmonic / Percussive
        harmonic = librosa.effects.harmonic(y)
        percussive = librosa.effects.percussive(y)
        feats["harmony_mean"] = harmonic.mean()
        feats["harmony_var"] = harmonic.var()
        feats["percussive_mean"] = percussive.mean()
        feats["percussive_var"] = percussive.var()

        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        feats["tempo"] = tempo

        # MFCCs (original)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=22)
        for i in range(22):
            feats[f"mfcc{i+1}_mean"] = mfcc[i].mean()
            feats[f"mfcc{i+1}_var"] = mfcc[i].var()

        # ===== LAYER 1: A-WEIGHTED MEL ENERGY FEATURES =====
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=22, power=1.0)  # energy
        mel_mean = mel.mean(axis=1)
        mel_mean_weighted = apply_a_weighting_to_mel(mel_mean, sr, n_mels=22)
        for i in range(22):
            feats[f"a_weighted_mel{i+1}"] = mel_mean_weighted[i]

        return feats

    except Exception as e:
        return {"filename": os.path.basename(file_path), "error": str(e)}

# ================================================================
# COLLECT ALL FILES
# ================================================================
def collect_files(audio_dir):
    file_paths = []
    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            if file.lower().endswith(".wav"):
                file_paths.append(os.path.join(root, file))
    return file_paths

# ================================================================
# PROCESS ONE CHUNK
# ================================================================
def process_chunk(files_chunk, chunk_index):
    all_features = []

    if USE_MULTIPROCESSING:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(extract_features, f): f for f in files_chunk}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Chunk {chunk_index}"):
                all_features.append(future.result())
    else:
        for f in tqdm(files_chunk, desc=f"Chunk {chunk_index}"):
            all_features.append(extract_features(f))

    df_chunk = pd.DataFrame(all_features)
    if "error" in df_chunk.columns:
        df_chunk = df_chunk[df_chunk["error"].isna()].drop(columns="error")

    # Ensure track_id and genre are first columns (helpful for Layer 2)
    cols_order = ["track_id", "genre"] + [c for c in df_chunk.columns if c not in ["track_id", "genre", "error"]]
    df_chunk = df_chunk[cols_order]

    chunk_file = os.path.join(OUTPUT_DIR, f"features_chunk{chunk_index}.csv")
    df_chunk.to_csv(chunk_file, index=False)
    print(f"Chunk {chunk_index} done. Saved {chunk_file}")

# ================================================================
# MAIN PIPELINE
# ================================================================
def main():
    multiprocessing.freeze_support()
    files = collect_files(AUDIO_DIR)
    print(f"Found {len(files)} WAV files.")

    chunk_size = len(files) // N_CHUNKS
    for i in range(START_CHUNK, N_CHUNKS):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < N_CHUNKS - 1 else len(files)
        files_chunk = files[start:end]
        process_chunk(files_chunk, i)

    # Merge chunk CSVs into a single combined CSV if not present
    combined_csv = "features_full_combined.csv"
    if not os.path.exists(combined_csv):
        chunk_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "features_chunk*.csv")))
        if len(chunk_files) > 0:
            df_all = pd.concat([pd.read_csv(f) for f in chunk_files], ignore_index=True)
            df_all.to_csv(combined_csv, index=False)
            print(f"Saved merged CSV as {combined_csv}")
        else:
            print("No chunk files found to merge.")

    # Only process missing files if the combined CSV exists
    if os.path.exists(combined_csv):
        find_and_process_missing_files(audio_dir=AUDIO_DIR, csv_path=combined_csv, output_csv=combined_csv)
    else:
        print("Skipping missing file processing: merged CSV not found.")
    

if __name__ == "__main__":
    main()
