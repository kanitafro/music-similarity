import os
import glob
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

# ================================================================
# MIN-MAX SCALING OF NUMERICAL FEATURES 
# ================================================================
def scale_features(df, metadata_df):
    """Min-max scale numeric columns, keeping track_id/genre order intact."""
    # explicitly include A-weighted columns if present
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # keep a_weighted_mel columns even if they are object type in CSV
    a_weighted_cols = [c for c in df.columns if c.startswith("a_weighted_mel")]
    numeric_cols = list(set(numeric_cols) | set(a_weighted_cols))

    if "length" in numeric_cols:
        numeric_cols.remove("length")

    min_vals = df[numeric_cols].min()
    max_vals = df[numeric_cols].max()

    X_scaled = (df[numeric_cols] - min_vals) / (max_vals - min_vals)
    X_scaled["track_id"] = metadata_df["track_id"]
    X_scaled["genre"] = metadata_df["genre"]

    cols = ["track_id", "genre"] + [c for c in X_scaled.columns if c not in ["track_id", "genre"]]
    return X_scaled[cols]


# ================================================================
# MEL SPECTROGRAM PROCESSING FOR QUERY-BY-HUMMING
# ================================================================
def scale_qbh_features(df_qbh):
    """
    Z-score scaling for QbH features only.
    Metadata columns are preserved.
    """
    meta_cols = ["track_id", "genre"]
    feature_cols = df_qbh.columns.difference(meta_cols)

    mean = df_qbh[feature_cols].mean()
    std = df_qbh[feature_cols].std() + 1e-6

    X_scaled = (df_qbh[feature_cols] - mean) / std
    X_scaled[meta_cols] = df_qbh[meta_cols]

    cols = meta_cols + list(feature_cols)
    return X_scaled[cols]

def extract_qbh_embedding(
    audio_path,
    sr=22050,
    n_mels=128,
    hop_length=512
):
    y, sr = librosa.load(audio_path, sr=sr, mono=True)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length,
        power=2.0
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    mean = mel_db.mean(axis=1)
    std = mel_db.std(axis=1)

    emb = np.concatenate([mean, std])
    emb = emb / (np.linalg.norm(emb) + 1e-8)

    return emb

def extract_qbh_query(
    wav_path,
    sr=22050,
    n_mels=128,
    hop_length=512
) -> np.ndarray:
    """
    Extract a Query-by-Humming embedding for a single audio file,
    returning a flat vector with [mel_mean_1..n, mel_std_1..n],
    compatible with the CSV built by `build_qbh_dataset`.
    """
    emb = extract_qbh_embedding(wav_path, sr=sr, n_mels=n_mels, hop_length=hop_length)
    return emb  # shape: (2*n_mels,)

def load_mel_spectrogram(path: str) -> np.ndarray:
    """Load a mel spectrogram saved as .npy (shape: [n_mels, n_frames])."""
    mel = np.load(path)
    if mel.ndim != 2:
        raise ValueError(f"Expected 2D mel spectrogram, got shape {mel.shape} for {path}")
    return mel.astype(np.float32)


def pad_or_truncate_frames(mel: np.ndarray, target_frames: int = 512) -> np.ndarray:
    """Center-crop or right-pad mel frames to a fixed length for comparability."""
    n_mels, n_frames = mel.shape
    if n_frames > target_frames:
        start = (n_frames - target_frames) // 2
        return mel[:, start:start + target_frames]
    if n_frames < target_frames:
        pad = target_frames - n_frames
        pad_val = mel.min() if np.isfinite(mel).all() else 0.0
        return np.pad(mel, ((0, 0), (0, pad)), mode="constant", constant_values=pad_val)
    return mel


def normalize_mel(mel: np.ndarray) -> np.ndarray:
    """Per-band z-score normalization to reduce loudness bias."""
    mean = mel.mean(axis=1, keepdims=True)
    std = mel.std(axis=1, keepdims=True) + 1e-6
    return (mel - mean) / std


def pooled_mel_stats(mel: np.ndarray) -> np.ndarray:
    """Statistical pooling (mean + std across time) for a fixed-length embedding."""
    mean = mel.mean(axis=1)
    std = mel.std(axis=1)
    return np.concatenate([mean, std], axis=0)


def spectrogram_to_row(mel: np.ndarray, genre: str, track_id: str):
    """Convert a processed mel spectrogram into a flat feature row with metadata."""
    vec = pooled_mel_stats(mel)
    n_mels = mel.shape[0]
    row = {"track_id": track_id, "genre": genre}
    for i in range(n_mels):
        row[f"mel_mean_{i+1}"] = vec[i]
    for i in range(n_mels):
        row[f"mel_std_{i+1}"] = vec[n_mels + i]
    return row

# ================================================================
# OBSOLETE: Use build_qbh_dataset from main.py instead
# ================================================================
def process_mel_spectrograms(base_dir: str = "data/spectrograms_original", target_frames: int = 512) -> pd.DataFrame:
    """
    Load, normalize, and pool mel spectrograms (.npy) into fixed-length vectors
    suitable for query-by-humming similarity.

    Steps per file:
      - load .npy mel (expects shape [n_mels, n_frames])
      - center-crop or pad to `target_frames`
      - per-band z-score normalization
      - statistical pooling (mean + std across time)

    Returns
    -------
    pd.DataFrame with columns: track_id, genre, mel_mean_1..n, mel_std_1..n
    """

    rows = []
    pattern = os.path.join(base_dir, "*", "*.npy")
    mel_paths = glob.glob(pattern)
    for path in tqdm(mel_paths, desc="Processing mel .npy"):
        mel = load_mel_spectrogram(path)
        mel = pad_or_truncate_frames(mel, target_frames=target_frames)
        mel = normalize_mel(mel)

        genre = os.path.basename(os.path.dirname(path))
        base = os.path.basename(path)
        parts = base.split(".")
        track_id = parts[1] if len(parts) > 1 else base

        rows.append(spectrogram_to_row(mel, genre=genre, track_id=track_id))

    return pd.DataFrame(rows)


def build_qbh_dataset(
    audio_dir="data/genres_original",
    output_csv="qbh_mel_features.csv",
    n_mels=128,
    sr=22050,
    hop_length=512
):
    """
    Build a Query-by-Humming dataset by converting audio files into fixed-length
    embeddings (mean+std of mel spectrograms) and save as CSV.
    Columns will match `process_mel_spectrograms` output: mel_mean_1..n, mel_std_1..n
    """
    rows = []
    wavs = glob.glob(os.path.join(audio_dir, "*", "*.wav"))

    for path in tqdm(wavs, desc="Building QbH dataset"):
        # Compute embedding
        emb = extract_qbh_embedding(path, sr=sr, n_mels=n_mels, hop_length=hop_length)

        # Metadata
        base = os.path.basename(path)
        genre = base.split(".")[0]
        track_id = base.replace(".wav", "").replace(".", "")

        # Build row
        row = {"track_id": track_id, "genre": genre}
        # Split embedding into mean and std like spectrogram_to_row
        for i in range(n_mels):
            row[f"mel_mean_{i+1}"] = emb[i]
        for i in range(n_mels):
            row[f"mel_std_{i+1}"] = emb[n_mels + i]

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved QbH dataset features to {output_csv}")
    return df