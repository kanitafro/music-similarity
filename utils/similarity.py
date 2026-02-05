import numpy as np
import pandas as pd
from numpy.linalg import norm
from process_spectrograms import extract_qbh_query
from utils.feature_groups import get_feature_groups

def normalize_group_distances(group_dists):
    """
    Normalize group distances to unit norm.
    
    Used in combined similarity calculation because:
    -MFCC group dominates numerically
    -Tempo and chroma are underweighted even with equal weights
    Hence layer 2 would be numerically biased without normalization.
    """
    vals = np.array(list(group_dists.values()))
    norm = np.linalg.norm(vals) + 1e-9
    return {k: v / norm for k, v in group_dists.items()}


def layer1_distance(x, y, a_weighted_cols):
    return np.linalg.norm(
        x[a_weighted_cols].values - y[a_weighted_cols].values
    )

def group_distance(x, y, feature_groups):
    distances = {}

    for group, cols in feature_groups.items():
        if group == "a_weighted":
            continue

        valid_cols = [c for c in cols if c in x.index]
        if len(valid_cols) == 0:
            continue

        distances[group] = np.linalg.norm(
            x[valid_cols].values - y[valid_cols].values
        )

    return distances
def combined_similarity(x, y, feature_groups, weights=None, normalize=True):
    """
    Compute weighted combined distance between two tracks using layered features.
    Optionally normalize each component to [0,1].
    
    param x, y: pd.Series rows of features
    param feature_groups: dict of feature groups (from get_feature_groups)
    param weights: dict of weights per layer/group
    param normalize: whether to normalize distances to [0,1] per layer/group

    return: total_distance, layer1_distance, dict of group distances
    """
    if weights is None:
        weights = {
            "layer1": 1.0,
            "timbre": 1.0,
            "spectral": 1.0,
            "rhythm_tempo": 1.0,
            "chroma_harmony": 1.0
        }

    # --- Layer 1 ---
    d_layer1 = layer1_distance(x, y, feature_groups["a_weighted"])
    if normalize:
        d_layer1 = d_layer1 / (np.linalg.norm(d_layer1) + 1e-9) if np.ndim(d_layer1) > 0 else d_layer1

    # --- Layer 2 (feature groups) ---
    group_dists = group_distance(x, y, feature_groups)
    if normalize:
        group_dists = normalize_group_distances(group_dists)

    # --- Weighted sum ---
    total = weights.get("layer1", 1.0) * d_layer1
    for g, d in group_dists.items():
        total += weights.get(g, 1.0) * d

    return total, d_layer1, group_dists


def find_similar(track_id, X_scaled, top_n=5):
    """
    Docstring for find_similar
    
    param track_id: Track identifier from track_id column
    param X_scaled: Scaled feature DataFrame including 'track_id' and 'genre' columns
    param top_n: Number of top similar tracks to return

    return: DataFrame of top_n similar tracks with their distances
    """
    feature_cols = X_scaled.columns.difference(['track_id', 'genre'])
    query_vec = X_scaled[X_scaled['track_id'] == track_id][feature_cols].values[0]
    all_vectors = X_scaled[feature_cols].values
    # L2-normalize vectors
    all_vectors_norm = all_vectors / np.linalg.norm(all_vectors, axis=1, keepdims=True)
    query_vec_norm = query_vec / np.linalg.norm(query_vec)

    # Compute Euclidean distance between normalized vectors
    distances = np.linalg.norm(all_vectors_norm - query_vec_norm, axis=1)  # range [0, 2]
    similarities = 1 - distances / 2  # range [0,1]

    results = X_scaled.copy()
    results['similarity'] = similarities
    # Sort by similarity descending
    return results[results['track_id'] != track_id].sort_values('similarity', ascending=False).head(top_n)

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / ((norm(a) * norm(b)) + 1e-9)

def euclidean_similarity(a, b):
    # normalize vectors
    a_norm = a / norm(a)
    b_norm = b / norm(b)
    distance = norm(a_norm - b_norm)  # range [0, 2]
    similarity = 1 - distance / 2      # range [0, 1]
    return similarity

def hybrid_similarity(a, b, alpha=0.7):
    """Compute hybrid similarity combining cosine and Euclidean similarity."""
    cos_sim = cosine_similarity(a, b)
    euc_sim = euclidean_similarity(a, b)   # normalized version
    return alpha * cos_sim + (1 - alpha) * euc_sim

def get_similar_tracks(track_id, df, feature_cols, alpha=0.7, top_n=10):
    """Get top_n similar tracks based on hybrid similarity.

    param track_id: Track identifier from track_id column
    param df: DataFrame containing track features and metadata
    param feature_cols: List of feature column names to use for similarity
    param alpha: Weighting factor for hybrid similarity
    param top_n: Number of top similar tracks to return

    return: DataFrame of top_n similar tracks with their similarity scores
    """
    query_vec = df.loc[df['track_id'] == track_id, feature_cols].values[0]
    similarities = []
    for _, row in df.iterrows():
        if row['track_id'] == track_id:
            continue
        sim = hybrid_similarity(query_vec, row[feature_cols].values, alpha)
        similarities.append({"track_id": row["track_id"], "genre": row["genre"], "similarity": sim})
    return pd.DataFrame(similarities).sort_values("similarity", ascending=False).head(top_n)

def find_similar_layered(track_id, X_scaled, top_n=5, weights=None):
    """
    Find top_n similar tracks using layered features, returning normalized similarity scores.

    param track_id: Track identifier from track_id column
    param X_scaled: Scaled feature DataFrame including 'track_id' and 'genre' columns
    param top_n: Number of top similar tracks to return
    param weights: Optional dict of weights per layer/group

    return: DataFrame of top_n similar tracks with similarity scores
    """
    feature_groups = get_feature_groups(X_scaled)
    query = X_scaled[X_scaled["track_id"] == track_id].iloc[0]

    rows = []
    for _, row in X_scaled.iterrows():
        if row["track_id"] == track_id:
            continue

        # Compute weighted distances
        total, d_l1, d_l2 = combined_similarity(query, row, feature_groups, weights)

        rows.append({
            "track_id": row["track_id"],
            "genre": row["genre"],
            "total_distance": total,
            "layer1_distance": d_l1,
            **{f"{k}_distance": v for k, v in d_l2.items()}
        })

    df = pd.DataFrame(rows)

    # --- Normalize distances to similarity [0,1] ---
    df["similarity"] = 1 - df["total_distance"] / (df["total_distance"].max() + 1e-9)
    df["layer1_similarity"] = 1 - df["layer1_distance"] / (df["layer1_distance"].max() + 1e-9)

    for k in d_l2.keys():
        col_dist = f"{k}_distance"
        col_sim = f"{k}_similarity"
        df[col_sim] = 1 - df[col_dist] / (df[col_dist].max() + 1e-9)

    # --- Sort by overall similarity ---
    return df.sort_values("similarity", ascending=False).head(top_n)


def find_similar_qbh(
    query_vec,
    qbh_dataset,
    top_n=10,
    metric="cosine"
):
    """
    Query-by-humming similarity search.
    """
    print("Running QbH search...")
    meta_cols = ["track_id", "genre"]
    feature_cols = qbh_dataset.columns.difference(meta_cols)

    sims = []
    for _, row in qbh_dataset.iterrows():
        x = row[feature_cols].values

        if metric == "cosine":
            sim = cosine_similarity(query_vec, x)
        else:
            sim = -np.linalg.norm(query_vec - x)

        sims.append({
            "track_id": row["track_id"],
            "genre": row["genre"],
            "similarity": sim
        })

    return (
        pd.DataFrame(sims)
        .sort_values("similarity", ascending=False)
        .head(top_n)
    )
