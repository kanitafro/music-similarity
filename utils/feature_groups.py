"""
Feature grouping for explainability (Layer 2).

Groups are defined strictly based on the columns produced by
feature_extraction.py.
"""

def get_feature_groups(df):
    """
    Returns a dictionary mapping group names to feature column lists.
    Expects a DataFrame with scaled numeric features + track_id, genre.
    """

    all_cols = df.columns

    # ------------------------------------------------
    # TIMBRE
    # ------------------------------------------------
    timbre_features = (
        ['rms_mean', 'rms_var'] +
        [c for c in all_cols if c.startswith('mfcc')]
    )

    # ------------------------------------------------
    # SPECTRAL
    # ------------------------------------------------
    spectral_features = [
        'spectral_centroid_mean', 'spectral_centroid_var',
        'spectral_bandwidth_mean', 'spectral_bandwidth_var',
        'rolloff_mean', 'rolloff_var'
    ]

    # ------------------------------------------------
    # RHYTHM / TEMPO
    # ------------------------------------------------
    rhythm_tempo_features = [
        'tempo',
        'zero_crossing_rate_mean',
        'zero_crossing_rate_var'
    ]

    # ------------------------------------------------
    # CHROMA / HARMONY
    # ------------------------------------------------
    chroma_harmony_features = [
        'chroma_stft_mean', 'chroma_stft_var',
        'harmony_mean', 'harmony_var',
        'percussive_mean', 'percussive_var'
    ]

    # ------------------------------------------------
    # A-WEIGHTED PERCPTUAL (Layer 1)
    # ------------------------------------------------
    a_weighted_features = [c for c in all_cols if c.startswith('a_weighted_mel')]

    return {
        "timbre": timbre_features,
        "spectral": spectral_features,
        "rhythm_tempo": rhythm_tempo_features,
        "chroma_harmony": chroma_harmony_features,
        "a_weighted": a_weighted_features
    }


def get_grouped_dataframes(X_scaled):
    """
    Helper function used for layer 2.

    Returns one DataFrame per feature group, including track_id and genre.
    """

    base_cols = ['track_id', 'genre']
    groups = get_feature_groups(X_scaled)

    grouped_dfs = {}
    for name, features in groups.items():
        grouped_dfs[name] = X_scaled[base_cols + features].copy()

    return grouped_dfs
