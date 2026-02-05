import umap
import pandas as pd

def compute_umap(
    X,
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    random_state=42
):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric="euclidean",
        random_state=random_state
    )

    embedding = reducer.fit_transform(X)
    return embedding

def compute_group_umaps(X_scaled, feature_groups):
    umap_results = {}

    base_cols = ["track_id", "genre"]

    for group, cols in feature_groups.items():
        # Filter to only columns that actually exist in X_scaled
        valid_cols = [c for c in cols if c in X_scaled.columns]
        
        if len(valid_cols) == 0:
            continue

        X_group = X_scaled[valid_cols].values

        embedding = compute_umap(X_group)

        df_umap = pd.DataFrame(
            embedding,
            columns=["umap_x", "umap_y"]
        )

        df_umap["track_id"] = X_scaled["track_id"].values
        df_umap["genre"] = X_scaled["genre"].values

        umap_results[group] = df_umap

    return umap_results


def compute_qbh_umap(X_qbh):
    """Compute UMAP embedding for QbH dataset."""
    feature_cols = X_qbh.columns.difference(["track_id", "genre"])
    embedding = compute_umap(X_qbh[feature_cols].values)

    df = pd.DataFrame(embedding, columns=["umap_x", "umap_y"])
    df["track_id"] = X_qbh["track_id"]
    df["genre"] = X_qbh["genre"]
    return df
