import tempfile
import streamlit as st
import pandas as pd
from pathlib import Path
import os
import plotly.express as px

from utils.process_spectrograms import extract_qbh_embedding, scale_features
from utils.similarity import find_similar_layered, find_similar_qbh
from utils.feature_groups import get_feature_groups
from utils.umap_layer import compute_group_umaps


BASE_DIR = Path(__file__).resolve().parent
AUDIO_BASE_DIR = BASE_DIR / "data" / "genres_original"


# -----------------------------
# App configuration
# -----------------------------
st.set_page_config(
    page_title="Music Similarity Explorer",
    layout="wide"
)

st.title("🎵 Music Similarity Explorer")

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("features_full_combined.csv")

    df["track_id"] = (
        df["filename"]
        .str.replace(".wav", "", regex=False)
        .str.replace(".", "", regex=False)
    )
    df["genre"] = df["filename"].str.split(".").str[0]

    X_scaled = scale_features(df, df[["track_id", "genre"]])
    feature_groups = get_feature_groups(X_scaled)
    umap_embeddings = compute_group_umaps(X_scaled, feature_groups)

    return df, X_scaled, umap_embeddings

df, X_scaled, umap_embeddings = load_data()

# Load Layer 4 QBH dataset
@st.cache_data
def load_qbh_dataset(path="qbh_mel_features.csv"):
    df_qbh = pd.read_csv(path)
    return df_qbh

df_qbh = load_qbh_dataset()
qbh_feature_cols = df_qbh.columns.difference(["track_id", "genre"])
qbh_n_mels = len(qbh_feature_cols) // 2
qbh_cols_even = len(qbh_feature_cols) % 2 == 0 and qbh_n_mels > 0

# -----------------------------
# Session state
# -----------------------------
if "selected_track" not in st.session_state:
    st.session_state.selected_track = df["track_id"].iloc[0]

if "qbh_features" not in st.session_state:
    st.session_state.qbh_features = None

if "qbh_results" not in st.session_state:
    st.session_state.qbh_results = None

if "qbh_active" not in st.session_state:
    st.session_state.qbh_active = False

# -----------------------------
# Layout: 3 columns
# -----------------------------
left_col, center_col, right_col = st.columns([1.2, 1.8, 2.2])

# ======================================================
# LEFT PANEL — MUSIC PLAYER
# ======================================================
with left_col:
    st.subheader("🎧 Player")

    # Playlist
    track_ids = df["track_id"].tolist()

    current_index = track_ids.index(
        st.session_state.selected_track
    )

    selected_track = st.selectbox(
        "Playlist",
        track_ids,
        index=int(current_index)
    )
    st.session_state.selected_track = selected_track

    track_row = df[df["track_id"] == selected_track].iloc[0]
    genre = track_row["genre"]

    # Cover image
    cover_path = f"covers/{genre}_cover.jpg"
    if os.path.exists(cover_path):
        st.image(
            cover_path,
            width="stretch"
        )
    else:
        st.warning("Cover image not found.")

    # Track name
    st.markdown(f"**{selected_track}**")

    # Audio player (timeline + controls handled by browser)
    audio_path = (
        AUDIO_BASE_DIR
        / genre
        / track_row["filename"]
    )

    if audio_path.exists():
        st.audio(str(audio_path))
    else:
        st.warning(f"Audio file not found: {audio_path}")

# ======================================================
# CENTER PANEL — RECOMMENDATIONS & QBH
# ======================================================
with center_col:
    st.subheader("🔎 Similar Tracks")

    # Compute results
    if st.session_state.get("qbh_active") and st.session_state.get("qbh_features") is not None:
        results = find_similar_qbh(
            query_vec=st.session_state.qbh_features,
            qbh_dataset=df_qbh,
            top_n=10,
            metric="cosine"
        )
        st.caption("Results from Query-by-Humming (Layer 4)")
    else:
        results = find_similar_layered(
            track_id=st.session_state.selected_track,
            X_scaled=X_scaled,
            top_n=10
        )
        st.caption("Results from Track-based Similarity")

    # Display each similar track with a mini player
    for _, row in results.iterrows():
        track_id = row['track_id']
        genre = row['genre']
        filename = df.loc[df['track_id'] == track_id, 'filename'].values[0]
        audio_path = AUDIO_BASE_DIR / genre / filename

        cols = st.columns([3, 2])  # Name | Player
        with cols[0]:
            if st.button(f"{track_id} ({genre})", key=f"name_{track_id}"):
                st.session_state.selected_track = track_id
                st.session_state.qbh_active = False
                st.rerun()
        with cols[1]:
            if audio_path.exists():
                st.audio(str(audio_path))
            else:
                st.warning("Audio not found")

    st.divider()

    # -----------------------------
    # Query by Humming (Layer 4)
    # -----------------------------
    st.subheader("🎤 Query by Humming")

    hum_file = st.file_uploader(
        "Upload a hummed audio query (.wav)",
        type=["wav"]
    )
    
    if hum_file is not None:
        #st.info("🎤 Humming query uploaded")
        st.audio(hum_file)
        
        if st.button("Find Similar Tracks (QBH)"):
            if not qbh_cols_even:
                st.error("QBH dataset feature columns are incompatible with mel mean/std format.")
            else:
                with st.spinner("Extracting humming features..."):
                    # extract_qbh_embedding() expects a file path (string), but hum_file 
                    # from st.file_uploader() is a Streamlit UploadedFile object. 
                    # Need to save it to a temporary file first with tempfile:
                   
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp.write(hum_file.getbuffer())
                        tmp_path = tmp.name
                    qbh_features = extract_qbh_embedding(tmp_path, n_mels=qbh_n_mels)
                    os.remove(tmp_path)  # Clean up temporary file

                # Store in session state to persist between reruns
                st.session_state.qbh_features = qbh_features
                st.session_state.qbh_active = True
                st.rerun()


# ======================================================
# RIGHT PANEL — UMAP VISUALIZATION
# ======================================================
UMAP_EXPLANATIONS = {
    "timbre": (
        "- X-axis: tracks with sharper, brighter sounds (like bells or guitars) vs softer, mellow sounds (like piano or vocals) ***[bright vs dark timbres]***\n"
        "- Y-axis: tracks with short, punchy sounds (like drums) vs smooth, continuous sounds (like strings or sustained notes) ***[percussive vs sustained timbres]***"
    ),
    "spectral": (
        "- X-axis: tracks with higher-pitched sounds vs lower-pitched sounds ***[higher vs lower spectral centroid/bandwidth]***\n"
        "- Y-axis: tracks with more variation in the sound (lots of highs and lows) vs more steady, consistent sounds ***[wider vs narrower spectral rolloff distributions]***"
    ),
    "rhythm_tempo": (
        "- X-axis: faster songs vs slower songs ***[higher vs lower tempo]***\n"
        "- Y-axis: songs with more complex rhythm patterns vs simpler, steady beats ***[higher vs lower zero-crossing rates]***"
    ),
    "chroma_harmony": (
        "- X-axis: songs that sound more ‘happy’ or major vs songs that sound more ‘sad’ or minor ***[tonal/harmonic content differences, like chord patterns]***\n"
        "- Y-axis: songs with richer, fuller chords vs songs with simpler harmony ***[stronger vs weaker harmonic textures]***"
    ),
    "a_weighted": (
        "- X-axis: songs with more high-pitched sounds vs more low-pitched sounds ***[higher vs lower frequency energy]***\n"
        "- Y-axis: songs that feel louder and more energetic vs songs that feel quieter and softer ***[more dynamic vs flatter perceived loudness]***"
    )
}

with right_col:
    st.subheader("🗺 Similarity Spaces (UMAP)")

    selected_track_id = st.session_state.selected_track

    # Custom titles for UMAP groups
    group_titles = {
        "timbre": "Timbre Features",
        "spectral": "Spectral Features",
        "rhythm_tempo": "Rhythm/Tempo Features",
        "chroma_harmony": "Chroma/Harmony Features",
        "a_weighted": "A-Weighted Perceptual Features"
    }

    for group, df_umap in umap_embeddings.items():
        title = group_titles.get(group, group.replace('_', ' ').title())
        st.markdown(f"**{title}**")

        df_plot = df_umap.copy()
        df_plot["is_query"] = (
            df_plot["track_id"] == selected_track_id
        )

        fig = px.scatter(
            df_plot,
            x="umap_x",
            y="umap_y",
            color="genre",
            symbol="is_query",
            hover_data=["track_id"],
            height=300
        )

        st.plotly_chart(fig, width="stretch")

        # Show intuitive explanation
        explanation = UMAP_EXPLANATIONS.get(group, "")
        if explanation:
            st.markdown(f"{explanation}")
