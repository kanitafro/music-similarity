import matplotlib.pyplot as plt
import seaborn as sns
import librosa.display

def plot_waveform(y, sr, title="Waveform"):
    plt.figure(figsize=(16, 6))
    librosa.display.waveshow(y=y, sr=sr, color="#A300F9")
    plt.title(title)
    plt.show()

def plot_distribution(df, features):
    for feat in features:
        plt.figure(figsize=(6, 3))
        sns.histplot(df[feat], kde=True)
        plt.title(f"Distribution of {feat}")
        plt.show()

def plot_umap(df, title):

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="umap_x",
        y="umap_y",
        hue="genre",
        s=20,
        linewidth=0
    )
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
