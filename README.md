# Music Similarity Finder
### Seminar paper for the course 'Algorithms and Data Structures 2'
Author: Kanita Tafro  
*University of Sarajevo, Data Science and Artificial Intelligence*

 ## Abstract
Music similarity estimation is a central problem in Music Information Retrieval (MIR), with applications in retrieval, recommendation, and exploratory browsing. This paper
presents a multi-layer music similarity framework that integrates psychoacoustic modeling, interpretable audio feature grouping, nonlinear dimensionality reduction, and melody-based querying.
Low- and mid-level audio descriptors are first extracted from music recordings and perceptually biased using A-weighting to approximate human loudness sensitivity. The features are
then organized into musically meaningful groups representing timbre, spectral characteristics, harmony, and rhythm, enabling explainable similarity analysis. To support visualization and
exploratory interaction, Uniform Manifold Approximation and Projection (UMAP) is applied separately to each feature group, producing low-dimensional embeddings that preserve perceptual
relationships in the data. Finally, a Query by Humming (QbH) component based on mel-spectral embeddings enables melody-driven retrieval that is robust to performance variability. The
proposed architecture emphasizes interpretability, perceptual relevance, and modularity, providing a scalable foundation for music similarity exploration in academic and educational settings.


📽️ [VIDEO PRESENTATION](https://youtu.be/RwvadVpLAsE?si=_S13-lRM2IquWtP0)


## Instructions

1. Follow the link to the GTZAN dataset and the instructions for data preparation located in *data/README.md*.
2. Run
   
   ```
   python feature_extraction.py
   ```
   This will take around 30 minutes and it will output the ***features_full_combined.csv*** alongside the ***features_chunks/*** folder thats serves as checkpoints to feature extraction. In `feature_extraction.py` the constants
   ```
   N_CHUNKS = 10           # number of parts to divide files into
   START_CHUNK = 0         # which chunk to start from (0-indexed)
   ```
   define the number of checkpoints (chunks) during feature extraction (set to 10 since there are 10 genres*100 samples per genre) and the 0-indexed starting chunk (START_CHUNK will not be set to 0 only if feature extraction was forcefully terminated).
3. Run all cells in `results.ipynb`. This will produce ***qbh_mel_features.csv*** and will display detailed results.
   * You can add any humming file (must be in wav format) to ***queries/*** and make sure you change the query path in the code from  ```query_path = "queries/humming2.wav"```  to your filename.
   * The `query_path` variable is defined in the 3rd cell in the notebook. Don't move it up.
5. Run
   
   ```
   streamlit run app.py
   ```
   which will open a new tab in your browser and start the streamlit application.

## Original Features (Layers)

### Layer 1 - A-Weighted Pschoacoustic Features
The human ear is most sensitive to frequencies between 500 Hz and 8 kHz and responds less to very low-pitch or high-pitch noises. The frequency weightings used on a modern sound level meter are A-, C-, and Z-weightings. The most common weighting used in noise measurement is A-weighting which is the first layer to the music similarity algorithm this experiment proposes.
A-weighting is a frequency-dependent weighting function derived from the equal-loudness contours, originally reported by Fletcher and Munson (1933), designed to approximate the sensitivity of human hearing at moderate sound pressure levels (SPLs) by reducing very low and high frequencies and emphasizing the mid-frequency range (approximately 2-5 kHz) where human auditory perception is most sensitive. This perceptual model is standardized in IEC 61672 and enables spectral representations to reflect perceived rather than physical loudness.

In this experiment, A-weighting is applied at the feature level rather than to raw audio or Mel spectrograms. The input consists of pre-extracted Mel-band energy features stored in CSV format, where each Mel band represents a perceptually motivated frequency range. The center frequency of each band is used to evaluate the A-weighting curve, and the resulting weighting factors are applied directly to the corresponding Mel-band energies. Frequencies to which human hearing is less sensitive are attenuated, while perceptually salient mid-frequency bands are preserved, producing A-weighted feature vectors for downstream similarity computation.

This feature-level approach provides a lightweight and interpretable perceptual approximation without performing signal-level processing or modeling nonlinear auditory effects such as masking. By biasing similarity measures toward perceptually relevant spectral regions, the method improves alignment with human judgments of timbre while maintaining computational efficiency and compatibility with CSV-based MIR workflows.


### Layer 2 - Explainable Feature Groups
While A-weighting in layer 1 models how the human ear perceives relative loudness through A-weighted representations that approximate human loudness sensitivity, it remains focused on low-level auditory perception rather than musically interpretable structure. Layer 2 builds on this foundation by reorganizing mid-level audio descriptors into four perceptually and musically meaningful feature groups, thus shifting the system from perceptual weighting toward structurally interpretable musical dimensions.

Spectral and timbre features describe a sound’s acoustic energy across different frequencies. They serve as a useful approximation for timbre—the perceptual quality that differentiates sounds with the same pitch and loudness. In this experiment, **timbre** features include RMS energy and MFCCs, described by their mean and variance. They capture a sound's overall energy, spectral shape, and articulation. **Spectral** features consist of spectral centroid, bandwidth, and rolloff, also represented by their mean and variance. These quantify a sound’s brightness, its spread across frequencies, and the distribution of high-frequency energy. Collectively, these descriptors capture characteristics such as instrumentation, production style, and timbral texture, which are valuable for tasks like genre classification and audio similarity. Previous MIR research shows these features represent information that is separate from a song’s harmony, providing a useful complement to pitch-based data.

```
    timbre_features = (
        ['rms_mean', 'rms_var'] +
        [c for c in all_cols if c.startswith('mfcc')]  # all MFCC columns
    )

    spectral_features = [
        'spectral_centroid_mean', 'spectral_centroid_var',
        'spectral_bandwidth_mean', 'spectral_bandwidth_var',
        'rolloff_mean', 'rolloff_var'
    ]
```

**Chroma (harmony)** features represent the distribution of spectral energy across the twelve pitch classes while collapsing information across octaves. In this experiment, the chroma feature group contains statistics derived from a short-time Fourier transform (STFT mean and variance) as well as mean and variance descriptors derived from harmonic–percussive source separation. These features focus on tonal and harmonic information while remaining resistant to variations in timbre and pitch register. This makes them particularly effective for tasks like key detection and melodic similarity. Enhanced harmonic representations focusing on stable spectral peaks have been shown to improve robustness in polyphonic and real-world recordings.

```
    chroma_harmony_features = [
        'chroma_stft_mean', 'chroma_stft_var',
        'harmony_mean', 'harmony_var',
        'percussive_mean', 'percussive_var'
    ]
```

**Tempo (rhythm)** features describe how musical events are organized in time by capturing patterns in pulse and meter. This group includes a global tempo estimate and statistics for zero-crossing rate (mean and variance). These features describe higher-level temporal patterns that simpler spectral or harmonic descriptors cannot directly access. They have been shown to form an independent dimension of musical description in MIR systems. As a result, rhythm-based representations are particularly informative for distinguishing musical styles and explaining similarities driven by temporal structure.

```
    rhythm_tempo_features = [
        'tempo',
        'zero_crossing_rate_mean',
        'zero_crossing_rate_var'
    ]
```

This new layer organizes audio features into intuitive groups like timbre, harmony, and rhythm, rather than using individual technical metrics. This makes it possible to explain similarity (e.g. saying that two songs sound alike because of their shared harmony). These groups reflect well-established musical qualities and provide a clearer, more interpretable foundation for comparing music. The next layer will use these grouped features to calculate similarity, allowing control over how much each musical aspect influences the final result.

```
    a_weighted_features = [c for c in all_cols if c.startswith('a_weighted_mel')]
```



### Layer 3 - UMAP-Based Similarity Embeddings

After computing feature-specific similarities in the second layer, layer 3 applies **Uniform Manifold Approximation and Projection (UMAP)** to produce low-dimensional embeddings of the high-dimensional audio feature space. Each feature group (timbre, spectral, rhythm/tempo, chroma/harmony, and A-weighted perceptual features) is embedded separately, producing multiple 2D visualizations that capture distinct aspects of musical similarity. UMAP is a nonlinear dimensionality reduction technique that constructs a continuous manifold from high-dimensional data while preserving both local neighborhoods and global structure. It is particularly suited for audio and music information retrieval tasks, where features such as MFCCs, spectral descriptors, and chroma vectors are high-dimensional and perceptual relationships between tracks are obscured.  
Computing one UMAP per feature group preserves the interpretability established in Layers 1–2. Each feature group represents a semantically distinct aspect of musical content:

 * **Timbre UMAP**: Encodes instrument textures and sonic color, primarily derived from MFCCs and RMS features. Clusters in this space reveal similarities in timbral characteristics across tracks.
 * **Spectral UMAP**: Captures brightness and frequency distribution patterns using spectral centroid, bandwidth, and rolloff. Tracks close in this embedding share similar spectral envelopes.
   
<img width="490" height="290" alt="Image" src="https://github.com/user-attachments/assets/9e60a13d-24bf-4f5f-81dc-598b81d30748" /> <img width="490" height="290" alt="Image" src="https://github.com/user-attachments/assets/498c142b-43f0-4509-84fd-192eef9075c3" />

 * **Rhythm/Tempo UMAP**: Represents rhythmic energy and tempo similarity, informed by zero-crossing rates and beat-tracking. This allows the visualization of tempo- and rhythm-related clusters.
 * **Chroma/Harmony UMAP**: Emphasizes tonal and harmonic relationships, organizing tracks according to chord structures and pitch content.

<img width="490" height="290" alt="Image" src="https://github.com/user-attachments/assets/cbd4fd1e-3d99-44ab-88ec-4fe8c9c6b1ea" /> <img width="490" height="290" alt="Image" src="https://github.com/user-attachments/assets/be984a1c-4c4a-454b-b78e-8042b9515f88" />

 * **A-weighted Mel UMAP**: Integrates perceptually weighted loudness and energy features, reflecting the human auditory response in the clustering structure.

<img width="490" height="290" alt="Image" src="https://github.com/user-attachments/assets/ba2c59dd-8dc3-4e22-9648-a51dae605a05" />

Each embedding provides a musically interpretable 2D space where proximity corresponds to similarity within that feature group. Comparing embeddings across groups enables multi-dimensional exploration: a track may cluster by timbre in one embedding while aligning differently in rhythm or harmony space, revealing nuanced relationships. This approach preserves explainability and interpretability from layers 1 and 2, supporting both analytical insight and interactive exploration of complex music collections.



### Layer 4 - Query by Humming
**Query by Humming (QbH)** is a MIR task in which users retrieve musical works by vocally imitating a melody rather than using textual metadata. Hummed queries differ significantly from studio recordings in timbre, pitch stability, tempo, and recording conditions. These differences require audio representations that emphasize perceptually relevant musical content while remaining robust to noise and performance variability. Early QbH systems relied on symbolic representations such as pitch contours or note sequences, but these approaches are sensitive to pitch extraction errors and temporal misalignment. To address these limitations, recent systems adopt embedding-based retrieval, where both queries and reference tracks are represented as fixed-length vectors and compared using standard distance metrics.

In this project, QbH is implemented using mel-spectral embeddings with statistical pooling. Each audio signal is converted to a monophonic waveform and transformed into a mel spectrogram using a perceptually motivated frequency scale. Spectrograms are converted to a logarithmic amplitude representation and then center-cropped or zero-padded to a fixed number of time frames to ensure comparability across inputs. Per-frequency-band z-score normalization is applied to reduce loudness bias and inter-recording variability. Temporal information is summarized by computing the mean and standard deviation of each mel band over time, producing a fixed-length embedding. Both reference tracks and hummed queries are processed using the same pipeline. Retrieval is performed by computing cosine distances between embeddings, yielding a temporally invariant and noise-robust baseline QbH system.

Top 10 similar tracks to *queries/humming2.wav*:
```
        track_id   genre  similarity
764     pop00064     pop    0.983434
732     pop00032     pop    0.983204
437  hiphop00037  hiphop    0.983119
755     pop00055     pop    0.982884
888  reggae00088  reggae    0.981342
797     pop00097     pop    0.981117
748     pop00048     pop    0.980685
872  reggae00072  reggae    0.980614
777     pop00077     pop    0.980483
887  reggae00087  reggae    0.980379
```
<img width="712" height="297" alt="Image" src="https://github.com/user-attachments/assets/4aab5256-bddd-4a56-a470-27cac3183524" />


## References

1. J. Futrelle and J. S. Downie, “Interdisciplinary research issues in music information retrieval: ISMIR 2000–2002,” Journal of New Music Research, vol. 32, no. 2, pp. 121–131, 2003.  
2. P. Knees and M. Schedl, Music similarity and retrieval: an introduction to audio-and web-based strategies. Springer, 2016, vol. 36.  
3. T. George, E. Georg, and C. Perry, “Automatic musical genre classification of audio signals,” in Proceedings of the 2nd international symposium on music information retrieval, Indiana, vol. 144, 2001.  
4. B. L. Sturm, “The GTZAN dataset: Its contents, its faults, their effects on evaluation, and its future use,” arXiv preprint arXiv:1306.1461, 2013.  
5. G. Tzanetakis and P. Cook, “Music analysis and retrieval systems for audio signals,” Journal of the American Society for Information Science and Technology, vol. 55, no. 12, pp. 1077–1083, 2004.  
6. H. Fletcher and W. A. Munson, “Loudness, its definition, measurement and calculation,” Bell System Technical Journal, vol. 12, no. 4, pp. 377–430, 1933.  
7. G. Peeters, B. L. Giordano, P. Susini, N. Misdariis, and S. McAdams, “The timbre toolbox: Extracting audio descriptors from musical signals,” The Journal of the Acoustical Society of America, vol. 130, no. 5, pp. 2902–2916, 2011.  
8. M. Schedl, E. Gomez, J. Urbano et al., “Music information retrieval: Recent developments and applications,” Foundations and Trends® in Information Retrieval, vol. 8, no. 2-3, pp. 127–261, 2014.  
9. J. M. Grey, “Multidimensional perceptual scaling of musical timbres,” the Journal of the Acoustical Society of America, vol. 61, no. 5, pp. 1270–1277, 1977.  
10. G. Tzanetakis and P. Cook, “Musical genre classification of audio signals,” IEEE Transactions on speech and audio processing, vol. 10, no. 5, pp. 293–302, 2002.  
11. D. P. Ellis, “Classifying music audio with timbral and chroma features,” 2007.  
12. T. Fujishima, “Realtime Chord Recognition of Musical Sound: a System Using Common Lisp Music,” in International Conference on Mathematics and Computing, 1999. 
13. M. Mueller, Fundamentals of music processing: Audio, analysis, algorithms, applications. Springer, 2015, vol. 5.
14. E. D. Scheirer, “Tempo and beat analysis of acoustic musical signals,” The Journal of the Acoustical Society of America, vol. 103, no. 1, pp. 588–601, 1998.  
15. L. McInnes, J. Healy, and J. Melville, “UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction,” arXiv preprint arXiv:1802.03426, 2018.  
16. B. Ghojogh, A. Ghodsi, F. Karray, and M. Crowley, “Uniform manifold approximation and projection (umap) and its variants: tutorial and survey,” arXiv preprint arXiv:2109.02508, 2021.  
17. Y. Jiale and Z. Ying, “Visualization method of sound effect retrieval based on UMAP,” in 2020 IEEE 4th Information Technology, Networking, Electronic and Automation Control Conference (ITNEC), vol. 1. IEEE, 2020, pp. 2216–2220.  
18. P. Tovstogan, X. Serra, and D. Bogdanov, “Visualization of deep audio embeddings for music exploration and rediscovery,” Proceedings of the SMC 2022 Music technology and design, pp. 493–500, 2022.  
19. J. Salamon, E. G´omez, D. P. Ellis, and G. Richard, “Melody extraction from polyphonic music signals: Approaches, applications, and challenges,” IEEE Signal Processing Magazine, vol. 31, no. 2, pp. 118–134, 2014.  
20. E. J. Humphrey, J. P. Bello, and Y. LeCun, “Feature learning and deep architectures: New directions for music informatics,” Journal of Intelligent Information Systems, vol. 41, no. 3, pp. 461–481, 2013.  
21. E. Gomez, “Tonal description of polyphonic audio for music content processing,” INFORMS Journal on Computing, vol. 18, no. 3, pp. 294–304, 2006.

