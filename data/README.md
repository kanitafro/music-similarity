# Dataset instructions

The dataset used for this experiment is the GTZAN dataset, created by and named after George Tzanetakis, which contains 1000 half-minute music audio samples. The samples are categorized into 10 genres, with each genre containing
100 samples. The original MARSYAS (Music Analysis and Retrieval Systems for Audio Signals) server for the dataset download is known to be unstable and currently the only available source is a reupload of the dataset on Kaggle.

1. Download the dataset from [here](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
2. The zip file will contain ***genres_original/*** and ***images_original/*** folders, alongside 2 csv files (**features_3_seconds.csv** and **features_30_seconds.csv**)
   * Keep only `genres_original` folder and delete the rest
3. Replace the old jazz.00054.wav file with the original file from [here](https://drive.google.com/file/d/1_uPw77ZwFK1IXjFSp2S6BMTit5KJ1RA4/view)
4. Make sure that ***genres_original*** is located directly in the folder ***data/***
5. The dataset is ready
