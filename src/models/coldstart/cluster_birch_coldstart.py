import logging
import os
import random
import string
import sys
import time
from collections import defaultdict
from random import sample
from typing import Dict, List, Optional, Set

import click
import gensim
import hdbscan
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from numpy.typing import ArrayLike, NDArray
from sklearn.cluster import Birch
from sklearn.decomposition import PCA
from tqdm import tqdm

from ...data.make_dataset import load_songs_encodings
from ...features.build_features import load_pickle, save_pickle


def missing_character(character):
    return ""


def process_playlist_names(playlist_names: List[str]) -> List[List[str]]:
    stop_words = set(stopwords.words('english'))

    printable = set(string.printable)
    playlist_names = [''.join(
        filter(lambda sign: sign in printable, name)) for name in playlist_names]
    playlist_names = [word_tokenize(sentence)
                      for sentence in tqdm(playlist_names)]
    playlist_names = [list(filter(lambda word: word.lower(
    ) not in stop_words, sentence)) for sentence in playlist_names]
    return playlist_names


def prepare_embedding_dict(model: Word2Vec) -> Dict[str, ArrayLike]:
    embeddings = {word: np.array(embedding) for word, embedding in zip(
        model.wv.index_to_key, model.wv)}
    return embeddings


def embed_playlists(embeddings_dict: Dict[str, ArrayLike], playlist_names: List, playlist_len=10):
    def embed_playlist(name):
        if name in embeddings_dict:
            return embeddings_dict[name]
        else:
            return ""
    embedded_playlists = [np.array(list(
        map(embed_playlist, name[:playlist_len]))).flatten() for name in playlist_names]
    max_name_len = max(list(map(len, embedded_playlists)))
    embedded_playlists = np.array(list(map(lambda embedding: np.pad(
        embedding, (0, max_name_len - len(embedding)), 'constant', constant_values=(0, 0)), embedded_playlists)))
    return embedded_playlists


def cluster_tracks(labels: List[int], playlists: List[List[int]]) -> Dict[int, Set[int]]:
    clustered_playlists = {}
    for label, playlist in tqdm(zip(labels, playlists)):
        if label == -1:
            continue
        if label not in clustered_playlists:
            clustered_playlists[label] = set()
        for track in playlist:
            if track == -1:
                continue
            clustered_playlists[label].add(track)
    return clustered_playlists


def missing_embedding():
    return np.zeros(10)


def missing_track():
    return np.zeros(30)


def missing_encoding():
    return ""


def prepare_playlist(embeddings_dict, pca, user_playlist) -> NDArray:
    processed_playlist = process_playlist_names(user_playlist)
    e_user_playlist = embed_playlists(embeddings_dict, processed_playlist)
    e_user_playlist = list(map(lambda p: p[:100], e_user_playlist))
    e_user_playlist = np.array(list(map(lambda p: np.pad(
        p, (0, 100-len(p)), 'constant', constant_values=(0, 0)), e_user_playlist)))
    rd_user_playlist = pca.transform(e_user_playlist)
    return rd_user_playlist


def cluster_labels(labels: List[int], playlists: NDArray, songs_encodings: Dict[int, str]) -> Dict[int, Set[str]]:
    clustered_tracks = {}
    for playlist_index, (cluster_index, playlist) in tqdm(enumerate(zip(labels, playlists)), total=playlists.shape[0]):
        if not cluster_index in clustered_tracks:
            clustered_tracks[cluster_index] = set()
        for track in playlist:
            if track != -1:
                clustered_tracks[cluster_index].add(songs_encodings[track])
    return clustered_tracks


def recommend_n_tracks(brc: Birch, clustered_tracks: Dict[int, Set], processed_playlist: List[float], n_recommendations: Optional[int] = 100):
    cluster_id = brc.predict(processed_playlist)[0]
    return random.sample(tuple(clustered_tracks[cluster_id]), min(n_recommendations, len(clustered_tracks[cluster_id])))


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('model_filepath', type=click.Path(exists=True))
def main(input_filepath: str, model_filepath: str):

    data_processing_start = time.time()
    logging.info("Loading playlists data")
    playlist_names = pd.read_csv(os.path.join(input_filepath, "playlist_names.csv"))[
        "0"].to_numpy(dtype=str)
    playlists = pd.read_csv(os.path.join(
        input_filepath, "playlists.csv")).to_numpy()

    logging.info("Loading songs encodings")
    songs_encodings = load_songs_encodings(
        os.path.join(input_filepath, "songs_encodings.csv"))
    songs_encodings = {track_id: track_uri for track_uri,
                       track_id in songs_encodings.items()}
    logging.info("Processing playlist info")
    playlist_names = process_playlist_names(playlist_names)

    logging.info("Training word2vec model")
    model = gensim.models.Word2Vec(
        playlist_names, min_count=1, vector_size=10, window=5)

    logging.info("Building embeddings")
    embeddings_dict = prepare_embedding_dict(model)
    embedded_playlists = embed_playlists(embeddings_dict, playlist_names)
    logging.info(
        f"Data processing execution time {time.time() - data_processing_start}")

    logging.info("Reducing dimensionality")
    pca = PCA(n_components=10)
    reduced_pn = pca.fit_transform(embedded_playlists)

    logging.info("Getting cluster number with hdbscan")
    clustering_start = time.time()
    clusterer = hdbscan.HDBSCAN(cluster_selection_epsilon=1)
    clusterer.fit(reduced_pn)
    print(clusterer.labels_[:100])
    logging.info(f"HDBSCAN execution time {time.time()-clustering_start}")
    n_labels = len(set(clusterer.labels_))
    logging.info(f"Clusters: {n_labels}")

    logging.info("Birch clustering")
    brc = Birch(n_clusters=n_labels)
    labels = brc.fit_predict(reduced_pn)
    clustered_tracks = cluster_labels(labels, playlists, songs_encodings)

    logging.info("Starting user processing")
    inf_start = time.time()
    user_playlist = ["rock and roll"]
    rd_user_playlist = prepare_playlist(embeddings_dict, pca, user_playlist)
    logging.info(f"Reduced playlist features {rd_user_playlist}")

    recommendations = recommend_n_tracks(
        brc, clustered_tracks, rd_user_playlist, 50)
    logging.info(f"Recommended tracks {recommendations}")
    logging.info(f"Inference execution time {time.time() - inf_start}")
    logging.info(f"Saving artifacts to {model_filepath}")
    sys.setrecursionlimit(10000)
    save_pickle(brc, os.path.join(model_filepath, "brc.pkl"))
    save_pickle(pca, os.path.join(model_filepath, "pca.pkl"))
    save_pickle(embeddings_dict, os.path.join(
        model_filepath, "embeddings_dict.pkl"))
    save_pickle(clustered_tracks, os.path.join(
        model_filepath, "clustered_tracks.pkl"))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
