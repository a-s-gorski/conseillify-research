import logging
import os
import string
import time
from collections import defaultdict
from random import sample
from typing import Dict, List, Optional, Set

import click
import dill
import gensim
import hdbscan
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from numpy.typing import ArrayLike, NDArray
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
        return embeddings_dict[name]
    embedded_playlists = [np.array(list(
        map(embed_playlist, name[:playlist_len]))).flatten() for name in playlist_names]
    max_name_len = max(list(map(len, embedded_playlists)))
    embedded_playlists = np.array(list(map(lambda embedding: np.pad(
        embedding, (0, max_name_len - len(embedding)), 'constant', constant_values=(0, 0)), embedded_playlists)))
    return embedded_playlists


def recommend_coldstart_cluster(clusterer: hdbscan.HDBSCAN, pca: PCA, playlist_name: str, embeddings_dict: Dict, max_name_len=100) -> int:
    playlist_names = process_playlist_names([playlist_name, ])[0]
    playlist_names = playlist_names[:10]
    playlist_embedding = np.array(
        list(map(lambda name: embeddings_dict[name], playlist_names))).flatten()
    playlist_embedding = playlist_embedding[:max_name_len]
    playlist_embedding = np.pad(playlist_embedding, (0, max_name_len - len(
        playlist_embedding)), 'constant', constant_values=(0, 0)).reshape(1, -1)
    playlist_embedding = pca.transform(playlist_embedding)
    approx_labels, approx_probs = hdbscan.approximate_predict(
        clusterer, playlist_embedding)
    probs = hdbscan.membership_vector(clusterer, playlist_embedding)
    if not approx_labels:
        return np.argmax(probs)
    if approx_labels[0] == -1:
        return np.argmax(probs)
    return approx_labels[0]


def select_n_from_cluster(clustered_tracks: Dict[int, Set[int]], cluster_id: int, n: int = 50):
    tracks_set = clustered_tracks[cluster_id]
    return sample(tracks_set, n)


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


class ColdstartRecommender:
    def __init__(self, embeddings_dict: Dict[str, List[int]], pca: PCA, clusterer: hdbscan.HDBSCAN, clustered_tracks: Dict[int, Set[int]], songs_encodings: Dict[int, str]):
        self._embeddings_dict = defaultdict(missing_embedding, embeddings_dict)
        self._pca = pca
        self._clusterer = clusterer
        self._clustered_tracks = defaultdict(missing_track, clustered_tracks)
        self._songs_encodings = defaultdict(missing_encoding, songs_encodings)

    def recommend_n_tracks(self, playlist_name: str, n: Optional[int] = 50):
        cluster_id = recommend_coldstart_cluster(
            self._clusterer, self._pca, playlist_name, self._embeddings_dict)
        logging.info(f"Cluster_id: {cluster_id}")
        track_ids = select_n_from_cluster(self._clustered_tracks, cluster_id)
        logging.info(f"track_ids: {track_ids}")
        track_uris = [self._songs_encodings[track_id]
                      for track_id in track_ids if track_id in self._songs_encodings]
        logging.info(f"Track_uris: {track_uris}")
        return track_uris


def missing_song_encoding(track):
    return -1


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

    training_start = time.time()
    logging.info("Reducing dimensionality using PCA")
    pca = PCA(n_components=3)
    reduced_pn = pca.fit_transform(embedded_playlists)[:200000, :]

    logging.info("Performing clustering")
    clusterer = hdbscan.HDBSCAN(
        cluster_selection_epsilon=1, prediction_data=True, alpha=0.5)
    clusterer.fit(reduced_pn)
    labels = clusterer.labels_
    clustered_tracks = cluster_tracks(labels, playlists)
    logging.info(f"Training time: {time.time() - training_start}")

    logging.info("Building coldstart model")
    coldstart_model = ColdstartRecommender(
        embeddings_dict, pca, clusterer, clustered_tracks, songs_encodings)

    recommendation_time = time.time()
    logging.info("Performing inference")
    print(coldstart_model.recommend_n_tracks("rock metal disco"))
    logging.info(
        f"Recommendation execution time: {time.time() - recommendation_time}")
    logging.info("Saving artifacts")

    with open(os.path.join(model_filepath, "embedding.pkl"), 'wb') as file:
        dill.dump(embeddings_dict, file)
    print(len(set(list(songs_encodings.keys()))))
    save_pickle(embeddings_dict, os.path.join(model_filepath, "embedding.pkl"))
    save_pickle(pca, os.path.join(model_filepath, "pca.pkl"))
    save_pickle(clusterer, os.path.join(model_filepath, "clusterer.pkl"))
    save_pickle(clustered_tracks, os.path.join(
        model_filepath, "clustered_tracks.pkl"))
    save_pickle(songs_encodings, os.path.join(
        model_filepath, "songs_encodings.pkl"))


# KNN / FLANN, WARD-clustering, ew knn
# https://en.wikipedia.org/wiki/Ward%27s_method


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
