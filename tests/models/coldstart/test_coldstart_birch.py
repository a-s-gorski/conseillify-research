import logging
import os
import sys
from collections import defaultdict
from typing import List, Set

import click
import gensim
import hdbscan
import numpy as np
import pandas as pd
from sklearn.cluster import Birch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from src.data.make_dataset import load_songs_encodings
from src.models.coldstart.cluster_birch_coldstart import (
    cluster_labels, embed_playlists, prepare_embedding_dict, prepare_playlist,
    process_playlist_names)
from src.models.coldstart.coldstart_birch_pipeline import \
    recommend_birch_coldstart_component
from tqdm import tqdm

from ..candidate_generation.test_candidate_generation import (
    calculate_map_at_k, calculate_mar_at_k, coverage, prepare_test_playlist,
    unknown_track)

sys.path.append("....")


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('reports_filepath', type=click.Path(exists=True))
def main(input_filepath: str, reports_filepath: str):
    playlist_names = pd.read_csv(os.path.join(input_filepath, "playlist_names.csv"))[
        "0"].to_numpy(dtype=str)
    playlists = pd.read_csv(os.path.join(
        input_filepath, "playlists.csv")).to_numpy()

    pn_train, p_train, pn_val, p_val = train_test_split(
        playlist_names, playlists, test_size=0.001)
    songs_encodings = load_songs_encodings(
        os.path.join(input_filepath, "songs_encodings.csv"))
    songs_encodings = {track_id: track_uri for track_uri,
                       track_id in songs_encodings.items()}
    pn_train = process_playlist_names(pn_train)
    model = gensim.models.Word2Vec(
        playlist_names, min_count=1, vector_size=10, window=5)
    embeddings_dict = prepare_embedding_dict(model)
    embedded_playlists = embed_playlists(embeddings_dict, p_train)
    pca = PCA(n_components=10)
    reduced_pn = pca.fit_transform(embedded_playlists)
    clusterer = hdbscan.HDBSCAN(cluster_selection_epsilon=1)
    clusterer.fit(reduced_pn)
    n_labels = len(set(clusterer.labels_))
    logging.info(f"Clusters: {n_labels}")
    logging.info("Birch clustering")
    brc = Birch(n_clusters=n_labels)
    labels = brc.fit_predict(reduced_pn)
    clustered_tracks = cluster_labels(labels, p_train, songs_encodings)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
