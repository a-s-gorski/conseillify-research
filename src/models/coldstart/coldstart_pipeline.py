import logging
import time
from builtins import len
from datetime import datetime
from json import load
from typing import List, Optional

import click
import dill

from ...features.build_features import load_pickle
from .cluster_coldstart import ColdstartRecommender


def coldstart_component(playlist_name: str, embeddings_path: str, pca_path: str, clusterer_path: str, clustered_tracks_path: str, songs_encodings_path: str, n: Optional[int] = 100) -> List[str]:
    embeddings_dict = load_pickle(embeddings_path)
    pca = load_pickle(pca_path)
    clusterer = load_pickle(clusterer_path)
    clustered_tracks = load_pickle(clustered_tracks_path)
    songs_encodings = load_pickle(songs_encodings_path)
    print(len(set(list(songs_encodings.keys()))))
    print(songs_encodings[57040])

    model = ColdstartRecommender(
        embeddings_dict, pca, clusterer, clustered_tracks, songs_encodings)
    return model.recommend_n_tracks(playlist_name, n)


@click.command()
@click.argument('embeddings_path', type=click.Path(exists=True))
@click.argument('pca_path', type=click.Path(exists=True))
@click.argument('clusterer_path', type=click.Path(exists=True))
@click.argument('clustered_tracks_path', type=click.Path(exists=True))
@click.argument('songs_encodings_path', type=click.Path(exists=True))
def main(embeddings_path: str, pca_path: str, clusterer_path: str, clustered_tracks_path: str, songs_encodings_path: str):
    playlist_name = "rock"
    logging.info("Starting recommendation")
    start = time.time()
    recommendations = coldstart_component(
        playlist_name, embeddings_path, pca_path, clusterer_path, clustered_tracks_path, songs_encodings_path, n=100)
    logging.info(f"Coldstart recommendation time {time.time() - start}")
    logging.info(f"Recommendations: {recommendations}")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
