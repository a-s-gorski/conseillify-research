import logging
import time
from typing import List, Optional

import click

from ...features.build_features import load_pickle, save_pickle
from .cluster_birch_coldstart import prepare_playlist, recommend_n_tracks


def recommend_birch_coldstart_component(playlist_name: str, embeddings_dict_path: str, pca_path: str, brc_path: str, clusterd_tracks_path: str, N: Optional[int] = 100) -> List[str]:
    embeddings_dict = load_pickle(embeddings_dict_path)
    pca = load_pickle(pca_path)
    brc = load_pickle(brc_path)
    clustered_tracks = load_pickle(clusterd_tracks_path)
    rd_user_playlist = prepare_playlist(
        embeddings_dict, pca, [playlist_name, ])
    return recommend_n_tracks(brc, clustered_tracks, rd_user_playlist, N)


@click.command()
@click.argument('embeddings_dict_path', type=click.Path(exists=True))
@click.argument('pca_path', type=click.Path(exists=True))
@click.argument('brc_path', type=click.Path(exists=True))
@click.argument('clusterd_tracks_path', type=click.Path(exists=True))
def main(embeddings_dict_path: str, pca_path: str, brc_path: str, clusterd_tracks_path: str):
    user_playlist_name = "Rock"
    rec_start = time.time()
    recommendations = recommend_birch_coldstart_component(
        user_playlist_name, embeddings_dict_path, pca_path, brc_path, clusterd_tracks_path, 100)
    logging.info(f"Recommendations {recommendations}")
    logging.info(f"Inference time {time.time() - rec_start}")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
