import logging
import os
from typing import Any, Optional

import click
import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
from numpy.typing import ArrayLike, NDArray

from ...features.build_features import save_pickle
from ..candidate_generation.generate_candidates import extract_n_tracks
from ..learn_to_rank.make_rank import extract_relevant, prepare_ranking


def ranking_component(features: NDArray, playlists: NDArray, user_playlist: ArrayLike, N: Optional[int]= 100, epochs: Optional[int]=1):
    model, user_interactions = prepare_ranking(features, playlists, user_playlist, epochs)

    logging.info("Making predictions")
    predictions = extract_n_tracks(model, 0, user_interactions.shape[0], 100)
    
    logging.info("Extracting relevant data")
    relevant_playlists, recommended_tracks, features, encodings = extract_relevant(playlists, predictions, features, user_playlist)

    user_playlist = np.array([encodings[track] for track in user_playlist if track in encodings])
    user_playlist = np.pad(user_playlist, (0, 375-len(user_playlist)), constant_values=(0,0))

    return relevant_playlists, features, user_playlist, recommended_tracks

@click.command()
@click.argument('features_path', type=click.Path(exists=True))
@click.argument('playlists_path', type=click.Path(exists=True))
@click.argument('user_playlist_path', type=click.Path(exists=True))
def main(features_path: str, playlists_path: str, user_playlist_path: str):
    logging.info("Loading features, playlists, user_playlist")
    features = np.array(pd.read_csv(features_path, index_col=False).to_numpy())
    playlists = pd.read_csv(playlists_path, index_col=False).to_numpy()
    user_playlist = np.array(pd.read_csv(user_playlist_path, index_col=False).to_numpy()[:, 1].flatten())
    relevant_playlists, features, user_playlist, recommended_tracks = ranking_component(features, playlists, user_playlist)
    print(len(recommended_tracks))
    print(relevant_playlists.shape)
    print(features.shape)
    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
