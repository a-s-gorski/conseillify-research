import logging
from builtins import len
from typing import Optional

import click
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from ..diversifaction.diversify import recommend_n_diverse
from ..learn_to_rank.make_rank_pipeline import ranking_component
from .diversify import recommend_n_diverse


def diversify_component(features: NDArray, playlists: NDArray, user_playlist: NDArray, n: Optional[int] = 10, sample_size: Optional[int] = 5, hid_dim: Optional[int] = 16, epochs: Optional[int] = 100):
    features = features[:, 2:13]
    predictions, model, predictor, hidden_dim = recommend_n_diverse(features, playlists, user_playlist, n, sample_size, hid_dim, epochs)
    predictions = np.array(predictions)
    predictions = predictions.astype('int32')
    return predictions

def rank_and_diversify_component(features: NDArray, playlists: NDArray, user_playlist: NDArray, ranking_n: Optional[int]=100, ranking_epochs: Optional[int]=1,  diversify_n: Optional[int] = 10, diversify_sample_size: Optional[int] = 5, diversify_hid_dim: Optional[int] = 16, diversify_epochs: Optional[int] = 100):
    relevant_playlists, features, user_playlist, recommended_tracks = ranking_component(features, playlists, user_playlist, ranking_n, ranking_epochs)
    additional_recommendations = diversify_component(features, relevant_playlists, user_playlist, diversify_n, diversify_sample_size, diversify_hid_dim, diversify_epochs)
    additional_recommendations = additional_recommendations.astype('int64')
    additional_tracks = features[additional_recommendations, 0].flatten()
    additional_tracks = additional_tracks.tolist()
    logging.info(f"additional_tracks {additional_tracks}")
    recommended_tracks = recommended_tracks.tolist()
    logging.info(f"common_tracks {set(additional_tracks) & set(recommended_tracks)}")
    recommendations = list(set(recommended_tracks + additional_tracks))
    return relevant_playlists, features, user_playlist, recommendations # modify recommended_tracks





@click.command()
@click.argument('features_path', type=click.Path(exists=True))
@click.argument('playlists_path', type=click.Path(exists=True))
@click.argument('user_playlist_path', type=click.Path(exists=True))
def main(features_path: str, playlists_path: str, user_playlist_path: str):
    logging.info("Loading features, playlists, user_playlist")
    features = np.array(pd.read_csv(features_path, index_col=False).to_numpy())
    playlists = pd.read_csv(playlists_path, index_col=False).to_numpy()
    user_playlist = np.array(pd.read_csv(user_playlist_path, index_col=False).to_numpy()[:, 1].flatten())
    rank_and_diversify_component(features, playlists, user_playlist)
    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
