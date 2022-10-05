import logging
from builtins import len
from datetime import datetime
from json import load
from pydoc import cli
from typing import Any, Dict, List, Optional, Tuple

import click
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from ...data.make_dataset import load_songs_encodings
from ...features.build_features import load_pickle, save_pickle
from .generate_candidates import extract_n_tracks, extract_relevant


def encode_tracks(tracks: List[str], songs_encodings: Dict[str, int]) -> List[int]:
    tracks = [songs_encodings[track]
              for track in tracks if track in songs_encodings]
    tracks = np.pad(tracks, (0, 375-len(tracks)), constant_values=(0, 0))
    return tracks


def decode_tracks(tracks: List[int], songs_encodings: Dict[str, int]) -> List[str]:
    reverse_se = {v: k for k, v in songs_encodings.items()}
    tracks = [reverse_se[track] for track in tracks if track in reverse_se]
    return tracks


def build_user_tuples(tracks: List[int]):
    return [(0, track_id, 1) for track_id in tracks if track_id != 0]


def candidate_generation_component(model_path: str, dataset_path: str, encodings_path: str, user_playlist: List[str], features_path: str, playlists_path: str, N: Optional[int] = 1000) -> Tuple[Any, ArrayLike, NDArray, List[int], List[str]]:
    model = load_pickle(model_path)
    dataset = load_pickle(dataset_path)
    encodings = load_songs_encodings(encodings_path)
    playlists = pd.read_csv(playlists_path, index_col=False).to_numpy()
    features = pd.read_csv(features_path).to_numpy()
    encoded_tracks = encode_tracks(user_playlist, encodings)
    tuples = build_user_tuples(encoded_tracks)
    interacions, _ = dataset.build_interactions(tuples)
    tracks = extract_n_tracks(model, 0, interacions.shape[0]+1, N)
    decoded_tracks = decode_tracks(tracks, encodings)
    relevant_playlists, tracks, features, encodings = extract_relevant(
        playlists, tracks, features, encoded_tracks)
    user_playlist = np.array([encodings[track]
                     for track in user_playlist if track in encodings])
    user_playlist = np.pad(user_playlist, (0, 375-len(user_playlist)), constant_values=(0,0))

    return relevant_playlists, tracks, features, user_playlist, decoded_tracks


@click.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('dataset_path', type=click.Path(exists=True))
@click.argument('encodings_path', type=click.Path(exists=True))
@click.argument('features_path', type=click.Path(exists=True))
@click.argument('playlists_path', type=click.Path(exists=True))
def main(model_path: str, dataset_path: str, encodings_path: str, features_path: str, playlists_path: str):
    user_playlist = ['spotify:track:1i5uaqmAkETSaIfZkcVYkx', 'spotify:track:67dA1a6OCUtLHgq9qdQ216',
                     'spotify:track:0MabrxpL9vrCJeOjGMnGgM', 'spotify:track:0F1yb5tFzXDocAXqcljA9H',
                     'spotify:track:1i5uaqmAkETSaIfZkcVYkx', 'spotify:track:1bweOqaO6SIGRRMupc7zMm',
                     'spotify:track:20ztml2STRF7Sq1UaBB6ox', 'spotify:track:2iLxXSM7AOzB4RCNzk4bjd',
                     'spotify:track:6EfP7rkoR1L1OpDzKN5lXX', 'spotify:track:7g2mskmb0okRLDtx0vtobh',
                     'spotify:track:0F1yb5tFzXDocAXqcljA9H', 'spotify:track:1Q3t9fWvHUXKsMmpD2XpUu',
                     'spotify:track:4mbYAXxvtRHjLZT92MTT8k', 'spotify:track:1i5uaqmAkETSaIfZkcVYkx',
                     'spotify:track:20ztml2STRF7Sq1UaBB6ox', 'spotify:track:20ztml2STRF7Sq1UaBB6ox',
                     'spotify:track:4YfHAn7SSb4onOyOHLlCbd', 'spotify:track:2iLxXSM7AOzB4RCNzk4bjd',
                     'spotify:track:2QPW157Ms76KiW8rByD73h', 'spotify:track:3JNxxCTPp5tDIsi8uAAj5j',
                     'spotify:track:3iag81mU7BylirRwRYXd8E', 'spotify:track:1oxPrSHvcFiX9cqiQXCIHE',
                     'spotify:track:2kWowW0k4oFymhkr7LmvzO', 'spotify:track:71VtqHXE00DYtK8hhPaKHn',
                     'spotify:track:6TOHnScC19Aw5fxiyPHB4z', 'spotify:track:2wBEMqe81CXIX88kph8GBO',
                     'spotify:track:7g2mskmb0okRLDtx0vtobh', 'spotify:track:0F1yb5tFzXDocAXqcljA9H']
    start = datetime.now()
    relevant_playlists, tracks, features, user_playlist, decoded_track_uris = candidate_generation_component(
        model_path, dataset_path, encodings_path, user_playlist, features_path, playlists_path)
    
    logging.info(f"Component execution time {datetime.now() - start}")



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
