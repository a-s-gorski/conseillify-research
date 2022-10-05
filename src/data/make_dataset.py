import json
import logging
import os
from enum import unique
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import click
import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from tqdm import tqdm

SONG_FEATURES_COLS = ["danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness",
                      "instrumentalness", "liveness", "valence", "tempo"]


def load_songs_features(input_path: str) -> Dict[str, np.ndarray]:
    songs_features_df = pd.read_csv(input_path)
    songs_features_dict = {}
    for _, row in tqdm(songs_features_df.iterrows(), total=songs_features_df.shape[0]):
        songs_features_dict[row[1]] = np.array(row[2:-2].to_list())
    return songs_features_dict


def save_songs_encodings(songs_encodings: Dict, output_path: str):
    songs_encodings_df = pd.DataFrame(
        {'track_uris': list(songs_encodings.keys()), 'track_encoding': list(songs_encodings.values())})
    songs_encodings_df.to_csv(output_path, index=False)


def load_songs_encodings(input_path: str) -> Dict[str, int]:
    songs_encodings_df = pd.read_csv(input_path, index_col=False)
    songs_encodings = {track_uri: track_encoding for track_uri, track_encoding in
                       zip(songs_encodings_df.track_uris, songs_encodings_df.track_encoding)}
    return songs_encodings


def add_track_ids(songs_features: pd.DataFrame, song_encodings: Dict[str, int]) -> pd.DataFrame:
    songs_features['track_id'] = songs_features.apply(
        lambda x: song_encodings[x['track_uri']], axis=1)
    return songs_features


def load_playlists(songs_encodings: Dict, input_path: str) -> List[List[int]]:
    playlists = []
    for file in tqdm(os.listdir(input_path)):
        with open(os.path.join(input_path, file)) as json_file:
            file_data = json.load(json_file)
            for playlist in file_data["playlists"]:
                temp_playlist = []
                for track in playlist["tracks"]:
                    track_uri = track["track_uri"]
                    if track_uri in songs_encodings:
                        temp_playlist.append(songs_encodings[track_uri])
                playlists.append(temp_playlist)
    return playlists


def process_playlists(playlists: List[List[int]]) -> NDArray[np.uint64]:
    max_playlist_len = max([len(p) for p in playlists])
    processed_playlists = []
    for playlist in tqdm(playlists):
        playlist = np.array(playlist, dtype=np.int64)
        playlist = np.pad(playlist, (0, max_playlist_len -
                          len(playlist)), mode='constant')
        processed_playlists.append(playlist)
    return np.array(processed_playlists)


def save_playlists(playlists: NDArray[np.uint64], output_path):
    df = pd.DataFrame(data=playlists, columns=[
                      f"track{i}" for i in range(playlists.shape[1])])
    df.to_csv(output_path, index=False)


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)

    logger.info('loading songs_features dataset')
    songs_features_df = pd.read_csv(os.path.join(input_filepath, "spotifysongfeatures/songs_features.csv"),
                                    index_col=False)

    logger.info('dropping column type from dataset')
    songs_features_df.drop(columns=["type"], inplace=True)

    logger.info("creating songs_encodings")
    songs_encodings = {track_uri: index + 1 for index,
                       track_uri in enumerate(songs_features_df.track_uri)}

    logger.info("adding song_encoding to song_features")
    songs_features_df = add_track_ids(songs_features_df, songs_encodings)

    logger.info("saving songs_features")
    songs_features_df.to_csv(os.path.join(
        output_filepath, "songs_features.csv"))

    logger.info("saving song encodings")
    save_songs_encodings(songs_encodings, os.path.join(
        output_filepath, "songs_encodings.csv"))

    logger.info("loading_playlists")
    playlists = load_playlists(songs_encodings, os.path.join(
        input_filepath, "spotify_million_playlist_dataset/data"))

    logger.info("processing tracks")
    playlists = process_playlists(playlists)

    logger.info("splitting playlists")
    train_p, holdout_p = train_test_split(playlists, test_size=0.0001)
    val_p, test_p = train_test_split(holdout_p, test_size=0.5)

    logger.info("saving_playlists")
    save_playlists(playlists, os.path.join(output_filepath, "playlists.csv"))
    save_playlists(train_p, os.path.join(
        output_filepath, "train_playlists.csv"))
    save_playlists(val_p, os.path.join(output_filepath, "val_playlists.csv"))
    save_playlists(test_p, os.path.join(output_filepath, "test_playlists.csv"))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    project_dir = Path(__file__).resolve().parents[2]
    load_dotenv(find_dotenv())
    main()
