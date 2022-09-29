import logging
import os
from typing import Any, Dict, List, Tuple

import click
import numpy as np
import pandas as pd
from lightfm import LightFM
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import load_npz

from ...features.build_features import load_pickle, save_pickle


def extract_n_tracks(model: LightFM, user_id: int, tracks_shape: int, n_tracks: int = 5):
    predictions = model.predict(user_id, np.arange(1, tracks_shape))
    predictions = [(score, index) for index, score in enumerate(predictions)]
    predictions.sort(reverse=False)
    predictions = predictions[:n_tracks]
    predictions = [index for _, index in predictions]
    return predictions

def extract_relevant(playlists: NDArray, tracks: List[int], songs_features: NDArray, user_playlist: ArrayLike) -> Tuple[Any, ArrayLike, NDArray, Dict[Any, int]]:
    def encode(track_encodings: Dict, track: int):
        if track == 0 or track not in track_encodings:
            return 0
        return track_encodings[track]
    tracks_set = set(tracks)
    relevant_playlists = [user_playlist, ]
    # extracting relevant playlists
    for playlist in playlists:
        if set(playlist) & set(tracks_set):
            relevant_playlists.append(playlist)
    # extracting all relevant tracks
    relevant_tracks = set(np.array(relevant_playlists).flatten())
    relevant_tracks.discard(0)
    tracks_encodings = {track: index + 1 for index, track in enumerate(relevant_tracks)}
    encode_vectorizer = np.vectorize(encode)
    relevant_playlists = encode_vectorizer(tracks_encodings, relevant_playlists)
    relevant_tracks_features = []
    for sf in songs_features:
        if sf[-1] in relevant_tracks:
            sf = sf[1:]
            sf[-1] = tracks_encodings[sf[-1]]
            relevant_tracks_features.append(sf)
    relevant_tracks_features = np.array(relevant_tracks_features)
    return relevant_playlists, np.array(list(relevant_tracks)), relevant_tracks_features, tracks_encodings

@click.command()
@click.argument('interactions_input_filepath', type=click.Path(exists=True))
@click.argument('playlists_input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('model_path', type=click.Path())
def main(interactions_input_filepath: str, playlists_input_filepath: str, output_filepath: str, model_path: str):
    logger = logging.getLogger(__name__)

    logger.info("Loading playlists, encodings, features")
    all_playlists = pd.read_csv(os.path.join(playlists_input_filepath, "playlists.csv"), index_col=False).to_numpy()
    songs_features = pd.read_csv(os.path.join(playlists_input_filepath, "songs_features.csv")).to_numpy()
    
    logger.info("Loading compressed interactions and dataset")
    train_interactions = load_npz(os.path.join(interactions_input_filepath, "train_interactions.npz"))
    val_interactions = load_npz(os.path.join(interactions_input_filepath, "val_interactions.npz"))
    test_interactions = load_npz(os.path.join(interactions_input_filepath, "test_interactions.npz"))
    dataset = load_pickle(os.path.join(interactions_input_filepath, "dataset_lightfm"))
    
    logger.info("Creating model")
    model = LightFM(no_components=100, loss='warp')
    logger.info("Fitting model")
    model.fit(train_interactions, epochs=3, verbose=True)
    model.fit_partial(val_interactions)
    model.fit_partial(test_interactions)

    logger.info("Fitting example user")
    example_interactions = all_playlists[999998]
    example_interactions = [(999998, track_id, 1) for track_id in example_interactions if track_id != 0]
    example_interactions, _ = dataset.build_interactions(example_interactions)
    model.fit_partial(example_interactions)

    logger.info("Saving model")
    save_pickle(model, os.path.join(model_path, "candidate_generator.pkl"))
    
    logger.info("Extracting predictions")
    predictions = extract_n_tracks(model, 999998, train_interactions.shape[0], 10000)
    relevant_playlists, tracks, features, encodings = extract_relevant(all_playlists, predictions, songs_features, all_playlists[999998])
    user_playlist = np.array([encodings[track] for track in all_playlists[999998] if track in encodings])
    user_playlist = np.pad(user_playlist, (0, 375-len(user_playlist)), constant_values=(0,0))

    logger.info("Saving relevant data")
    logger.info(f"Tracks common in selected and relevant playlists {set(user_playlist) & set(tracks)}")
    pd.DataFrame(relevant_playlists).to_csv(os.path.join(output_filepath, "playlists.csv"), index=False)
    pd.DataFrame(features).to_csv(os.path.join(output_filepath, "features.csv"), index=False)




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
