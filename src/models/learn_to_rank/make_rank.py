from lightfm import LightFM
from lightfm.data import Dataset
import pandas as pd
import os
import numpy as np
from ..candidate_generation.generate_candidates import extract_relevant, extract_n_tracks
import click
import logging
from ...features.build_features import save_pickle


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('model_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath: str, model_filepath: str, output_filepath: str):
    logging.info("Loading features, playlists, user_playlist")
    features = pd.read_csv(os.path.join(input_filepath, "features.csv"), index_col=False).to_numpy()[:, 1:12]
    playlists = pd.read_csv(os.path.join(input_filepath, "playlists.csv"), index_col=False).to_numpy()
    user_playlist = pd.read_csv(os.path.join(input_filepath, "user_playlist.csv"), index_col=False).to_numpy()[:, 1].flatten()
    
    logging.info("setting up data")
    N_USERS = playlists.shape[0]
    unique_tracks = set(playlists.flatten())
    unique_tracks.discard(0)
    N_ITEMS = len(unique_tracks)
    N_FEATURES = features.shape[1]
    interactions_tuples = np.array([(user_id, track_id, 1) for user_id, playlist in enumerate(playlists) for track_id in playlist if track_id != 0])
    features_tuples = np.array([(item_id+1, {feature_id: feature_value} )for item_id, f in enumerate(features) for feature_id, feature_value in enumerate(f)])
    logging.info(f"N_USERS: {N_USERS}, N_ITEMS: {N_ITEMS}, N_FEATURES: {N_FEATURES}")

    logging.info("Setting up dataset")
    dataset = Dataset(item_identity_features=True)
    dataset.fit(users=np.arange(0, N_USERS), items=np.arange(1, N_ITEMS+1), item_features=np.arange(0, N_FEATURES))

    logging.info("Builiding interactions")
    interactions, _ = dataset.build_interactions(interactions_tuples)
    item_features = dataset.build_item_features(features_tuples)

    logging.info("Fitting model")
    model = LightFM(no_components=2, loss='warp')
    model.fit(interactions, item_features=item_features, epochs=1, verbose=True)

    logging.info("Fitting user")
    user_playlist_tuples = [(0, track_id, 1) for track_id in user_playlist if track_id != 0]
    user_interactions, _ = dataset.build_interactions(user_playlist_tuples)
    model.fit_partial(user_interactions, item_features=item_features, epochs=3, verbose=True)

    logging.info("Making predictions")
    predictions = extract_n_tracks(model, 0, user_interactions.shape[0], 100)
    
    logging.info("Extracting relevant data")
    relevant_playlists, tracks, features, encodings = extract_relevant(playlists, predictions, features, user_playlist)

    user_playlist = np.array([encodings[track] for track in user_playlist if track in encodings])
    user_playlist = np.pad(user_playlist, (0, 375-len(user_playlist)), constant_values=(0,0))

    logging.info("Saving data")
    pd.DataFrame(relevant_playlists).to_csv(os.path.join(output_filepath, "playlists.csv"))
    pd.DataFrame(user_playlist).to_csv(os.path.join(output_filepath, "user_playlist.csv"))
    pd.DataFrame(item_features).to_csv(os.path.join(output_filepath, "features.csv"))
    
    logging.info("Saving model")
    save_pickle(model, os.path.join(model_filepath, "ranking_model.pkl"))
        


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
