import logging
import os
import pickle
from typing import Any

import click
import numpy as np
import pandas as pd
from lightfm.data import Dataset
from numpy.typing import NDArray
from scipy.sparse import save_npz
from tqdm import tqdm

def create_interactions_tuples(playlists: NDArray, starting_index: int = 0):
    return np.array([(user + starting_index, item, 1)  for user, row in tqdm(enumerate(playlists), total=playlists.shape[0]) for item in row if item != -1])

def load_pickle(input_path) -> Any:
    with open(input_path, 'rb+') as file:
        return pickle.load(file)

def save_pickle(object: Any, output_path: str):
    with open(output_path, 'wb+') as file:
        pickle.dump(object, file)


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())


def main(input_filepath, output_filepath):
    dataset = Dataset(user_identity_features=False, item_identity_features=True)
    logging.info("Loading playlists")
    train_playlist = pd.read_csv(os.path.join(input_filepath, "train_playlists.csv"), index_col=False).to_numpy()
    val_playlist = pd.read_csv(os.path.join(input_filepath, "val_playlists.csv"), index_col=False).to_numpy()
    test_playlist = pd.read_csv(os.path.join(input_filepath, "test_playlists.csv"), index_col=False).to_numpy()
    songs_encodings = pd.read_csv(os.path.join(input_filepath, "songs_encodings.csv"), index_col=False)
    
    N_USERS = train_playlist.shape[0] + test_playlist.shape[0] + test_playlist.shape[0]
    N_ITEMS = songs_encodings.shape[0]
    logging.info(f"N_USERS: {N_USERS} N_ITEMS: {N_ITEMS}")

    logging.info("Fitting datset")
    dataset = Dataset(user_identity_features=False, item_identity_features=False)
    dataset.fit(users=np.arange(0, N_USERS + 1), items=np.arange(0, N_ITEMS + 1))

    train_interactions = create_interactions_tuples(train_playlist, 0)
    train_interactions, _ = dataset.build_interactions(train_interactions)

    val_interactions = create_interactions_tuples(val_playlist, train_playlist.shape[0])
    val_interactions, _ = dataset.build_interactions(val_interactions)   
    
    test_interactions = create_interactions_tuples(test_playlist, train_playlist.shape[0] + test_playlist.shape[0])
    test_interactions, _ = dataset.build_interactions(test_interactions)

    logging.info("Saving interactions")
    save_npz(os.path.join(output_filepath, 'train_interactions.npz'), train_interactions)
    save_npz(os.path.join(output_filepath, 'val_interactions.npz'), val_interactions)
    save_npz(os.path.join(output_filepath, 'test_interactions.npz'), test_interactions)

    logging.info("Saving dataset")
    save_pickle(dataset, os.path.join(output_filepath, "dataset_lightfm"))






if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
