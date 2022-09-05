from tqdm import tqdm
from typing import Dict, Optional, Set, List, Tuple, Any
import pandas as pd
from scipy.sparse import lil_matrix, save_npz
import logging
import numpy as np
from dotenv import find_dotenv, load_dotenv
import click
import logging
import os
import pickle
from sklearn.decomposition import TruncatedSVD
import time
from ..data.make_dataset import load_songs_encodings, load_songs_features

MAX_PLAYLIST_LEN = 500


def embed_playlists(playlists_df: pd.DataFrame, total_features=2500000) -> lil_matrix:
    playlists_csr = lil_matrix((playlists_df.shape[0], total_features), dtype=np.int32)
    for index, row in tqdm(playlists_df.iterrows(), total=playlists_df.shape[0]):
        for track in row:
            if track == 0:
                continue
            playlists_csr[index, [track - 1]] = 1
    return playlists_csr


def save_compressed(content: Any, output_path: str):
    with open(output_path, "wb") as f:
        pickle.dump(content, f)


def load_compressed(input_path) -> lil_matrix:
    with open(input_path, "rb") as f:
        return pickle.load(f)


def reduce_dimensionality(sparse_matrix: lil_matrix, n_features: int = 100, n_iterations: Optional[int] = 5) -> Tuple[
    TruncatedSVD, lil_matrix]:
    svd = TruncatedSVD(n_components=n_features, n_iter=n_iterations)
    truncated_matrix = svd.fit_transform(sparse_matrix)
    return svd, truncated_matrix


def convert_to_collaborative(history: List[str], songs_encodings: Dict[str, int],
                             total_features: Optional[int] = 2500000) -> lil_matrix:
    history = [songs_encodings[track] for track in history if track in songs_encodings]
    playlist = lil_matrix((1, total_features), dtype=np.int32)
    for track in history:
        playlist[0, [track]] = 1
    return playlist


def truncate_playlist(playlist: lil_matrix, svd_model: TruncatedSVD) -> lil_matrix:
    return svd_model.transform(playlist)


def convert_to_collaborative_truncated(history: List[str], songs_encodings: Dict[str, int], svd_model: TruncatedSVD,
                                       total_features: Optional[int] = 2500000):
    collaborative = convert_to_collaborative(history, songs_encodings, total_features)
    truncated_collaborative = truncate_playlist(collaborative, svd_model)
    return truncated_collaborative


def convert_to_songs_features(history: List[str], songs_features: Dict[str, List[float]]):
    history_features = [songs_features[uri] for uri in history if uri in songs_features]
    return history_features


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('model_path', type=click.Path())
def main(input_filepath, output_filepath, model_path):
    playlists_path = os.path.join(input_filepath, "train_playlists.csv")
    logger = logging.getLogger(__name__)

    logger.info(f"loading playlists from {playlists_path}")
    playlists_df = pd.read_csv(playlists_path)

    logger.info(f"embedding playlists")
    compressed_playlists = embed_playlists(playlists_df)

    compressed_path = os.path.join(output_filepath, "train_playlists_compressed.npz")
    logger.info(f"saving compressed to {compressed_path}")
    save_compressed(compressed_playlists, compressed_path)

    start = time.time()
    logger.info("reducing dimensionality")
    svd_model, truncated_playlists = reduce_dimensionality(compressed_playlists, 100, 10)
    logger.info(f"successfully reduced in {time.time() - start} seconds")

    svd_model_path = os.path.join(model_path, "svd_model.pkl")
    logger.info(f"Saving svd_model to {svd_model_path}")
    save_compressed(svd_model, svd_model_path)

    truncated_path = os.path.join(output_filepath, "train_truncated_playlists.npz")
    logger.info(f"Saving truncated playlists to {truncated_path}")
    save_compressed(truncated_playlists, truncated_path)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())

    main()
