import logging
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import click
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from src.data.make_dataset import load_songs_encodings
from src.models.candidate_generation.generate_candidates_pipeline import \
    candidate_generation_component
from tqdm import tqdm

sys.path.append("....")


def calculate_map_at_k(ground_truths: List[List[str]], predictions: List[List[str]]) -> float:
    precisions_at_k = []
    for ground_truth, prediction in zip(ground_truths, predictions):
        k = len(set(prediction))
        tp = len(set(ground_truth) & set(prediction))
        precisions_at_k.append(tp/k)
    return np.mean(np.array(precisions_at_k))


def calculate_mar_at_k(ground_truths: List[List[str]], predictions: List[List[str]]) -> float:
    recall_at_k = []
    for ground_truth, prediction in zip(ground_truths, predictions):
        k = len(set(ground_truth))
        tp = len(set(ground_truth) & set(prediction))
        recall_at_k.append(tp/k)
    return np.mean(np.array(recall_at_k))


def coverage(all_tracks: Set[str], predictions: List[List[str]]) -> float:
    preds = [track for prediction in predictions for track in prediction]
    unique_preds = set(preds)
    covered = all_tracks & unique_preds
    return len(covered) / len(all_tracks) * 100


def unknown_track():
    return -1


def prepare_test_playlist(test_playlist: ArrayLike, reverse_songs_encodings: Dict[int, str], train_size: Optional[float] = 0.5) -> Tuple[List[str], List[str]]:
    test_playlist = list(filter(lambda x: x != -1, test_playlist))
    test_playlist = map(
        lambda track: reverse_songs_encodings[track], test_playlist)
    test_playlist = list(filter(lambda x: x != -1, test_playlist))
    train_size = int(len(test_playlist)*0.5)
    train = test_playlist[:train_size]
    test = test_playlist[train_size:]
    return train, test


@click.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('dataset_path', type=click.Path(exists=True))
@click.argument('encodings_path', type=click.Path(exists=True))
@click.argument('features_path', type=click.Path(exists=True))
@click.argument('playlists_path', type=click.Path(exists=True))
@click.argument('test_playlists_path', type=click.Path(exists=True))
@click.argument('report_path', type=click.Path(exists=True))
def main(model_path: str, dataset_path: str, encodings_path: str, features_path: str, playlists_path: str, test_playlists_path: str, report_path: str):

    N_tracks = [50, 100, 200, 300, 450, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 5000, 6000, 7000,
                8000, 9000, 10000]
    total_predictions = max(N_tracks)
    ground_truths = []
    predictions = []
    maps_at_k = []
    mars_at_k = []
    coverages = []
    songs_encodings = load_songs_encodings(encodings_path)
    reverse_songs_encodings = defaultdict(unknown_track, {
                                          track_id: track_uri for track_uri, track_id in songs_encodings.items()})
    test_playlists = pd.read_csv(test_playlists_path).to_numpy()
    all_tracks = set(list(songs_encodings.keys()))

    for test_playlist in tqdm(test_playlists, total=test_playlists.shape[0]):
        train, test = prepare_test_playlist(
            test_playlist, reverse_songs_encodings)
        _, _, _, _, tracks = candidate_generation_component(
            model_path, dataset_path, encodings_path, train, features_path, playlists_path, N=total_predictions)
        ground_truths.append(test)
        predictions.append(tracks)
        print("finished")
    for n in N_tracks:
        preds = [pred[:n] for pred in predictions]
        preds = list(map(lambda pred: pred[:n], predictions))
        maps_at_k.append(calculate_map_at_k(ground_truths, preds))
        mars_at_k.append(calculate_mar_at_k(ground_truths, preds))
        coverages.append(coverage(all_tracks, preds))
    logging.info(f"Maps_at_k{maps_at_k}")
    logging.info(f"Mars_at_k{mars_at_k}")
    logging.info(f"N_tracks {N_tracks}")
    logging.info(f"Coverage {coverages}")
    pd.DataFrame({"Maps_at_k": maps_at_k, "Mars_at_k": mars_at_k, "N_tracks": N_tracks,
                 "Coverages": coverages}).to_csv(os.path.join(report_path, "stats.csv"))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
