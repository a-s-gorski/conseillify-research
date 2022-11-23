import logging
import os
import sys
from collections import defaultdict
from typing import List, Set

import click
import numpy as np
import pandas as pd
from src.data.make_dataset import load_songs_encodings
from src.models.candidate_generation.generate_candidates_pipeline import \
    candidate_generation_component
from src.models.diversifaction.diversify_pipeline import \
    rank_and_diversify_component
from tqdm import tqdm

from ..candidate_generation.test_candidate_generation import (
    calculate_map_at_k, calculate_mar_at_k, coverage, prepare_test_playlist,
    unknown_track)

sys.path.append("....")


@click.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('dataset_path', type=click.Path(exists=True))
@click.argument('encodings_path', type=click.Path(exists=True))
@click.argument('features_path', type=click.Path(exists=True))
@click.argument('playlists_path', type=click.Path(exists=True))
@click.argument('test_playlists_path', type=click.Path(exists=True))
@click.argument('report_path', type=click.Path(exists=True))
def main(model_path: str, dataset_path: str, encodings_path: str, features_path: str, playlists_path: str, test_playlists_path: str, report_path: str):
    N_candidates = 2500
    N_tracks = [10, 20, 30, 50, 70, 100, 150, 200, 300, 450, 500, 600,
                750, 800, 1000, 1250, 1500, 1750, 2000, 2100, 2200, 2300, 2400, 2500]
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

    for test_playlist in tqdm(test_playlists[:2, :], total=test_playlists[:2, :].shape[0]):
        train, test = prepare_test_playlist(
            test_playlist, reverse_songs_encodings)
        relevant_playlists, tracks, features, user_playlist, recommended_tracks = candidate_generation_component(
            model_path, dataset_path, encodings_path, train, features_path, playlists_path, N=N_candidates)
        relevant_playlists, features, user_playlist, recommended_tracks = rank_and_diversify_component(
            features, relevant_playlists, user_playlist, ranking_n=total_predictions, diversify_n=int(total_predictions//10), diversify_epochs=1000)
        print(f"recommendation_len {len(recommended_tracks)}")
        ground_truths.append(test)
        predictions.append(recommended_tracks)
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
