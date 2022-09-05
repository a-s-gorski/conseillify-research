# Estimate accuracy using holdout techniques
import logging
import os.path
import click
import pandas as pd
from random import sample
from typing import Tuple, List
from tqdm import tqdm
import matplotlib.pyplot as plt

from ...features.build_features import load_compressed, load_songs_encodings


def prepare_holdouts(test_dataset: List[List[str]]) -> Tuple[List[List[str]], List[List[str]]]:
    test_datasets = []
    holdouts = []
    for playlist in test_dataset:
        test_datasets.append(playlist[:-10])
        holdouts.append(playlist[-10:])
    return test_datasets, holdouts


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('model_filepath', type=click.Path())
def main(input_filepath, model_filepath):
    logger = logging.getLogger(__name__)
    songs_encodings_path = os.path.join(input_filepath, "songs_encodings.csv")
    test_playlists_path = os.path.join(input_filepath, "test_playlists.csv")
    model_path = os.path.join(model_filepath, 'candidate_generator.pkl')

    logger.info(f"loading songs_encodings and lookup from {songs_encodings_path}")
    songs_encodings = load_songs_encodings(songs_encodings_path)

    logger.info("creating lookup")
    songs_lookup = {value: uri for uri, value in songs_encodings.items()}

    logger.info(f"loading model from {model_path}")
    candidate_generator = load_compressed(model_path)

    logger.info(f"loading test_playlists from {test_playlists_path}")
    test_playlists = pd.read_csv(test_playlists_path)

    logger.info("processing test_playlists")
    test_playlists = [[songs_lookup[track] for track in row if track in songs_lookup] for _, row in
                      tqdm(test_playlists.iterrows(), total=test_playlists.shape[0])]
    print(test_playlists[:10])

    logger.info("sampling test_playlists")
    test_playlists = sample(test_playlists, 5)

    logger.info("preparing holdouts")
    test_ds, holdouts = prepare_holdouts(test_playlists)

    samples_sizes = [200, 500, 1000, 2500, 5000]
    # samples_sizes = [5, 10]
    precisions_list = []
    for s_size in tqdm(samples_sizes):
        preds = [candidate_generator.predict(ds, s_size, 1) for ds in test_ds]
        precisions = [len(set(h) & set(p)) / len(set(p)) for p, h in zip(preds, holdouts)]
        precisions_list.append(sum(precisions) / len(precisions))

    # logger.info("generate preds")
    # preds = [candidate_generator.predict(ds, 100, 1) for ds in test_ds]

    # logger.info("generating precision")
    # precisions = [len(set(h).union(set(h))) / len(set(p)) for p, h in zip(preds, holdouts)]

    logger.info("Precisions")
    print(precisions_list)

    logger.info("Plotting evaluations")
    plt.plot(samples_sizes, precisions_list)
    plt.show()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
