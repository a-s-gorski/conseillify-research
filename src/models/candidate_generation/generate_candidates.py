import logging
import os.path

import click
import pandas as pd

from ...features.build_features import load_compressed, save_compressed
from ...data.make_dataset import load_songs_encodings, load_songs_features
from .CandidateGenerator import HybridCandidateGenerator, Similarity


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('model_filepath', type=click.Path())
def main(input_filepath, model_filepath):
    logger = logging.getLogger(__name__)
    playlists_path = os.path.join(input_filepath, "train_playlists.csv")
    encodings_path = os.path.join(input_filepath, "songs_encodings.csv")
    truncated_playlists_path = os.path.join(input_filepath, "train_truncated_playlists.npz")
    svd_models_path = os.path.join(model_filepath, "svd_model.pkl")
    songs_features_path = os.path.join(input_filepath, "songs_features.csv")

    logger.info(f"loading playlists from {playlists_path}")
    playlists = pd.read_csv(playlists_path, index_col=False)

    logger.info(f"loading encodings from {encodings_path}")
    encodings = load_songs_encodings(encodings_path)

    logger.info(f"loading truncated_playlists from {truncated_playlists_path}")
    truncated_playlists = load_compressed(truncated_playlists_path)

    logger.info(f"loading svd_model {truncated_playlists_path}")
    svd_model = load_compressed(svd_models_path)

    logger.info(f"loading songs_features{songs_features_path}")
    songs_features = load_songs_features(songs_features_path)

    logger.info("creating candidate generator")

    test_tracks = ['spotify:track:3xG8Xcrke8F1gfYjzQE1Le', 'spotify:track:6EfP7rkoR1L1OpDzKN5lXX',
                   'spotify:track:3sRSpnXTVOLjjcDse6P6a0', 'spotify:track:2RKsy1MhBTcvAWIFpdUgvj',
                   'spotify:track:7g2mskmb0okRLDtx0vtobh', 'spotify:track:0wydxbEaB3KSKJstJVhTG5',
                   'spotify:track:63AYMkPl6rRn9m9NuUMrxI', 'spotify:track:1coJ1H1FiEZG93naXkOsNF',
                   'spotify:track:66s38Z7xubVergCcTx7dxs', 'spotify:track:7gLSX6HlNso7WkoWPCGNGr',
                   'spotify:track:0vw9ESz1MJJ4UGmGgGVSDb', 'spotify:track:2BZYVqGyL1L1adBbq2ClVv',
                   'spotify:track:1CrjSenHvVEOl67ZlVU5De', 'spotify:track:2ng9pJznvGiuQ86OwWZ8Qa',
                   'spotify:track:0Axyu4TO4vrPEJI6Hi7NhW', 'spotify:track:2BZYVqGyL1L1adBbq2ClVv',
                   'spotify:track:71VtqHXE00DYtK8hhPaKHn', 'spotify:track:3SgoXZ25TPqJW5QiGwD65f',
                   'spotify:track:4kFFagMTbsVWsJ120Sp7a5', 'spotify:track:6vc7GNW3FxMkgKuProVOoB',
                   'spotify:track:4py8spIzvJETEC3srLB0q4', 'spotify:track:6EfP7rkoR1L1OpDzKN5lXX',
                   'spotify:track:10qbHF920zH5K8C8IcE5AL', 'spotify:track:7ARLbcqLgOrBI2JfzfKtHD',
                   'spotify:track:1coJ1H1FiEZG93naXkOsNF', 'spotify:track:1SHB1hp6267UK9bJQUxYvO',
                   'spotify:track:0YBYp69ne0vdDNF0Tqf4F1', 'spotify:track:1WeGHBnuEgegV7hSrQkh9I',
                   'spotify:track:681Q0ngEIOMctYw2I5qd3m', 'spotify:track:6OYXQ94pvwySRmcZmtYdpu',
                   'spotify:track:681Q0ngEIOMctYw2I5qd3m', 'spotify:track:2ZQiPRPoIvbNDmzspeb7Zi',
                   'spotify:track:14BOSGH6R6FgksCvu8c9iM', 'spotify:track:68LomqjAydy23XtWRWaIM7',
                   'spotify:track:681Q0ngEIOMctYw2I5qd3m', 'spotify:track:6OYXQ94pvwySRmcZmtYdpu',
                   'spotify:track:4uS7ipqw4uwv08NAN63NaE', 'spotify:track:1TnAeT4ZZ0xtnxKNuYeazy',
                   'spotify:track:2nbkEofLoPSxPCDhgFS48X', 'spotify:track:2PjdJaw0FHUHYy97ZwZrOv',
                   'spotify:track:01j3nwwNmfKObu5eNndc2d', 'spotify:track:1KkLfm1ohpMKyUGPQJ3N2Y',
                   'spotify:track:4DP7xqmYjAkC7991IYNDne', 'spotify:track:4G21tyvLXKNWHHLXcr4B0m',
                   'spotify:track:1YM5ZcVeeCs3MzTzSNod7L', 'spotify:track:4J4cJDxMSnkEnLIEp1DI7C',
                   'spotify:track:2nA6fKXgWsrCOrNaf1Zn0m']

    hybridGenerator = HybridCandidateGenerator(playlists, truncated_playlists, svd_model, encodings, songs_features, similarity_collab=Similarity.EUCLID, similarity_content=Similarity.COSINE)
    # preds = hybridGenerator.predict(test_tracks, 10000, ratio=0.4)
    save_compressed(hybridGenerator, output_path=os.path.join(model_filepath, "candidate_generator.pkl"))



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
