import logging
from datetime import datetime

import click

from src.models.diversifaction.diversify_pipeline import \
    rank_and_diversify_component

from ..models.candidate_generation.generate_candidates_pipeline import \
    candidate_generation_component


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
    relevant_playlists, tracks, features, user_playlist, recommended_tracks = candidate_generation_component(
        model_path, dataset_path, encodings_path, user_playlist, features_path, playlists_path)
    logging.info(f"Candidate generation execution time: {datetime.now() - start}")
    start = datetime.now()
    relevant_playlists, features, user_playlist, recommended_tracks = rank_and_diversify_component(features, relevant_playlists, user_playlist, 100)
    logging.info(f"Ranking and diversification execution time: {datetime.now() - start}")
    start = datetime.now()

    
    print(list(recommended_tracks))
    print(relevant_playlists.shape)
    print(user_playlist.shape)
    print(features.shape)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
