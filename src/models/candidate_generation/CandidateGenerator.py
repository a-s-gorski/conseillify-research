import random
from enum import Enum
from typing import Optional, List, Dict
import typing
from scipy.spatial.distance import cosine
from scipy.sparse import lil_matrix, csr_matrix
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict
from numpy.linalg import norm
from scipy.stats import pearsonr
import numpy as np
from random import sample
from numpy.typing import NDArray, ArrayLike
# from ...data.make_dataset import load_songs_features
from ...features.build_features import convert_to_collaborative_truncated
import logging


class Similarity(Enum):
    COSINE = 1
    EUCLID = 2
    PEARSON = 3


# Collaborative filtering

class CollaborativeGenerator:
    def __init__(self, playlists: pd.DataFrame, truncated_playlists, svd_model, songs_encodings: Dict[str, int],
                 similarity: Optional[Similarity] = Similarity.COSINE):
        self.playlists = playlists
        self.truncated_playlists = csr_matrix(truncated_playlists)
        self.svd_model = svd_model
        self.songs_encodings = songs_encodings
        self.logger = logging.getLogger(__name__)
        self.lookup_encodings = {track_uri: track_id for track_id, track_uri in self.songs_encodings.items()}
        self.similarity = similarity

    def similarity_calc(self, x1, x2):
        if self.similarity == Similarity.COSINE:
            return cosine(x1, x2)
        elif self.similarity == Similarity.EUCLID:
            return norm(np.array(x1) - np.array(x2))
        elif self.similarity == Similarity.PEARSON:
            return pearsonr(np.array(x1), np.array(x2[0]))[1]

    def predict(self, history: List[str], n_songs: int) -> List[str]:
        logging.info("truncating playlists")
        truncated_playlist = convert_to_collaborative_truncated(history, self.songs_encodings, self.svd_model)
        ranked_playlists = []
        self.logger.info("Ranking")
        for index, t_playlist in tqdm(enumerate(self.truncated_playlists), total=self.truncated_playlists.shape[0]):
            ranked_playlists.append((self.similarity_calc(t_playlist.toarray(), truncated_playlist), index))
        ranked_playlists.sort(reverse=True)
        ranked_playlists = [p[1] for p in ranked_playlists]
        generated_tracks = list(self.generate_n_tracks(ranked_playlists, n_songs).keys())
        generated_tracks = [self.lookup_encodings[track_id] for track_id in generated_tracks]
        return generated_tracks

    def extract_tracks(self, playlist_id):
        playlist = self.playlists.iloc[playlist_id, :].to_list()
        playlist = [track for track in playlist if track != 0]
        return playlist

    def generate_n_tracks(self, ranked_playlists: List[int], n_songs: int) -> typing.OrderedDict[str, None]:
        selected_tracks = OrderedDict()
        for playlist_id in tqdm(ranked_playlists):
            tracks = self.extract_tracks(playlist_id)
            for track in tracks:
                if track not in selected_tracks:
                    selected_tracks[track] = None
                if len(selected_tracks) == n_songs:
                    return selected_tracks
        return selected_tracks


class ContentGenerator:
    def __init__(self, songs_features: Dict[str, List[float]], similarity: Optional[Similarity] = Similarity.COSINE):
        self.songs_features = songs_features
        self.track_uris = np.array(list(songs_features.keys()))
        self.compressed_features = csr_matrix(np.array([np.array(f) for f in songs_features.values()]))
        self.similarity = similarity

    def similarity_calc(self, x1, x2):
        if self.similarity == Similarity.COSINE:
            return cosine(x1, x2)
        elif self.similarity == Similarity.EUCLID:
            return norm(np.array(x1) - np.array(x2))
        elif self.similarity == Similarity.PEARSON:
            return pearsonr(np.array(x1), np.array(x2[0]))[1]

    def predict(self, history: List[str], n_songs: Optional[int] = 500) -> List[str]:
        selected_tracks = OrderedDict()
        track_features = [self.songs_features[track] for track in history if track in self.songs_features]
        if not track_features:
            return np.random.choice(self.track_uris, n_songs, replace=False)
        ranked_features = [
            (
                self.get_max_similarity(track_features, candidate_feature.toarray()),
                track_uri
            )
            for track_uri, candidate_feature
            in tqdm(zip(self.track_uris, self.compressed_features), total=self.compressed_features.shape[0])
        ]
        ranked_features.sort(reverse=True)
        ranked_features = [f[1] for f in ranked_features]
        return ranked_features[:n_songs]

    def get_max_similarity(self, track_features, candidate_feature):
        return max([self.similarity_calc(track_feature, candidate_feature) for track_feature in track_features])


class HybridCandidateGenerator:
    def __init__(self, playlists: pd.DataFrame, truncated_playlists, svd_model, songs_encodings: Dict[str, int],
                 songs_features: Dict[str, List[float]], similarity_collab: Optional[Similarity] = Similarity.COSINE,
                 similarity_content: Optional[Similarity] = Similarity.COSINE):
        self.collaborative_generator = CollaborativeGenerator(playlists, truncated_playlists, svd_model,
                                                              songs_encodings, similarity_collab)
        self.content_generator = ContentGenerator(songs_features, similarity_content)

    def predict(self, history: List[str], n_songs: int, ratio: Optional[float] = 0.5) -> List[str]:
        collab_size = int(n_songs * ratio)
        content_size = n_songs - collab_size
        collab_preds = []
        content_preds = []

        if collab_size > 0:
            collab_preds = self.collaborative_generator.predict(history, collab_size)
        if content_size > 0:
            content_preds = self.content_generator.predict(history, content_size)

        preds = set(collab_preds)
        content_preds = set([pred for pred in content_preds if pred not in preds])
        preds = preds.union(content_preds)
        return list(preds)
