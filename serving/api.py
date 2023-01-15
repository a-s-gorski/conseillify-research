import dill
from typing import List
import os
from fastapi import FastAPI, Query
import pickle
import sys
sys.path.append("..")
sys.setrecursionlimit(10000)
from typing import Any
from src.models.candidate_generation.generate_candidates_pipeline import candidate_generation_component
from src.models.diversifaction.diversify_pipeline import rank_and_diversify_component
from src.models.reranking.rerank_pipeline import recommendation_reranking_component
from src.data.make_dataset import load_songs_encodings
from src.models.coldstart.coldstart_birch_pipeline import recommend_birch_coldstart_component

def load_pickle(input_path) -> Any:
    with open(input_path, 'rb+') as file:
        return pickle.load(file)


def save_pickle(object: Any, output_path: str):
    with open(output_path, 'wb+') as file:
        pickle.dump(object, file)

artifacts_path = os.getenv("ARTIFACTS_PATH", ".")

# base engine artifacts
model_path = os.path.join(artifacts_path, "candidate_generator.pkl")
dataset_path = os.path.join(artifacts_path, "dataset_lightfm")
encodings_path = os.path.join(artifacts_path, "songs_encodings.csv")
features_path = os.path.join(artifacts_path, "songs_features.csv")
playlists_path = os.path.join(artifacts_path, "playlists.csv")
uris_dict_filepath = os.path.join(artifacts_path, "uris_dict.pkl")
desired_distribution_filepath = os.path.join(artifacts_path, "desired_distribution.pkl")
# coldstart
embeddings_dict_path = os.path.join(artifacts_path, "embeddings_dict.pkl")
pca_path = os.path.join(artifacts_path, "pca.pkl")
brc_path = os.path.join(artifacts_path, "brc.pkl")
clusterd_tracks_path = os.path.join(artifacts_path, "clustered_tracks.pkl")


songs_encodings = load_songs_encodings(encodings_path)
datset_uris = set(list(songs_encodings.keys()))

app = FastAPI()

# main architecture - collab -> hybrid -> gnn -> reranking || coldstart
@app.get("/reranked/history")
def recommend(history: List[str] = Query(None),  playlist_name: str = "", n_recommendation: int = 10):
    n_recommendation = min(n_recommendation, 100)
    n_recommendation = max(n_recommendation, 10)
    if not (set(history) & datset_uris):
        coldstart_rec = recommend_birch_coldstart_component(playlist_name, embeddings_dict_path, pca_path, brc_path, clusterd_tracks_path, n_recommendation)
        print("finished coldstart")
        return {"recommendations": coldstart_rec, "coldstart": True}
    relevant_playlists, tracks, features, user_playlist, recommended_tracks = candidate_generation_component(
        model_path, dataset_path, encodings_path, history, features_path, playlists_path, N=200)    
    print("candidate_generation")
    relevant_playlists, features, user_playlist, recommended_tracks = rank_and_diversify_component(
        features, relevant_playlists, user_playlist, 10*n_recommendation)
    reranked_rec = recommendation_reranking_component(uris_dict_filepath, desired_distribution_filepath, recommended_tracks, n_recommendation)
    print("finished")
    return {"recommendations" : reranked_rec, "coldstart": False}

# pure coldstart model
@app.get("/coldstart/history")
def recommend(history: List[str] = Query(None),  playlist_name: str = "", n_recommendation: int = 10):
    n_recommendation = min(n_recommendation, 100)
    n_recommendation = max(n_recommendation, 10)
    coldstart_rec = recommend_birch_coldstart_component(playlist_name, embeddings_dict_path, pca_path, brc_path, clusterd_tracks_path, n_recommendation)
    if not (set(history) & datset_uris):
        return {"recommendations": coldstart_rec, "coldstart": True}
    else:
        return {"recommendations": coldstart_rec, "coldstart": False}

# pure collaborative model || coldstart
@app.get("/collaborative/history")
def recommend(history: List[str] = Query(None),  playlist_name: str = "", n_recommendation: int = 10):
    n_recommendation = min(n_recommendation, 100)
    n_recommendation = max(n_recommendation, 10)
    coldstart_rec = recommend_birch_coldstart_component(playlist_name, embeddings_dict_path, pca_path, brc_path, clusterd_tracks_path, n_recommendation)
    if not (set(history) & datset_uris):
        coldstart_rec = recommend_birch_coldstart_component(playlist_name, embeddings_dict_path, pca_path, brc_path, clusterd_tracks_path, n_recommendation)
        return {"recommendations": coldstart_rec, "coldstart": True}
    relevant_playlists, tracks, features, user_playlist, recommended_tracks = candidate_generation_component(
        model_path, dataset_path, encodings_path, history, features_path, playlists_path, N=n_recommendation)
    return {"recommendations": recommended_tracks, "coldstart": False}

# model without reranking
@app.get("/ranked/history")
def recommend(history: List[str] = Query(None),  playlist_name: str = "", n_recommendation: int = 10):
    n_recommendation = min(n_recommendation, 100)
    n_recommendation = max(n_recommendation, 10)
    if not (set(history) & datset_uris):
        coldstart_rec = recommend_birch_coldstart_component(playlist_name, embeddings_dict_path, pca_path, brc_path, clusterd_tracks_path, n_recommendation)
        return {"recommendations": coldstart_rec, "coldstart": True}
    relevant_playlists, tracks, features, user_playlist, recommended_tracks = candidate_generation_component(
        model_path, dataset_path, encodings_path, history, features_path, playlists_path, N=200)    
    
    relevant_playlists, features, user_playlist, recommended_tracks = rank_and_diversify_component(
        features, relevant_playlists, user_playlist, 10*n_recommendation)
    return {"recommendations" : recommended_tracks, "coldstart": False}