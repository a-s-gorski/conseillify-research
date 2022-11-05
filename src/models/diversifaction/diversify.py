import itertools
import logging
import os
from builtins import len
from itertools import combinations, chain
from typing import Optional, Tuple, Union

import click
import dgl
import numpy as np
import pandas as pd
import torch
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

from ...features.build_features import load_pickle, save_pickle
from .GNN_models import DotPredictor, GraphSAGE, MLPPredictor, predict, train


def generate_neg_edges(pos_edges: NDArray, neg_size, max_iterations: Optional[int] = 1000000) -> NDArray:
    pos_edges_set = set([tuple(edge) for edge in pos_edges])
    neg_edges = set()
    min_edge, max_edge = pos_edges.flatten().min(
        axis=0), pos_edges.flatten().max(axis=0)
    iterations = 0
    while len(neg_edges) < neg_size and iterations < max_iterations:
        start_edge = np.random.randint(min_edge, max_edge)
        end_edge = np.random.randint(min_edge, max_edge)
        if start_edge != end_edge and (start_edge, end_edge) not in pos_edges_set:
            neg_edges.add((start_edge, end_edge))
        iterations += 1
    return np.array(list(neg_edges))


def generate_candidate_edges(pos_edges: NDArray, neg_edges: NDArray, user_playlist: ArrayLike, n_candidates: Optional[int] = 1000, max_iterations: Optional[int] = 100000):
    start_candidates = set(user_playlist)
    start_candidates.discard(-1)
    end_candidates = set([track for track in set(
        pos_edges.flatten()) if track not in start_candidates])
    end_candidates.discard(-1)
    candidates = set()
    iterations = 0

    pos_edges_set = set(pos_edges.flatten())
    neg_edges_set = set(neg_edges.flatten())

    while len(candidates) < n_candidates and iterations < max_iterations:
        candidate = (np.random.choice(list(start_candidates), size=1)[
                     0], np.random.choice(list(end_candidates), size=1)[0])
        if candidate not in candidates.union(pos_edges_set).union(neg_edges_set):
            candidates.add(candidate)
        iterations += 1
    return np.array(list(candidates))


def recommend_n_diverse(features: NDArray, playlists: NDArray, user_playlist: NDArray, n: Optional[int] = 10, sample_size: Optional[int] = 5, hid_dim: Optional[int] = 16, epochs: Optional[int] = 100) -> Tuple[ArrayLike, GraphSAGE, Union[MLPPredictor, DotPredictor], torch.Tensor]:
    features = torch.from_numpy(np.array(features, dtype=np.float64))
    print(f"features shape {features.shape}")
    logging.info("Sampling playlists")
    sampled_playlists = playlists[np.random.choice(
        playlists.shape[0], size=sample_size, replace=False), :]
    logging.info("Building edges")
    sampled_playlists = [[track for track in playlist if track != -1] for playlist in sampled_playlists]
    sampled_playlists = [row for row in sampled_playlists if len(row) > 1]
    pos_edges = [list(combinations(playlist, 2)) for playlist in sampled_playlists]
    pos_edges = list(itertools.chain(*pos_edges))
    pos_edges = np.array(pos_edges)
    print(pos_edges.shape)
    pos_edges = np.unique(pos_edges, axis=0)
    print(pos_edges.shape)
    print(f"pos_edges_shape {pos_edges.shape} min_pos_edge {min(pos_edges.flatten())} max_pos_edge {max(pos_edges.flatten())}")





    
    print(f"pos_edges_shape {pos_edges.shape}")
    neg_edges = generate_neg_edges(pos_edges, pos_edges.shape[0])
    print(f"max_neg_edge {max(neg_edges.flatten())}")
    print(f"min_neg_edge {min(neg_edges.flatten())}")
    print(f"neg_edges_shape {neg_edges.shape}")
    pos_graph = dgl.graph((pos_edges[:, 0].flatten(
    ), pos_edges[:, 1].flatten()), num_nodes=features.shape[0])
    pos_graph.ndata['feat'] = features
    neg_graph = dgl.graph((neg_edges[:, 0].flatten(
    ), neg_edges[:, 1].flatten()), num_nodes=features.shape[0])
    neg_graph.ndata['feat'] = features
    total_start_edges = np.concatenate(
        (pos_edges[:, 0], neg_edges[:, 0]), axis=0).flatten()
    total_end_edges = np.concatenate(
        (pos_edges[:, 1], neg_edges[:, 1]), axis=0).flatten()
    graph = dgl.graph((total_start_edges, total_end_edges),
                      num_nodes=features.shape[0])
    graph.ndata['feat'] = features

    model = GraphSAGE(graph.ndata['feat'].shape[1], hid_dim)
    predictor = DotPredictor()
    logging.info("Training model")
    model, predictor, hidden_dim = train(
        model, predictor, graph, pos_graph, neg_graph, epochs=20, verbose=True)
    logging.info("Generating candidates")
    candidates = generate_candidate_edges(pos_edges, neg_edges, user_playlist)
    print(candidates.shape)
    logging.info("Inference")
    predictions = predict(candidates, features, predictor, hidden_dim, n=n)
    return predictions, model, predictor, hidden_dim


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('model_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path(exists=True))
def main(input_filepath: str, model_filepath: str, output_filepath: str):
    logging.info("Loading data")
    print(os.getcwd())
    features = pd.read_csv(os.path.join(
        input_filepath, "features.csv"), index_col=False).to_numpy()[:, 2:13]
    playlists = pd.read_csv(os.path.join(
        input_filepath, "playlists.csv")).to_numpy()
    user_playlist = pd.read_csv(os.path.join(
        input_filepath, "user_playlist.csv")).to_numpy()[:, 1]
    SAMPLE_SIZE = 3
    predictions, model, predictor, hidden_dim = recommend_n_diverse(features, playlists, user_playlist,
                        100, 10, 32, epochs=20)
    pd.Series(predictions).to_csv(os.path.join(output_filepath, "recommendations.csv"))
    save_pickle(model, os.path.join(model_filepath, "gnn_model.pkl"))
    save_pickle(predictor, os.path.join(model_filepath, "gnn_predictor.pkl"))
    pd.DataFrame(hidden_dim.detach().numpy()).to_csv(os.path.join(model_filepath, "hidden_dim.csv"))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
