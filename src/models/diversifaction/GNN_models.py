import itertools
from builtins import len
from itertools import combinations
from typing import Any, Optional, Tuple, Union

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLHeteroGraph
from dgl.nn import SAGEConv
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat.float())
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
import dgl.function as fn


class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return g.edata['score'][:, 0]
class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']

def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)

def train(model: GraphSAGE, predictor: Union[MLPPredictor, DotPredictor], graph: DGLHeteroGraph, pos_graph: DGLHeteroGraph, neg_graph: DGLHeteroGraph, lr: Optional[float]=0.01, epochs: Optional[int]=100, verbose: Optional[bool]=False) -> Tuple[GraphSAGE, Union[MLPPredictor, DotPredictor], torch.Tensor]:
    optimizer = torch.optim.Adadelta(itertools.chain(model.parameters(), predictor.parameters()), lr=lr)
    for epoch in range(epochs):
        h = model(graph, graph.ndata['feat'])
        pos_score = predictor(pos_graph, h)
        neg_score = predictor(neg_graph, h)
        loss = compute_loss(pos_score, neg_score)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0 and verbose:
            print('In epoch {}, loss: {}'.format(epoch, loss))
    return model, predictor, h

def predict(candidates: NDArray, features: torch.tensor, predictor: Union[MLPPredictor, DotPredictor], hidden_dim: torch.Tensor, n: Optional[int]=100) -> ArrayLike:
    candidates_graph = dgl.graph((candidates[:, 0].flatten(), candidates[:, 1].flatten()), num_nodes=features.shape[0])
    candidates_graph.ndata['feat']=features
    candidates_preds = predictor(candidates_graph, hidden_dim)
    predictions = candidates_preds.detach().numpy()
    predictions = [(score, index) for index, score in enumerate(predictions)]
    predictions.sort(reverse=False)
    recommended_tracks = np.array(predictions)[:n,1].flatten()
    return recommended_tracks