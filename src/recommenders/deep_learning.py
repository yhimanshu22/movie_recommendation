# Minimal scaffold for a neural recommender (embeddings).
# Keeps deps light; uses NumPy SGD. Replace later with PyTorch/TF for production.

from __future__ import annotations
import numpy as np
from typing import Tuple

class MatrixFactorizationLite:
    """
    Simple MF: R ~ U @ V^T (with SGD)
    Inputs: triplets (user, item, rating), ids must be mapped to 0..n-1
    """

    def __init__(self, n_users: int, n_items: int, k: int = 32, lr: float = 0.01, reg: float = 0.01, epochs: int = 5, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.U = 0.1 * rng.standard_normal((n_users, k))
        self.V = 0.1 * rng.standard_normal((n_items, k))
        self.lr = lr
        self.reg = reg
        self.epochs = epochs

    def fit(self, users: np.ndarray, items: np.ndarray, ratings: np.ndarray):
        for _ in range(self.epochs):
            idx = np.arange(len(ratings))
            np.random.shuffle(idx)
            for i in idx:
                u, v, r = users[i], items[i], ratings[i]
                pu = self.U[u]; qi = self.V[v]
                pred = pu @ qi
                err = r - pred
                # updates
                self.U[u] += self.lr * (err * qi - self.reg * pu)
                self.V[v] += self.lr * (err * pu - self.reg * qi)

    def predict(self, u: int, v: int) -> float:
        return float(self.U[u] @ self.V[v])

    def recommend_for_user(self, u: int, known_items: np.ndarray, top_n: int = 10) -> list[tuple[int,float]]:
        scores = self.V @ self.U[u]
        scores[known_items] = -np.inf
        idx = np.argsort(-scores)[:top_n]
        return [(int(i), float(scores[i])) for i in idx]
