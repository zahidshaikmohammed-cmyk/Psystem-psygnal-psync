"""Historical pattern-similarity memory.

Finds historically similar market states (by normalized feature-vector
distance) and reports what followed them over the next ~60 minutes. This
is one input to the forecast ensemble, not a standalone signal.

Leakage safety: `query(..., max_index=T)` restricts matches to patterns
recorded strictly before `T`. Internally this over-fetches neighbors and
filters by stored index rather than maintaining a fully re-fit expanding
window per query — documented as a pragmatic approximation for a
framework that has not yet been exercised against a real historical
dataset; it never returns a match at or after `max_index`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from psygnal import config


@dataclass
class PatternMemoryStore:
    feature_columns: list[str]
    scaled_matrix: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    labels: np.ndarray
    forward_returns: np.ndarray
    stored_index: np.ndarray  # positional index within the original historical series
    nn_model: NearestNeighbors

    @classmethod
    def fit(cls, X: pd.DataFrame, labels: pd.Series, forward_returns: pd.Series) -> "PatternMemoryStore":
        feature_columns = list(X.columns)
        raw = X.to_numpy(dtype=float)
        mean = np.nanmean(raw, axis=0)
        std = np.nanstd(raw, axis=0)
        std[std == 0] = 1.0
        raw_filled = np.where(np.isnan(raw), mean, raw)
        scaled = (raw_filled - mean) / std

        nn_model = NearestNeighbors(algorithm="brute", metric="euclidean")
        nn_model.fit(scaled)

        return cls(
            feature_columns=feature_columns,
            scaled_matrix=scaled,
            mean=mean,
            std=std,
            labels=labels.to_numpy(),
            forward_returns=forward_returns.to_numpy(dtype=float),
            stored_index=np.arange(len(X)),
            nn_model=nn_model,
        )

    def query(
        self,
        x_row: pd.Series,
        k: int = config.PATTERN_MEMORY_K_NEIGHBORS,
        max_index: Optional[int] = None,
    ) -> dict[str, Any]:
        n_available = len(self.stored_index) if max_index is None else int((self.stored_index < max_index).sum())
        if n_available < config.PATTERN_MEMORY_MIN_SAMPLES:
            return {"status": "INSUFFICIENT_DATA", "k_used": 0}

        raw = x_row.reindex(self.feature_columns).to_numpy(dtype=float)
        raw_filled = np.where(np.isnan(raw), self.mean, raw)
        scaled = ((raw_filled - self.mean) / self.std).reshape(1, -1)

        fetch_k = min(len(self.scaled_matrix), max(k * 5, k))
        distances, indices = self.nn_model.kneighbors(scaled, n_neighbors=fetch_k)
        distances, indices = distances[0], indices[0]

        if max_index is not None:
            keep = self.stored_index[indices] < max_index
            distances, indices = distances[keep], indices[keep]

        distances, indices = distances[:k], indices[:k]
        if len(indices) == 0:
            return {"status": "INSUFFICIENT_DATA", "k_used": 0}

        neighbor_labels = self.labels[indices]
        neighbor_returns = self.forward_returns[indices]

        total = len(neighbor_labels)
        prob_up = float((neighbor_labels == "UP").sum() / total)
        prob_down = float((neighbor_labels == "DOWN").sum() / total)
        prob_neutral = float((neighbor_labels == "NEUTRAL").sum() / total)

        return {
            "status": "OK",
            "k_used": total,
            "probability_up": prob_up,
            "probability_down": prob_down,
            "probability_neutral": prob_neutral,
            "mean_forward_return": float(np.nanmean(neighbor_returns)),
            "mean_distance": float(np.mean(distances)) if len(distances) else None,
        }
