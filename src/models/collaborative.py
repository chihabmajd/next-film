from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.utils.extmath import randomized_svd

MODEL_DIR = Path(__file__).parent.parent.parent / "data" / "index"
MODEL_PATH = MODEL_DIR / "svd_model.pkl"

N_FACTORS = 100


class CollaborativeModel:
    def __init__(self) -> None:
        self.item_factors: np.ndarray | None = None   # (n_items, k)
        self.movie_raw_ids: list[str] = []            # idx → raw movieId string
        self.movie_to_idx: dict[str, int] = {}        # raw movieId → idx
        self.global_mean: float = 0.0
        self.movie_rating_counts: dict[str, int] = {} # raw movieId → number of ratings
        self.median_confidence: float = 1.0           # median log1p(count), used to normalize

    def train(self, ratings_df: pd.DataFrame) -> None:
        # ratings_df: columns [userId, movieId, rating]
        movie_ids = ratings_df["movieId"].astype(str).unique()
        user_ids = ratings_df["userId"].unique()

        self.movie_raw_ids = movie_ids.tolist()
        self.movie_to_idx = {m: i for i, m in enumerate(self.movie_raw_ids)}
        user_to_idx: dict[int, int] = {u: i for i, u in enumerate(user_ids)}

        self.global_mean = float(ratings_df["rating"].mean())

        # Confidence weighting: popular films get amplified signal in the matrix.
        # log(1 + count) is the standard choice — grows fast at first, flattens out.
        # Normalized by the median so a median-popularity film has confidence=1.
        count_map = ratings_df.groupby("movieId")["rating"].count()
        self.movie_rating_counts = {str(k): int(v) for k, v in count_map.items()}
        counts_per_row = ratings_df["movieId"].astype(str).map(self.movie_rating_counts).to_numpy(dtype=np.float32)
        confidence = np.log1p(counts_per_row)
        self.median_confidence = float(np.median(confidence)) or 1.0
        confidence /= self.median_confidence

        rows: np.ndarray = ratings_df["userId"].map(user_to_idx).to_numpy(dtype=np.int32)
        cols: np.ndarray = ratings_df["movieId"].astype(str).map(self.movie_to_idx).to_numpy(dtype=np.int32)
        data: np.ndarray = (ratings_df["rating"].to_numpy(dtype=np.float32) - np.float32(self.global_mean)) * confidence.astype(np.float32)

        R = csr_matrix(
            (data, (rows, cols)),
            shape=(len(user_ids), len(movie_ids)),
            dtype=np.float32,
        )

        # Randomized truncated SVD: R ≈ U * diag(S) * Vt
        _, S, Vt = randomized_svd(R, n_components=N_FACTORS, random_state=42)

        # Item factors: (n_items, k)
        self.item_factors = (Vt.T * S).astype(np.float32)

    def save(self) -> None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({
                "item_factors": self.item_factors,
                "movie_raw_ids": self.movie_raw_ids,
                "movie_to_idx": self.movie_to_idx,
                "global_mean": self.global_mean,
                "movie_rating_counts": self.movie_rating_counts,
                "median_confidence": self.median_confidence,
            }, f)

    def load(self) -> None:
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        self.item_factors = data["item_factors"]
        self.movie_raw_ids = data["movie_raw_ids"]
        self.movie_to_idx = data["movie_to_idx"]
        self.global_mean = data["global_mean"]
        self.movie_rating_counts = data.get("movie_rating_counts", {})
        self.median_confidence = data.get("median_confidence", 1.0)

    def is_trained(self) -> bool:
        return MODEL_PATH.exists()

    def fold_in_user(self, user_ratings: dict[int, float]) -> np.ndarray:
        """Project a new user into the SVD latent space via confidence-weighted fold-in.

        user_ratings: {movielens_movie_id (int): rating}
        Returns a k-dimensional user vector.
        """
        assert self.item_factors is not None, "Model not loaded"
        k = N_FACTORS
        rows, r_vec = [], []
        for ml_id, rating in user_ratings.items():
            idx = self.movie_to_idx.get(str(ml_id))
            if idx is not None:
                count = self.movie_rating_counts.get(str(ml_id), 1)
                conf = np.log1p(count) / self.median_confidence
                rows.append(self.item_factors[idx] * conf)
                r_vec.append((rating - self.global_mean) * conf)

        if not rows:
            return np.zeros(k, dtype=np.float32)

        V = np.array(rows, dtype=np.float64)   # (n, k)
        r = np.array(r_vec, dtype=np.float64)  # (n,)

        # Solve confidence-weighted least squares: (V^T W^2 V + λI) u = V^T W^2 r
        VtV = V.T @ V + np.eye(k) * 0.1
        u = np.linalg.solve(VtV, V.T @ r)
        return u.astype(np.float32)

    def score_films(self, user_vector: np.ndarray, ml_movie_ids: list[int]) -> dict[int, float]:
        """Score a list of MovieLens movie IDs for a given user vector."""
        assert self.item_factors is not None, "Model not loaded"
        scores = {}
        for ml_id in ml_movie_ids:
            idx = self.movie_to_idx.get(str(ml_id))
            if idx is not None:
                scores[ml_id] = float(np.dot(user_vector, self.item_factors[idx]))
        return scores
