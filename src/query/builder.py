import numpy as np

from src.models.embeddings import EmbeddingModel, FilmIndex


class QueryBuilder:
    def __init__(self, embedder: EmbeddingModel, film_index: FilmIndex):
        self.embedder = embedder
        self.film_index = film_index

    def build_taste_vector(
        self, rated_films: dict[int, float], dislike_weight: float = 0.5
    ) -> np.ndarray | None:
        """Weighted average of film vectors, signed by rating minus the mean: likes pull toward,
        dislikes push away scaled by dislike_weight since dislikes are the noisier signal.

        rated_films: {tmdb_id: rating}
        """
        if not rated_films:
            return None
        mean_rating = sum(rated_films.values()) / len(rated_films)

        vecs, weights = [], []
        for tmdb_id, rating in rated_films.items():
            delta = rating - mean_rating
            if delta == 0.0:
                continue
            vec = self.film_index.get_vector(tmdb_id)
            if vec is None:
                continue
            vecs.append(vec)
            weights.append(delta if delta > 0 else delta * dislike_weight)

        if not vecs:
            return None
        return self._weighted_centroid(np.stack(vecs), weights)

    def build_taste_profiles(
        self,
        rated_films: dict[int, float],
        n_profiles: int = 4,
        min_per_profile: int = 3,
    ) -> list[np.ndarray]:
        """Clusters liked films into taste centroids instead of one average, since a single
        centroid over multi-modal taste collapses into a generic mushy vector.

        Returns one unit vector per discovered profile; falls back to a single taste vector
        when there aren't enough liked films to cluster.
        """
        if not rated_films:
            return []
        mean_rating = sum(rated_films.values()) / len(rated_films)

        vecs, weights = [], []
        for tmdb_id, rating in rated_films.items():
            delta = rating - mean_rating
            if delta <= 0.0:
                continue
            vec = self.film_index.get_vector(tmdb_id)
            if vec is not None:
                vecs.append(vec)
                weights.append(delta)

        if len(vecs) < min_per_profile * 2:
            single = self.build_taste_vector(rated_films)
            return [single] if single is not None else []

        # len(vecs) >= min_per_profile * 2 here, so k >= 2: no single-cluster special case.
        V = np.stack(vecs)
        w = np.array(weights, dtype=np.float64)
        k = min(n_profiles, len(vecs) // min_per_profile)

        from sklearn.cluster import KMeans

        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(V, sample_weight=w)
        profiles: list[np.ndarray] = []
        for c in range(k):
            mask = labels == c
            centroid = self._weighted_centroid(V[mask], w[mask]) if mask.any() else None
            if centroid is not None:
                profiles.append(centroid)
        return profiles

    def build_query_vector(
        self,
        taste_vector: np.ndarray | None,
        reference_films: list[tuple[int, float]] | None = None,  # [(tmdb_id, weight)]
        mood_text: str | None = None,
        beta: float = 0.5,
        gamma: float = 0.6,
    ) -> np.ndarray:
        """
        beta:  0 = pure taste, 1 = pure intent
        gamma: 0 = pure mood text, 1 = pure reference films
        """
        intent_vector = self._build_intent_vector(reference_films, mood_text, gamma)

        if intent_vector is None and taste_vector is None:
            raise ValueError("Need at least one of: reference films, mood text, or rated films.")

        if intent_vector is None:
            assert taste_vector is not None
            return taste_vector
        if taste_vector is None:
            return intent_vector

        query = beta * intent_vector + (1 - beta) * taste_vector
        return self._normalize(query)

    def _build_intent_vector(
        self,
        reference_films: list[tuple[int, float]] | None,
        mood_text: str | None,
        gamma: float,
    ) -> np.ndarray | None:
        ref_vector = self._build_reference_vector(reference_films)
        mood_vector = self.embedder.encode(mood_text)[0] if mood_text else None

        if ref_vector is not None and mood_vector is not None:
            intent = gamma * ref_vector + (1 - gamma) * mood_vector
            return self._normalize(intent)
        if ref_vector is not None:
            return ref_vector
        if mood_vector is not None:
            return mood_vector
        return None

    def _build_reference_vector(
        self, reference_films: list[tuple[int, float]] | None
    ) -> np.ndarray | None:
        if not reference_films:
            return None
        vecs, weights = [], []
        for tmdb_id, weight in reference_films:
            vec = self.film_index.get_vector(tmdb_id)
            if vec is not None:
                vecs.append(vec)
                weights.append(weight)
        if not vecs:
            return None
        return self._weighted_centroid(np.stack(vecs), weights)

    @staticmethod
    def _weighted_centroid(vectors: np.ndarray, weights) -> np.ndarray | None:
        """Normalized weighted mean of `vectors` by `weights`: divides by the L1 norm so it
        works whether weights are all positive or signed. Returns None when weights sum to
        zero magnitude.
        """
        w = np.asarray(weights, dtype=np.float64)
        denom = float(np.abs(w).sum())
        if denom == 0.0:
            return None
        centroid = (w[:, None] * vectors).sum(axis=0) / denom
        return QueryBuilder._normalize(centroid)

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm == 0:
            return v.astype(np.float32)
        return (v / norm).astype(np.float32)
