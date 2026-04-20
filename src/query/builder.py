import numpy as np

from src.models.embeddings import EmbeddingModel, FilmIndex


class QueryBuilder:
    def __init__(self, embedder: EmbeddingModel, film_index: FilmIndex):
        self.embedder = embedder
        self.film_index = film_index

    def build_taste_vector(self, rated_films: dict[int, float]) -> np.ndarray | None:
        """Weighted average of film vectors, weighted by how much above your mean each rating is.
        rated_films: {tmdb_id: rating}
        Films rated below your personal mean contribute zero — taste vector only reflects what you liked.
        """
        if not rated_films:
            return None
        mean_rating = sum(rated_films.values()) / len(rated_films)

        vecs, weights = [], []
        for tmdb_id, rating in rated_films.items():
            weight = max(0.0, rating - mean_rating)
            if weight == 0.0:
                continue
            vec = self.film_index.get_vector(tmdb_id)
            if vec is not None:
                vecs.append(vec)
                weights.append(weight)

        if not vecs:
            return None
        V = np.stack(vecs)
        w = np.array(weights, dtype=np.float32)
        taste = (w[:, None] * V).sum(axis=0) / w.sum()
        return self._normalize(taste)

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
        V = np.stack(vecs)
        w = np.array(weights, dtype=np.float32)
        total = w.sum()
        if total == 0.0:
            return None
        w /= total
        return self._normalize((w[:, None] * V).sum(axis=0))

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return (v / norm).astype(np.float32)
