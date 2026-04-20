from dataclasses import dataclass

import numpy as np

from src.models.collaborative import CollaborativeModel
from src.models.embeddings import FilmIndex


@dataclass
class Recommendation:
    tmdb_id: int
    cf_score: float | None   # None when film is not in MovieLens
    similarity: float        # FAISS cosine similarity


class HybridRanker:
    def __init__(self, film_index: FilmIndex, cf_model: CollaborativeModel, tmdb_to_ml: dict[int, int]):
        self.film_index = film_index
        self.cf_model = cf_model
        self.tmdb_to_ml = tmdb_to_ml

    def recommend(
        self,
        query_vector: np.ndarray,
        user_vector: np.ndarray,
        watched_tmdb_ids: set[int],
        top_n: int = 10,
    ) -> list[Recommendation]:
        # Stage 1: FAISS retrieval — fetch enough to absorb watched films
        k = 500 + len(watched_tmdb_ids)
        candidates = self.film_index.search(query_vector, k=k)

        # Filter watched films before shortlisting
        candidates = [
            (tmdb_id, sim) for tmdb_id, sim in candidates
            if tmdb_id not in watched_tmdb_ids
        ][:500]

        if not candidates:
            return []

        sim_map: dict[int, float] = {tmdb_id: sim for tmdb_id, sim in candidates}

        # If user_vector is all zeros there is no CF signal — rank purely by FAISS similarity
        if not np.any(user_vector):
            return [
                Recommendation(tmdb_id=tmdb_id, cf_score=None, similarity=sim)
                for tmdb_id, sim in candidates
            ][:top_n]

        # Stage 2: CF reranking for films that exist in MovieLens
        ml_ids = [self.tmdb_to_ml[tmdb_id] for tmdb_id, _ in candidates if tmdb_id in self.tmdb_to_ml]
        cf_scores = self.cf_model.score_films(user_vector, ml_ids)
        ml_to_tmdb = {self.tmdb_to_ml[tmdb_id]: tmdb_id for tmdb_id, _ in candidates if tmdb_id in self.tmdb_to_ml}

        cf_results: list[Recommendation] = []
        for ml_id, cf_score in cf_scores.items():
            tmdb_id = ml_to_tmdb[ml_id]
            cf_results.append(Recommendation(tmdb_id=tmdb_id, cf_score=cf_score, similarity=sim_map[tmdb_id]))
        cf_results.sort(key=lambda r: r.cf_score, reverse=True)  # type: ignore[arg-type]

        # Films not in MovieLens: sorted by FAISS similarity, appended after CF results
        cf_tmdb_ids = {r.tmdb_id for r in cf_results}
        fallback: list[Recommendation] = [
            Recommendation(tmdb_id=tmdb_id, cf_score=None, similarity=sim)
            for tmdb_id, sim in candidates
            if tmdb_id not in cf_tmdb_ids
        ]

        return (cf_results + fallback)[:top_n]
