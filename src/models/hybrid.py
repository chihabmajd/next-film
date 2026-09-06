from dataclasses import dataclass

import numpy as np

from src.models.collaborative import CollaborativeModel
from src.models.embeddings import FilmIndex


@dataclass
class Recommendation:
    tmdb_id: int
    score: float                 # final blended score (arbitrary scale, higher = better)
    content_sim: float           # cosine similarity to the query vector, in [-1, 1]
    cf_z: float | None           # debiased CF signal (z-scored); None when film is not in MovieLens
    popularity: float            # how mainstream the film is, in [0, 1]


# Content dominates the blend; CF adds a smaller taste nudge; popularity is subtracted.
DEFAULT_WEIGHTS = {"content": 1.0, "cf": 0.6, "popularity": 0.15}

# CF z-scores are heavy-tailed; clip to ±2.5σ so one outlier doesn't swamp content and popularity.
CLIP = 2.5


def _clip(x: float) -> float:
    return max(-CLIP, min(CLIP, x))


class HybridRanker:
    def __init__(self, film_index: FilmIndex, cf_model: CollaborativeModel, tmdb_to_ml: dict[int, int]):
        self.film_index = film_index
        self.cf_model = cf_model
        self.tmdb_to_ml = tmdb_to_ml
        self.ml_to_tmdb = {ml: tmdb for tmdb, ml in tmdb_to_ml.items()}

    def recommend(
        self,
        query_vector: np.ndarray,
        user_vector: np.ndarray,
        watched_tmdb_ids: set[int],
        taste_profiles: list[np.ndarray] | None = None,
        top_n: int = 10,
        weights: dict[str, float] | None = None,
        diversity: float = 0.3,
        candidate_pool: int = 1000,
        cf_retrieval: int = 0,
        exploration: float = 0.0,
        taste_weight: float = 1.0,
    ) -> list[Recommendation]:
        """taste_weight (= 1 − β) scales the CF weight and the profile retrieval breadth."""
        w = {**DEFAULT_WEIGHTS, **(weights or {})}
        taste_weight = max(0.0, min(1.0, taste_weight))
        w["cf"] = w["cf"] * taste_weight

        if exploration > 0.0:
            query_vector = self._jitter(query_vector, exploration)

        # Stage 1: candidate generation via query vector, taste profiles, and CF retrieval.
        # Content still scores relevance at ranking time, so off-taste CF candidates rank low.
        candidates: list[int] = []
        seen: set[int] = set()

        def add_ids(ids) -> None:
            for tmdb_id in ids:
                if tmdb_id in watched_tmdb_ids or tmdb_id in seen:
                    continue
                seen.add(tmdb_id)
                candidates.append(tmdb_id)

        add_ids(t for t, _ in self.film_index.search(query_vector, k=candidate_pool + len(watched_tmdb_ids)))
        profile_k = int(candidate_pool // 3 * taste_weight)
        if profile_k > 0:
            for profile in taste_profiles or []:
                add_ids(t for t, _ in self.film_index.search(profile, k=profile_k))
        if cf_retrieval > 0 and np.any(user_vector):
            add_ids(
                tmdb for ml_raw in self.cf_model.top_items(user_vector, cf_retrieval)
                if (tmdb := self.ml_to_tmdb.get(int(ml_raw))) is not None
            )

        if not candidates:
            return []

        # One reconstruct + matmul; get_vectors drops ids absent from the index.
        V, candidates = self.film_index.get_vectors(candidates)
        if not candidates:
            return []
        content_arr = V @ query_vector  # (N,) cosine, both sides are unit-normalized
        vecs = {tid: V[i] for i, tid in enumerate(candidates)}

        # Stage 2: debiased CF signal.
        # z-score CF within the candidate set to remove the popularity offset.
        # Films outside MovieLens get None and contribute 0.
        cf_z: dict[int, float] = {}
        if np.any(user_vector):
            ml_of = {c: self.tmdb_to_ml[c] for c in candidates if c in self.tmdb_to_ml}
            raw = self.cf_model.score_films(user_vector, list(ml_of.values()))
            vals = np.array(list(raw.values()), dtype=np.float64)
            if len(vals) >= 2 and vals.std() > 0:
                mean, std = float(vals.mean()), float(vals.std())
                for tmdb_id, ml_id in ml_of.items():
                    if ml_id in raw:
                        cf_z[tmdb_id] = (raw[ml_id] - mean) / std

        # Stage 3: blend.
        # z-score content across the candidate pool to match cf_z's scale.
        c_mean = float(content_arr.mean())
        c_std = float(content_arr.std()) or 1.0

        scored: list[Recommendation] = []
        for i, c in enumerate(candidates):
            content_sim = float(content_arr[i])
            content_z = _clip((content_sim - c_mean) / c_std)
            raw_cf = cf_z.get(c)
            czi = _clip(raw_cf) if raw_cf is not None else None
            ml_id = self.tmdb_to_ml.get(c)
            popularity = self.cf_model.popularity(ml_id) if ml_id is not None else 0.0
            score = (
                w["content"] * content_z
                + w["cf"] * (czi if czi is not None else 0.0)
                - w["popularity"] * popularity
            )
            scored.append(
                Recommendation(
                    tmdb_id=c,
                    score=score,
                    content_sim=content_sim,
                    cf_z=czi,
                    popularity=popularity,
                )
            )

        scored.sort(key=lambda r: r.score, reverse=True)

        # Stage 4: MMR. Each pick trades score against similarity to those already picked.
        shortlist = scored[: max(top_n * 6, top_n)]
        return self._mmr(shortlist, vecs, diversity, top_n)

    @staticmethod
    def _jitter(query_vector: np.ndarray, exploration: float) -> np.ndarray:
        noise = np.random.randn(len(query_vector)).astype(np.float32)
        noise /= np.linalg.norm(noise) or 1.0
        q = query_vector + exploration * noise
        norm = np.linalg.norm(q)
        return (q / norm) if norm > 0 else query_vector

    @staticmethod
    def _mmr(
        recs: list[Recommendation],
        vecs: dict[int, np.ndarray],
        diversity: float,
        top_n: int,
    ) -> list[Recommendation]:
        if diversity <= 0.0 or len(recs) <= 1:
            return recs[:top_n]

        # Normalize scores to [0, 1] so the diversity trade-off is scale-stable.
        vals = [r.score for r in recs]
        lo, hi = min(vals), max(vals)
        rng = (hi - lo) or 1.0
        norm = {r.tmdb_id: (r.score - lo) / rng for r in recs}

        pool = list(recs)
        selected: list[Recommendation] = []
        # Running max similarity to the selected set, updated incrementally.
        max_sim = {r.tmdb_id: 0.0 for r in recs}
        while pool and len(selected) < top_n:
            pick = max(pool, key=lambda r: norm[r.tmdb_id] - diversity * max_sim[r.tmdb_id])
            selected.append(pick)
            pool.remove(pick)
            picked_vec = vecs[pick.tmdb_id]
            for r in pool:
                sim = float(np.dot(vecs[r.tmdb_id], picked_vec))
                if sim > max_sim[r.tmdb_id]:
                    max_sim[r.tmdb_id] = sim
        return selected
