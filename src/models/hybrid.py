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


# Default blend. Content relevance leads; CF is a supporting "people like you" nudge;
# popularity is subtracted so obvious blockbusters don't dominate a cinephile's list.
DEFAULT_WEIGHTS = {"content": 1.0, "cf": 0.6, "popularity": 0.3}

# Standardized signals are clipped to ±this many standard deviations before blending. CF
# scores are heavy-tailed — without clipping, one outlier film gets a z-score of ~20 and
# swamps content relevance and the popularity penalty, re-creating the mainstream bias.
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
    ) -> list[Recommendation]:
        w = {**DEFAULT_WEIGHTS, **(weights or {})}

        if exploration > 0.0:
            query_vector = self._jitter(query_vector, exploration)

        # ---- Stage 1: candidate generation ----------------------------------------------
        # Three retrieval channels, unioned:
        #   (a) the query vector — content relevance to taste+intent,
        #   (b) each taste profile — widens recall across the facets of your taste,
        #   (c) collaborative — films people with your rating pattern love, which content
        #       retrieval structurally misses (they can sit far away in embedding space).
        # Content still governs relevance at scoring time, so off-taste CF candidates score low.
        watched = watched_tmdb_ids
        candidates: list[int] = []
        seen: set[int] = set()

        def add_ids(ids) -> None:
            for tmdb_id in ids:
                if tmdb_id in watched or tmdb_id in seen:
                    continue
                seen.add(tmdb_id)
                candidates.append(tmdb_id)

        add_ids(t for t, _ in self.film_index.search(query_vector, k=candidate_pool + len(watched)))
        for profile in taste_profiles or []:
            add_ids(t for t, _ in self.film_index.search(profile, k=candidate_pool // 3))
        if cf_retrieval > 0 and np.any(user_vector):
            add_ids(
                tmdb for ml_raw in self.cf_model.top_items(user_vector, cf_retrieval)
                if (tmdb := self.ml_to_tmdb.get(int(ml_raw))) is not None
            )

        if not candidates:
            return []

        # Content similarity of every candidate against the query vector (uniform scale).
        vecs: dict[int, np.ndarray] = {}
        content: dict[int, float] = {}
        for tmdb_id in candidates:
            vec = self.film_index.get_vector(tmdb_id)
            if vec is None:
                continue
            vecs[tmdb_id] = vec
            content[tmdb_id] = float(np.dot(query_vector, vec))
        candidates = [c for c in candidates if c in vecs]

        # ---- Stage 2: debiased collaborative-filtering signal ---------------------------
        # Raw CF scores carry a popularity offset (well-loved films score high for everyone).
        # We z-score CF *within the candidate set*, which removes that offset and turns CF into
        # a relative "more/less for you than the pool average" signal. Films outside MovieLens
        # get cf_z = None and contribute 0 to the blend — they compete on content, not buried.
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

        # ---- Stage 3: blend -------------------------------------------------------------
        # Put content on the same scale as cf_z by z-scoring it across the candidate pool.
        c_vals = np.array([content[c] for c in candidates], dtype=np.float64)
        c_mean, c_std = float(c_vals.mean()), float(c_vals.std() or 1.0)

        scored: list[Recommendation] = []
        for c in candidates:
            content_z = _clip((content[c] - c_mean) / c_std)
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
                    content_sim=content[c],
                    cf_z=czi,
                    popularity=popularity,
                )
            )

        scored.sort(key=lambda r: r.score, reverse=True)

        # ---- Stage 4: diversify (MMR) ---------------------------------------------------
        # Re-rank the top slice so we don't return five near-identical films (same director,
        # same franchise). Each pick trades its score against similarity to already-picked films.
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
        while pool and len(selected) < top_n:
            best, best_val = None, -float("inf")
            for r in pool:
                if not selected:
                    val = norm[r.tmdb_id]
                else:
                    max_sim = max(
                        float(np.dot(vecs[r.tmdb_id], vecs[s.tmdb_id])) for s in selected
                    )
                    val = norm[r.tmdb_id] - diversity * max_sim
                if val > best_val:
                    best, best_val = r, val
            assert best is not None
            selected.append(best)
            pool.remove(best)
        return selected
