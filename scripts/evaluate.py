"""Offline evaluation: does the recommender rank the films you actually liked near the top?

Leave-N-out protocol. From your Letterboxd likes we repeatedly hide a random handful, build
taste from everything else, rank all unwatched candidates, and check where the hidden films
land. This turns "these picks feel mediocre" into numbers, so ranking changes can be compared
instead of eyeballed.

    python scripts/evaluate.py                       # compare a preset sweep on the current index
    python scripts/evaluate.py --reps 12 --holdout 15
    python scripts/evaluate.py --index-dir data/index/backup-YYYYMMDD-HHMMSS   # A/B a different index

Metrics (higher is better), averaged over reps × held-out films:
  recall@K — fraction of held-out likes that appear in the top K
  MRR      — mean reciprocal rank (1/position of each held-out like; 0 if it never appears)

Notes:
  * Pure-taste evaluation (no mood text), so it needs no embedding model loaded — it reads
    vectors straight from the index. Fast.
  * MMR diversity is forced off here: we're measuring relevance ranking, and diversity
    deliberately trades relevance for variety, which would muddy the metric.
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.collaborative import CollaborativeModel
from src.models.embeddings import FilmIndex
from src.models.hybrid import HybridRanker
from src.query.builder import QueryBuilder

console = Console()
ROOT = Path(__file__).parent.parent
INDEX_DIR = ROOT / "data" / "index"
KS = (10, 20, 50)


def load_rated() -> dict[int, float]:
    """Your Letterboxd ratings, mapped to TMDB ids via the cached resolution."""
    resolved = json.loads((INDEX_DIR / "lb_resolved.json").read_text())
    rated: dict[int, float] = {}
    with open(ROOT / "letterboxd-export" / "ratings.csv") as f:
        for row in csv.DictReader(f):
            tid = resolved.get(f"{row['Name']}|{row['Year']}")
            if tid and tid > 0 and row["Rating"]:
                rated[int(tid)] = float(row["Rating"])
    return rated


def preset_configs() -> dict[str, dict]:
    base = {"content": 1.0, "cf": 0.6, "popularity": 0.15}
    # cf_retrieval = size of the collaborative retrieval channel (0 disables it).
    return {
        "OLD (no cf-retrieval)":       {"weights": {"content": 1.0, "cf": 0.6, "popularity": 0.3}, "profiles": True, "cf_retrieval": 0},
        "new default (+cf-retrieval)": {"weights": base, "profiles": True, "cf_retrieval": 1000},
        "cf-retrieval 2000":           {"weights": base, "profiles": True, "cf_retrieval": 2000},
        "cf-retrieval only (content0)":{"weights": {"content": 0.0, "cf": 1.0, "popularity": 0.15}, "profiles": True, "cf_retrieval": 1000},
        "pop off":                     {"weights": {**base, "popularity": 0.0}, "profiles": True, "cf_retrieval": 1000},
        "no profiles":                 {"weights": base, "profiles": False, "cf_retrieval": 1000},
    }


def build_folds(
    fi: FilmIndex,
    cf: CollaborativeModel,
    tmdb_to_ml: dict[int, int],
    rated: dict[int, float],
    likeable: list[int],
    reps: int,
    holdout: int,
    n_profiles: int,
    seed: int,
) -> list[tuple[list[int], "np.ndarray", list, "np.ndarray"]]:
    """Precompute the hold-out folds ONCE. taste/profiles/user-vector depend only on the
    train split, not on the ranking config, so building them here avoids recomputing them
    for every preset config (they were rebuilt 6× before)."""
    qb = QueryBuilder(embedder=None, film_index=fi)  # taste needs no text encoder
    rng = random.Random(seed)
    folds = []
    for _ in range(reps):
        held = rng.sample(likeable, holdout)
        train = {t: r for t, r in rated.items() if t not in held}
        taste = qb.build_taste_vector(train)
        if taste is None:
            continue
        profiles = qb.build_taste_profiles(train, n_profiles=n_profiles)
        uv = cf.fold_in_user({tmdb_to_ml[t]: r for t, r in train.items() if t in tmdb_to_ml})
        folds.append((held, taste, profiles, uv))
    return folds


def evaluate_config(
    fi: FilmIndex,
    cf: CollaborativeModel,
    tmdb_to_ml: dict[int, int],
    all_watched: set[int],
    folds: list,
    cfg: dict,
    pool: int,
) -> dict[str, float]:
    ranker = HybridRanker(fi, cf, tmdb_to_ml)
    hits = {k: [] for k in KS}
    rr: list[float] = []
    for held, taste, profiles, uv in folds:
        recs = ranker.recommend(
            taste, uv, all_watched - set(held),  # held-out films must be eligible candidates
            taste_profiles=profiles if cfg["profiles"] else None,
            top_n=pool, weights=cfg["weights"],
            diversity=0.0, candidate_pool=pool, cf_retrieval=cfg.get("cf_retrieval", 0),
        )
        rank_of = {r.tmdb_id: i for i, r in enumerate(recs)}
        for t in held:
            pos = rank_of.get(t)
            for k in KS:
                hits[k].append(1.0 if (pos is not None and pos < k) else 0.0)
            rr.append(1.0 / (pos + 1) if pos is not None else 0.0)

    out = {f"recall@{k}": float(np.mean(hits[k])) if hits[k] else 0.0 for k in KS}
    out["MRR"] = float(np.mean(rr)) if rr else 0.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10, help="random hold-out folds to average over")
    ap.add_argument("--holdout", type=int, default=12, help="liked films hidden per fold")
    ap.add_argument("--like-threshold", type=float, default=4.0, help="min rating to count as a 'like'")
    ap.add_argument("--pool", type=int, default=2000, help="candidates ranked (also the max rank measured)")
    ap.add_argument("--n-profiles", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--index-dir", type=str, default=None, help="alternate index to evaluate (A/B)")
    args = ap.parse_args()

    index_dir = Path(args.index_dir) if args.index_dir else INDEX_DIR
    fi = FilmIndex(); fi.load(index_dir=index_dir)
    cf = CollaborativeModel(); cf.load()
    tmdb_to_ml = {int(k): v for k, v in json.loads((INDEX_DIR / "tmdb_to_ml.json").read_text()).items()}

    rated = load_rated()
    # Only hold out likes that are actually in this index — otherwise we'd be measuring
    # resolution gaps, not ranking, and it would penalize every config identically.
    likeable = [t for t, r in rated.items() if r >= args.like_threshold and fi.has(t)]
    console.print(
        f"Index: [cyan]{index_dir.name}[/cyan] (dim={fi.index.d}) | "
        f"rated={len(rated)} | likes≥{args.like_threshold} in index={len(likeable)} | "
        f"holdout={args.holdout} × {args.reps} folds"
    )
    if len(likeable) <= args.holdout:
        console.print("[red]Not enough in-index likes to hold out. Lower --holdout or --like-threshold.[/red]")
        sys.exit(1)

    metrics = [*(f"recall@{k}" for k in KS), "MRR"]
    folds = build_folds(fi, cf, tmdb_to_ml, rated, likeable,
                        args.reps, args.holdout, args.n_profiles, args.seed)
    all_watched = set(rated)

    table = Table(title="Leave-N-out evaluation", show_lines=False)
    table.add_column("config", style="cyan")
    for m in metrics:
        table.add_column(m, justify="right")

    rows = [
        (name, evaluate_config(fi, cf, tmdb_to_ml, all_watched, folds, cfg, args.pool))
        for name, cfg in preset_configs().items()
    ]
    best = {metric: max(r[1][metric] for r in rows) for metric in metrics}
    for name, m in rows:
        cells = [name]
        for metric in metrics:
            val = m[metric]
            hot = abs(val - best[metric]) < 1e-9
            cells.append(f"[bold green]{val:.3f}[/bold green]" if hot else f"{val:.3f}")
        table.add_row(*cells)
    console.print(table)


if __name__ == "__main__":
    main()
