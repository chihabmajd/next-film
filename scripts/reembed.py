"""Rebuild the FAISS content index for the films already in it — new model + new text blob.

Unlike setup.py, this does NOT touch MovieLens/Letterboxd resolution or the SVD model. It
re-embeds the exact set of films currently in data/index/films_ids.npy, reading their metadata
from the TMDB cache (no network), re-merging MovieLens tags, and rebuilding the index with the
embedding model from config.yaml. The previous index is backed up so the change is reversible.

Run after changing the embedding model or FilmMetadata.to_text_blob():

    python scripts/reembed.py
"""

import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rich.console import Console
from rich.progress import track

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.enrichment.tmdb import CACHE_DIR, FilmMetadata, TMDBClient
from src.models.embeddings import EmbeddingModel, FilmIndex

console = Console()

ROOT = Path(__file__).parent.parent
INDEX_DIR = ROOT / "data" / "index"
MOVIELENS_DIR = ROOT / "data" / "movielens"


def load_cached_metadata(tmdb_ids: list[int], client: TMDBClient) -> dict[int, FilmMetadata]:
    """Load FilmMetadata for each id straight from the on-disk TMDB cache (no HTTP)."""
    out: dict[int, FilmMetadata] = {}
    missing = 0
    for tid in track(tmdb_ids, description="Reading TMDB cache"):
        path = CACHE_DIR / f"{tid}.json"
        if path.exists():
            out[tid] = client._from_cache(path)
        else:
            missing += 1
    if missing:
        console.print(f"  [yellow]{missing:,} films not in cache — skipped.[/yellow]")
    return out


def merge_movielens_tags(metadata: dict[int, FilmMetadata]) -> None:
    """Re-apply the top MovieLens user tags to each film's keywords, as setup.py does."""
    import json

    tags_path = MOVIELENS_DIR / "tags.csv"
    ml_map_path = INDEX_DIR / "ml_to_tmdb.json"
    if not (tags_path.exists() and ml_map_path.exists()):
        console.print("[yellow]tags.csv or ml_to_tmdb.json missing — skipping tag merge.[/yellow]")
        return

    ml_to_tmdb = {int(k): int(v) for k, v in json.loads(ml_map_path.read_text()).items()}
    tags_df = pd.read_csv(tags_path, usecols=["movieId", "tag"])
    tags_df["tag"] = tags_df["tag"].str.strip().str.lower()
    counts = (
        tags_df.groupby(["movieId", "tag"]).size().reset_index(name="n")
        .sort_values(["movieId", "n"], ascending=[True, False])
    )
    counts["tmdbId"] = counts["movieId"].map(ml_to_tmdb)
    counts = counts.dropna(subset=["tmdbId"])
    counts["tmdbId"] = counts["tmdbId"].astype(int)

    merged = 0
    for tmdb_id, group in counts.groupby("tmdbId"):
        meta = metadata.get(int(tmdb_id))
        if meta is None:
            continue
        existing = {k.lower() for k in meta.keywords}
        meta.keywords.extend(t for t in group["tag"].head(10).tolist() if t not in existing)
        merged += 1
    console.print(f"[green]Merged MovieLens tags into {merged:,} films.[/green]")


def main() -> None:
    ids_path = INDEX_DIR / "films_ids.npy"
    if not ids_path.exists():
        console.print("[red]No existing index (data/index/films_ids.npy). Run scripts/setup.py first.[/red]")
        sys.exit(1)

    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    model_name = config.get("content", {}).get("model")
    client = TMDBClient(config["tmdb"]["api_key"])

    tmdb_ids = [int(x) for x in np.load(ids_path).tolist()]
    console.print(f"Re-embedding [cyan]{len(tmdb_ids):,}[/cyan] films.")

    metadata = load_cached_metadata(tmdb_ids, client)
    merge_movielens_tags(metadata)

    # Back up the current index before overwriting.
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = INDEX_DIR / f"backup-{stamp}"
    backup.mkdir(parents=True, exist_ok=True)
    for f in ("films.faiss", "films_ids.npy", "embedding_model.txt"):
        src = INDEX_DIR / f
        if src.exists():
            shutil.copy2(src, backup / f)
    console.print(f"[dim]Backed up previous index to {backup}[/dim]")

    embedder = EmbeddingModel(model_name)
    console.print(f"Embedding with [cyan]{embedder.model_name}[/cyan] (dim={embedder.dim})...")

    ordered_ids = [tid for tid in tmdb_ids if tid in metadata]
    texts = [metadata[tid].to_text_blob() for tid in ordered_ids]

    batch_size = 256
    all_vecs = []
    for i in track(range(0, len(texts), batch_size), description="Embedding"):
        all_vecs.append(embedder.encode(texts[i: i + batch_size]))
    vectors = np.vstack(all_vecs)

    film_index = FilmIndex()
    film_index.build(vectors, ordered_ids)
    film_index.save(model_name=embedder.model_name)
    console.print(f"[bold green]Rebuilt index: {len(ordered_ids):,} films, {embedder.dim}-dim, {embedder.model_name}.[/bold green]")


if __name__ == "__main__":
    main()
