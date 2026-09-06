"""Rebuilds the FAISS content index for films already in it, with a new model or text blob.

Does not touch MovieLens/Letterboxd resolution or the SVD model. Re-embeds the films in
data/index/films_ids.npy from the cached TMDB metadata (no network), re-merging MovieLens
tags. The previous index is backed up.

Run after changing the embedding model or FilmMetadata.to_text_blob():

    python scripts/reembed.py
"""

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from rich.console import Console

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))  # to import the sibling `setup` module

from setup import _merge_tags, load_movielens_tags  # reuse the exact tag-merge setup.py uses
from src.enrichment.tmdb import TMDBClient
from src.models.embeddings import EmbeddingModel, FilmIndex, embed_and_index

console = Console()
INDEX_DIR = ROOT / "data" / "index"


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

    # Metadata straight from the TMDB cache (every id here was cached by setup, so no network).
    metadata = client.get_metadata_batch(tmdb_ids)
    if (missing := len(tmdb_ids) - len(metadata)):
        console.print(f"  [yellow]{missing:,} films not in cache — skipped.[/yellow]")
    ml_to_tmdb = {int(k): int(v) for k, v in json.loads((INDEX_DIR / "ml_to_tmdb.json").read_text()).items()}
    _merge_tags(metadata, load_movielens_tags(ml_to_tmdb))

    # Back up the current index before overwriting.
    backup = INDEX_DIR / f"backup-{time.strftime('%Y%m%d-%H%M%S')}"
    backup.mkdir(parents=True, exist_ok=True)
    for name in ("films.faiss", "films_ids.npy", "embedding_model.txt"):
        if (src := INDEX_DIR / name).exists():
            shutil.copy2(src, backup / name)
    console.print(f"[dim]Backed up previous index to {backup}[/dim]")

    embedder = EmbeddingModel(model_name)
    console.print(f"Embedding with [cyan]{embedder.model_name}[/cyan] (dim={embedder.dim})...")
    ordered_ids = [tid for tid in tmdb_ids if tid in metadata]
    texts = [metadata[tid].to_text_blob() for tid in ordered_ids]
    embed_and_index(embedder, FilmIndex(), ordered_ids, texts)
    console.print(
        f"[bold green]Rebuilt index: {len(ordered_ids):,} films, {embedder.dim}-dim, {embedder.model_name}.[/bold green]"
    )


if __name__ == "__main__":
    main()
