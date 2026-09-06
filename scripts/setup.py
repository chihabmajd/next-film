"""One-time setup: build FAISS index and train SVD model.

Place the ml-32m CSV files in data/movielens/ before running.
Download from: https://grouplens.org/datasets/movielens/32m/
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, cast

import pandas as pd
from rich.console import Console
from rich.progress import track

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.enrichment.tmdb import FilmMetadata, TMDBClient
from src.models.collaborative import CollaborativeModel
from src.models.embeddings import EmbeddingModel, FilmIndex, embed_and_index

console = Console()

DATA_DIR = Path(__file__).parent.parent / "data"
MOVIELENS_DIR = DATA_DIR / "movielens"
INDEX_DIR = DATA_DIR / "index"


def check_movielens() -> None:
    required = ["ratings.csv", "movies.csv", "links.csv", "tags.csv"]
    missing = [f for f in required if not (MOVIELENS_DIR / f).exists()]
    if not missing:
        console.print("[green]MovieLens files found.[/green]")
        return
    console.print(f"[red]Missing files in {MOVIELENS_DIR}: {missing}[/red]")
    console.print("  Download ml-32m from https://grouplens.org/datasets/movielens/32m/")
    console.print(f"  Unzip and place the CSV files in: {MOVIELENS_DIR}")
    sys.exit(1)


def build_tmdb_mapping(tmdb_client: TMDBClient) -> dict[int, int]:
    """Map MovieLens movie IDs to TMDB IDs using links.csv."""
    mapping_path = INDEX_DIR / "ml_to_tmdb.json"
    if mapping_path.exists():
        console.print("[green]MovieLens→TMDB mapping already exists.[/green]")
        return {int(k): v for k, v in json.loads(mapping_path.read_text()).items()}

    links = pd.read_csv(MOVIELENS_DIR / "links.csv")
    valid = links.dropna(subset=["tmdbId"])
    valid = valid[valid["tmdbId"] != 0]
    mapping: dict[int, int] = {
        int(k): int(v)
        for k, v in zip(valid["movieId"].tolist(), valid["tmdbId"].tolist())
    }

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    mapping_path.write_text(json.dumps(mapping))
    console.print(f"[green]Mapped {len(mapping)} MovieLens films to TMDB.[/green]")
    return mapping


def load_movielens_base_metadata(ml_to_tmdb: dict[int, int]) -> dict[int, FilmMetadata]:
    """Builds basic FilmMetadata from movies.csv (no API calls); used as a fallback when
    the TMDB fetch fails."""
    movies = pd.read_csv(MOVIELENS_DIR / "movies.csv")
    ml_ids: list[Any] = movies["movieId"].tolist()
    raw_titles: list[Any] = movies["title"].tolist()
    raw_genres: list[Any] = movies["genres"].tolist()
    base: dict[int, FilmMetadata] = {}
    for ml_id, raw_title, raw_genre in track(
        zip(ml_ids, raw_titles, raw_genres), description="Loading movies.csv", total=len(movies)
    ):
        tmdb_id = ml_to_tmdb.get(int(ml_id))
        if tmdb_id is None:
            continue
        title, year = _parse_ml_title(str(raw_title))
        genres_raw = str(raw_genre)
        genres = [] if genres_raw == "(no genres listed)" else genres_raw.split("|")
        base[tmdb_id] = FilmMetadata(
            tmdb_id=tmdb_id,
            title=title,
            year=year,
            genres=genres,
        )
    console.print(f"[green]Base metadata loaded for {len(base):,} films.[/green]")
    return base


def _parse_ml_title(raw: str) -> tuple[str, int]:
    """'Toy Story (1995)' → ('Toy Story', 1995). Handles titles with parentheses."""
    match = re.match(r"^(.+)\s*\((\d{4})\)\s*$", raw)
    if match:
        return match.group(1).strip(), int(match.group(2))
    return raw.strip(), 0


def load_movielens_tags(ml_to_tmdb: dict[int, int], top_n: int = 10) -> dict[int, list[str]]:
    """Aggregates the most-applied user tags per film from tags.csv.

    Merged into film keywords before embedding, to enrich content vectors especially
    for films with sparse TMDB metadata.
    """
    console.print("Loading tags.csv...")
    tags_df = pd.read_csv(MOVIELENS_DIR / "tags.csv", usecols=["movieId", "tag"])
    tags_df["tag"] = tags_df["tag"].str.strip().str.lower()
    tag_counts = (
        tags_df.groupby(["movieId", "tag"])
        .size()
        .reset_index(name="count")
        .sort_values(["movieId", "count"], ascending=[True, False])
    )

    tag_counts["tmdbId"] = tag_counts["movieId"].map(ml_to_tmdb)
    tag_counts = tag_counts.dropna(subset=["tmdbId"])
    tag_counts["tmdbId"] = tag_counts["tmdbId"].astype(int)

    result: dict[int, list[str]] = {}
    for tmdb_id, group in tag_counts.groupby("tmdbId"):
        result[cast(int, tmdb_id)] = group["tag"].head(top_n).tolist()

    console.print(f"[green]Tags loaded for {len(result):,} films.[/green]")
    return result


def fetch_tmdb_metadata(
    tmdb_client: TMDBClient,
    tmdb_ids: list[int],
    base_metadata: dict[int, FilmMetadata],
) -> dict[int, FilmMetadata]:
    """Fetches rich metadata from TMDB (plot, director, cast, keywords); films where it fails
    fall back to base_metadata (title, year, genres from movies.csv)."""
    console.print(f"Fetching TMDB metadata for {len(tmdb_ids):,} films...")
    tmdb_results = tmdb_client.get_metadata_batch(tmdb_ids)
    merged = {**base_metadata, **tmdb_results}  # TMDB overwrites base where available
    fallback_count = len(tmdb_ids) - len(tmdb_results)
    if fallback_count:
        console.print(f"  [yellow]{fallback_count:,} films fell back to movies.csv metadata.[/yellow]")
    return merged


def _merge_tags(metadata: dict[int, FilmMetadata], ml_tags: dict[int, list[str]]) -> None:
    """Extend film keywords with MovieLens user tags, skipping duplicates."""
    for tmdb_id, tags in ml_tags.items():
        if tmdb_id not in metadata:
            continue
        existing = {k.lower() for k in metadata[tmdb_id].keywords}
        metadata[tmdb_id].keywords.extend(t for t in tags if t not in existing)


def build_embeddings(metadata: dict[int, FilmMetadata], embedder: EmbeddingModel, film_index: FilmIndex) -> None:
    if film_index.is_built():
        console.print("[green]FAISS index already built.[/green]")
        return

    tmdb_ids = list(metadata.keys())
    texts = [metadata[tid].to_text_blob() for tid in tmdb_ids]

    console.print(f"Embedding {len(texts):,} films with {embedder.model_name}...")
    embed_and_index(embedder, film_index, tmdb_ids, texts)
    console.print("[green]FAISS index saved.[/green]")


def train_svd(cf_model: CollaborativeModel) -> None:
    if cf_model.is_trained():
        console.print("[green]SVD model already trained.[/green]")
        return

    console.print("Loading ratings (this may take a moment)...")
    ratings = pd.read_csv(MOVIELENS_DIR / "ratings.csv")
    console.print(f"Training SVD on {len(ratings):,} ratings...")
    cf_model.train(ratings)
    cf_model.save()
    console.print("[green]SVD model saved.[/green]")


def fetch_letterboxd_tmdb_ids(tmdb_client: TMDBClient, config: dict) -> list[int]:
    from src.scrapers.letterboxd import fetch_watched, load_from_export
    lb_config = config["letterboxd"]
    export_dir = lb_config.get("export_dir")

    try:
        if export_dir:
            console.print(f"Loading Letterboxd export from {export_dir}...")
            watched = load_from_export(export_dir)
        else:
            watched = fetch_watched(lb_config["username"])
    except Exception as e:
        console.print(f"[yellow]Could not load Letterboxd history: {e}[/yellow]")
        return []

    tmdb_ids = []
    for film in track(watched, description="Resolving Letterboxd films to TMDB"):
        tmdb_id = tmdb_client.best_match(film.title, film.year)
        if tmdb_id is not None:
            tmdb_ids.append(tmdb_id)
    return tmdb_ids


def main() -> None:
    import yaml
    config_path = Path(__file__).parent.parent / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    api_key = config["tmdb"]["api_key"]

    if api_key == "your_tmdb_api_key":
        console.print("[red]Set your TMDB API key in config.yaml first.[/red]")
        sys.exit(1)

    check_movielens()

    tmdb_client = TMDBClient(api_key)
    ml_to_tmdb = build_tmdb_mapping(tmdb_client)

    base_metadata = load_movielens_base_metadata(ml_to_tmdb)
    ml_tags = load_movielens_tags(ml_to_tmdb)

    lb_tmdb_ids = fetch_letterboxd_tmdb_ids(tmdb_client, config)
    all_tmdb_ids = list(set(base_metadata.keys()) | set(lb_tmdb_ids))
    console.print(f"Total films to embed: {len(all_tmdb_ids):,}")

    metadata = fetch_tmdb_metadata(tmdb_client, all_tmdb_ids, base_metadata)
    _merge_tags(metadata, ml_tags)

    model_name = config.get("content", {}).get("model")
    embedder = EmbeddingModel(model_name)
    film_index = FilmIndex()
    build_embeddings(metadata, embedder, film_index)

    cf_model = CollaborativeModel()
    train_svd(cf_model)

    tmdb_to_ml = {v: k for k, v in ml_to_tmdb.items()}
    (INDEX_DIR / "tmdb_to_ml.json").write_text(json.dumps({str(k): v for k, v in tmdb_to_ml.items()}))

    titles = {str(tid): f"{m.title} ({m.year})" for tid, m in metadata.items()}
    (INDEX_DIR / "film_titles.json").write_text(json.dumps(titles))

    console.print("\n[bold green]Setup complete. Run: python src/cli.py[/bold green]")


if __name__ == "__main__":
    main()
