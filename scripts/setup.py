"""One-time setup: download MovieLens 25M, build FAISS index, train SVD model."""

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from rich.console import Console
from rich.progress import track

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.enrichment.tmdb import FilmMetadata, TMDBClient
from src.models.collaborative import CollaborativeModel
from src.models.embeddings import EmbeddingModel, FilmIndex

console = Console()

DATA_DIR = Path(__file__).parent.parent / "data"
MOVIELENS_DIR = DATA_DIR / "movielens"
INDEX_DIR = DATA_DIR / "index"
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"


def download_movielens() -> None:
    if (MOVIELENS_DIR / "ratings.csv").exists():
        console.print("[green]MovieLens already downloaded.[/green]")
        return

    console.print("Downloading MovieLens 25M (~250MB)...")
    MOVIELENS_DIR.mkdir(parents=True, exist_ok=True)
    response = requests.get(MOVIELENS_URL, stream=True, timeout=120)
    response.raise_for_status()

    content = b""
    total = int(response.headers.get("content-length", 0))
    for chunk in track(response.iter_content(chunk_size=1024 * 1024), description="Downloading", total=total // (1024 * 1024)):
        content += chunk

    with zipfile.ZipFile(io.BytesIO(content)) as z:
        for name in z.namelist():
            filename = Path(name).name
            if filename in ("ratings.csv", "movies.csv", "links.csv"):
                data = z.read(name)
                (MOVIELENS_DIR / filename).write_bytes(data)
    console.print("[green]MovieLens downloaded.[/green]")


def build_tmdb_mapping(tmdb_client: TMDBClient) -> dict[int, int]:
    """Map MovieLens movie IDs to TMDB IDs using the links.csv file."""
    mapping_path = INDEX_DIR / "ml_to_tmdb.json"
    if mapping_path.exists():
        console.print("[green]MovieLens→TMDB mapping already exists.[/green]")
        return {int(k): v for k, v in json.loads(mapping_path.read_text()).items()}

    links = pd.read_csv(MOVIELENS_DIR / "links.csv")
    mapping = {}
    for _, row in track(links.iterrows(), description="Building ML→TMDB mapping", total=len(links)):
        ml_id = int(row["movieId"])
        tmdb_id = row.get("tmdbId")
        if pd.notna(tmdb_id):
            mapping[ml_id] = int(tmdb_id)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    mapping_path.write_text(json.dumps(mapping))
    console.print(f"[green]Mapped {len(mapping)} MovieLens films to TMDB.[/green]")
    return mapping


def fetch_tmdb_metadata(tmdb_client: TMDBClient, tmdb_ids: list[int]) -> dict[int, FilmMetadata]:
    console.print(f"Fetching TMDB metadata for {len(tmdb_ids)} films...")
    return tmdb_client.get_metadata_batch(tmdb_ids)


def build_embeddings(metadata: dict[int, FilmMetadata], embedder: EmbeddingModel, film_index: FilmIndex) -> None:
    if film_index.is_built():
        console.print("[green]FAISS index already built.[/green]")
        return

    tmdb_ids = list(metadata.keys())
    texts = [metadata[tid].to_text_blob() for tid in tmdb_ids]

    console.print(f"Embedding {len(texts)} films...")
    batch_size = 256
    all_vecs = []
    for i in track(range(0, len(texts), batch_size), description="Embedding"):
        batch = texts[i: i + batch_size]
        vecs = embedder.encode(batch)
        all_vecs.append(vecs)

    all_vecs = np.vstack(all_vecs)
    film_index.build(all_vecs, tmdb_ids)
    film_index.save()
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


def fetch_letterboxd_tmdb_ids(tmdb_client: TMDBClient, username: str) -> list[int]:
    """Resolve Letterboxd watched films to TMDB IDs so they get embedded in the index."""
    from src.scrapers.letterboxd import fetch_watched
    try:
        watched = fetch_watched(username)
    except Exception as e:
        console.print(f"[yellow]Could not fetch Letterboxd history: {e}[/yellow]")
        return []

    tmdb_ids = []
    for film in track(watched, description="Resolving Letterboxd films to TMDB"):
        results = tmdb_client.search(film.title, film.year)
        if results:
            tmdb_ids.append(results[0]["id"])
    return tmdb_ids


def main() -> None:
    import yaml
    config_path = Path(__file__).parent.parent / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    api_key = config["tmdb"]["api_key"]
    username = config["letterboxd"]["username"]

    if api_key == "your_tmdb_api_key":
        console.print("[red]Set your TMDB API key in config.yaml first.[/red]")
        sys.exit(1)

    download_movielens()

    tmdb_client = TMDBClient(api_key)
    ml_to_tmdb = build_tmdb_mapping(tmdb_client)

    # Combine MovieLens films + user's Letterboxd films so all are embedded
    lb_tmdb_ids = fetch_letterboxd_tmdb_ids(tmdb_client, username)
    tmdb_ids = list(set(ml_to_tmdb.values()) | set(lb_tmdb_ids))
    console.print(f"Total films to embed: {len(tmdb_ids):,} (MovieLens + Letterboxd)")

    metadata = fetch_tmdb_metadata(tmdb_client, tmdb_ids)

    embedder = EmbeddingModel()
    film_index = FilmIndex()
    build_embeddings(metadata, embedder, film_index)

    cf_model = CollaborativeModel()
    train_svd(cf_model)

    # Save tmdb→ml reverse map for the ranker
    tmdb_to_ml = {v: k for k, v in ml_to_tmdb.items()}
    (INDEX_DIR / "tmdb_to_ml.json").write_text(json.dumps({str(k): v for k, v in tmdb_to_ml.items()}))

    # Save title lookup for rapidfuzz fallback in film search
    titles = {str(tid): f"{m.title} ({m.year})" for tid, m in metadata.items()}
    (INDEX_DIR / "film_titles.json").write_text(json.dumps(titles))

    console.print("\n[bold green]Setup complete. Run: python src/cli.py[/bold green]")


if __name__ == "__main__":
    main()
