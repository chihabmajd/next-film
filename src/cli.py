"""next-film CLI — intelligent movie recommender."""

import json
import sys
from pathlib import Path

import yaml
from rich.console import Console
from rich.prompt import Confirm, FloatPrompt, Prompt
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.enrichment.tmdb import FilmMetadata, TMDBClient
from src.llm.explainer import LLMExplainer
from src.models.collaborative import CollaborativeModel
from src.models.embeddings import EmbeddingModel, FilmIndex
from src.models.hybrid import HybridRanker, Recommendation
from src.query.builder import QueryBuilder
from src.scrapers.letterboxd import fetch_watched
from src.search.film_search import FilmSearcher

console = Console()
INDEX_DIR = Path(__file__).parent.parent / "data" / "index"
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def load_models() -> tuple[EmbeddingModel, FilmIndex, CollaborativeModel, dict, dict[int, str]]:
    console.print("Loading models...", end=" ")
    embedder = EmbeddingModel()
    film_index = FilmIndex()
    film_index.load()
    cf_model = CollaborativeModel()
    cf_model.load()
    tmdb_to_ml = {int(k): v for k, v in json.loads((INDEX_DIR / "tmdb_to_ml.json").read_text()).items()}
    titles_path = INDEX_DIR / "film_titles.json"
    local_titles: dict[int, str] = (
        {int(k): v for k, v in json.loads(titles_path.read_text()).items()}
        if titles_path.exists() else {}
    )
    console.print("[green]done.[/green]")
    return embedder, film_index, cf_model, tmdb_to_ml, local_titles


def prompt_reference_films(searcher: FilmSearcher) -> list[tuple[int, float]]:
    console.print("\n[bold]Reference films[/bold] (press Enter to skip)")
    references: list[tuple[int, float]] = []

    while True:
        query = Prompt.ask("  Add a film (or leave blank to continue)", default="")
        if not query.strip():
            break

        matches = searcher.search(query.strip())
        if not matches:
            console.print("  [yellow]No results found. Try a different spelling.[/yellow]")
            continue

        if len(matches) == 1:
            film = matches[0]
            console.print(f"  Found: [cyan]{film.title} ({film.year})[/cyan]")
        else:
            console.print("  Multiple matches:")
            for i, m in enumerate(matches, 1):
                console.print(f"    [bold]{i}.[/bold] {m.title} ({m.year})")
            choice = Prompt.ask("  Pick one", choices=[str(i) for i in range(1, len(matches) + 1)])
            film = matches[int(choice) - 1]

        weight = FloatPrompt.ask(f"  Weight for [cyan]{film.title}[/cyan]", default=1.0)
        references.append((film.tmdb_id, weight))

    return references


def display_results(
    recommendations: list[Recommendation],
    metadata: dict[int, FilmMetadata],
) -> list[Recommendation]:
    table = Table(title="Recommendations", show_lines=True)
    table.add_column("#", style="bold", width=3)
    table.add_column("Title", style="cyan")
    table.add_column("Year", width=6)
    table.add_column("Genres")
    table.add_column("Score", justify="right")

    for i, rec in enumerate(recommendations, 1):
        meta = metadata.get(rec.tmdb_id)
        title = meta.title if meta else f"TMDB:{rec.tmdb_id}"
        year = str(meta.year) if meta else "—"
        genres = ", ".join(meta.genres[:3]) if meta else "—"
        score = f"{rec.cf_score:.3f}" if rec.cf_score is not None else f"{rec.similarity:.3f}"
        table.add_row(str(i), title, year, genres, score)

    console.print(table)
    return recommendations


def main() -> None:
    config = load_config()

    if not film_index_exists():
        console.print("[red]Index not found. Run: python scripts/setup.py[/red]")
        sys.exit(1)

    embedder, film_index, cf_model, tmdb_to_ml, local_titles = load_models()
    tmdb_client = TMDBClient(config["tmdb"]["api_key"])
    searcher = FilmSearcher(tmdb_client, local_titles)
    query_builder = QueryBuilder(embedder, film_index)

    defaults = config.get("defaults", {})
    top_n = defaults.get("top_n", 10)

    # Fetch Letterboxd history
    username = config["letterboxd"]["username"]
    console.print(f"\nFetching Letterboxd history for [cyan]{username}[/cyan]...")
    try:
        watched = fetch_watched(username)
        console.print(f"  Found [green]{len(watched)}[/green] films.")
    except Exception as e:
        console.print(f"[red]Could not fetch Letterboxd data: {e}[/red]")
        watched = []

    # Resolve watched films to TMDB IDs
    watched_tmdb_ids: set[int] = set()
    rated_films: dict[int, float] = {}
    for film in watched:
        results = tmdb_client.search(film.title, film.year)
        if results:
            tmdb_id = results[0]["id"]
            watched_tmdb_ids.add(tmdb_id)
            if film.rating > 0:
                rated_films[tmdb_id] = film.rating

    console.print(f"  Resolved [green]{len(watched_tmdb_ids)}[/green] films to TMDB IDs.")

    # Build taste vector
    taste_vector = query_builder.build_taste_vector(rated_films)
    if taste_vector is None:
        console.print("[yellow]No rated films found — recommendations will be based on intent only.[/yellow]")

    # Reference films
    references = prompt_reference_films(searcher)

    # Mood text
    mood_text = Prompt.ask("\n[bold]Mood / description[/bold] (or leave blank)", default="")
    mood_text = mood_text.strip() or None

    if not references and not mood_text and taste_vector is None:
        console.print("[red]Please provide at least reference films, a mood, or have rated films on Letterboxd.[/red]")
        sys.exit(1)

    # β and γ
    has_intent = bool(references or mood_text)
    if has_intent and taste_vector is not None:
        beta = max(0.0, min(1.0, FloatPrompt.ask(
            "\n[bold]Intent vs taste balance β[/bold] (0=pure taste, 1=pure intent)",
            default=defaults.get("beta", 0.5),
        )))
    elif has_intent:
        beta = 1.0  # no taste signal, pure intent
    else:
        beta = 0.0  # no intent signal, pure taste

    gamma = defaults.get("gamma", 0.6)
    if references and mood_text:
        gamma = max(0.0, min(1.0, FloatPrompt.ask(
            "[bold]Reference films vs mood γ[/bold] (0=pure mood, 1=pure refs)",
            default=gamma,
        )))

    # Build query vector
    query_vector = query_builder.build_query_vector(
        taste_vector=taste_vector,
        reference_films=references,
        mood_text=mood_text,
        beta=beta,
        gamma=gamma,
    )

    # Fold-in user to CF latent space
    ml_rated = {
        tmdb_to_ml[tmdb_id]: rating
        for tmdb_id, rating in rated_films.items()
        if tmdb_id in tmdb_to_ml
    }
    user_vector = cf_model.fold_in_user(ml_rated)

    # Recommend
    ranker = HybridRanker(film_index, cf_model, tmdb_to_ml)
    recommendations = ranker.recommend(query_vector, user_vector, watched_tmdb_ids, top_n=top_n)

    if not recommendations:
        console.print("[yellow]No recommendations found.[/yellow]")
        return

    # Fetch metadata for display
    meta_map = tmdb_client.get_metadata_batch([r.tmdb_id for r in recommendations])
    display_results(recommendations, meta_map)

    # Optional LLM explanations
    llm_config = config.get("llm", {})
    if llm_config.get("enabled") and llm_config.get("provider") == "ollama":
        explainer = LLMExplainer(model=llm_config.get("model", "llama3"))
        ref_titles = []
        for tmdb_id, _ in references:
            meta = tmdb_client.get_metadata(tmdb_id)
            if meta:
                ref_titles.append(meta.title)

        if Confirm.ask("\nGenerate explanations via Ollama?"):
            for i, rec in enumerate(recommendations, 1):
                meta = meta_map.get(rec.tmdb_id)
                if meta:
                    explanation = explainer.explain(meta, ref_titles, mood_text)
                    if explanation:
                        console.print(f"\n[bold]{i}. {meta.title}[/bold]")
                        console.print(f"   {explanation}")


def film_index_exists() -> bool:
    return (
        (INDEX_DIR / "films.faiss").exists()
        and (INDEX_DIR / "films_ids.npy").exists()
        and (INDEX_DIR / "svd_model.pkl").exists()
        and (INDEX_DIR / "tmdb_to_ml.json").exists()
    )


if __name__ == "__main__":
    main()
