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
from src.scrapers.letterboxd import fetch_watched, load_from_export
from src.search.film_search import FilmMatch, FilmSearcher

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
    console.print("\n[bold]Reference films[/bold] — press Enter when done")
    found: list[FilmMatch] = []

    while True:
        query = Prompt.ask("  Film", default="")
        if not query.strip():
            break
        matches = searcher.search(query.strip())
        if not matches:
            console.print("  [yellow]No results. Try a different spelling.[/yellow]")
            continue
        if len(matches) == 1:
            film = matches[0]
        else:
            for i, m in enumerate(matches, 1):
                console.print(f"    [bold]{i}.[/bold] {m.title} ({m.year})")
            choice = Prompt.ask("  Pick one", choices=[str(i) for i in range(1, len(matches) + 1)])
            film = matches[int(choice) - 1]
        console.print(f"  → [cyan]{film.title} ({film.year})[/cyan]")
        found.append(film)

    if not found:
        return []
    if len(found) == 1:
        return [(found[0].tmdb_id, 1.0)]

    console.print(f"\n  {len(found)} films added. Set relative weights (Enter = 1.0 for all).")
    console.print("  [dim]Weights are relative — 2.0 means twice the influence of 1.0. Must be positive.[/dim]")
    references = []
    for film in found:
        while True:
            weight = FloatPrompt.ask(f"    {film.title} ({film.year})", default=1.0)
            if weight > 0:
                break
            console.print("    [yellow]Weight must be positive.[/yellow]")
        references.append((film.tmdb_id, weight))
    return references


def display_results(
    recommendations: list[Recommendation],
    metadata: dict[int, FilmMetadata],
    global_mean: float,
    avg_ratings: dict[int, float],
    explanations: dict[int, str] | None = None,
) -> None:
    table = Table(title="Recommendations", show_lines=True)
    table.add_column("#", style="bold", width=3)
    table.add_column("Title", style="cyan")
    table.add_column("Year", width=6)
    table.add_column("Genres")
    table.add_column("Predicted", justify="right", width=10)
    table.add_column("Avg rating", justify="right", width=10)
    if explanations:
        table.add_column("Why", max_width=60)

    for i, rec in enumerate(recommendations, 1):
        meta = metadata.get(rec.tmdb_id)
        title = meta.title if meta else f"TMDB:{rec.tmdb_id}"
        year = str(meta.year) if meta else "—"
        genres = ", ".join(meta.genres[:3]) if meta else "—"
        predicted = (
            f"{min(5.0, max(0.5, rec.cf_score + global_mean)):.1f}/5"
            if rec.cf_score is not None else "—"
        )
        avg = avg_ratings.get(rec.tmdb_id)
        avg_col = f"{avg:.1f}/5" if avg is not None else "—"
        row = [str(i), title, year, genres, predicted, avg_col]
        if explanations:
            row.append(explanations.get(rec.tmdb_id, ""))
        table.add_row(*row)

    console.print(table)


def main() -> None:
    config = load_config()

    if not film_index_exists():
        console.print("[red]Index not found. Run: python scripts/setup.py[/red]")
        sys.exit(1)

    embedder, film_index, cf_model, tmdb_to_ml, local_titles = load_models()
    tmdb_client = TMDBClient(config["tmdb"]["api_key"])
    searcher = FilmSearcher(tmdb_client, local_titles)
    query_builder = QueryBuilder(embedder, film_index)

    llm_config = config.get("llm", {})
    use_llm = (
        llm_config.get("enabled")
        and llm_config.get("provider") == "ollama"
        and Confirm.ask("\nGenerate LLM explanations for recommendations?", default=False)
    )

    defaults = config.get("defaults", {})
    top_n = defaults.get("top_n", 10)
    exploration = float(defaults.get("exploration", 0.0))

    # Fetch Letterboxd history
    lb_config = config["letterboxd"]
    username = lb_config["username"]
    export_dir = lb_config.get("export_dir")

    if export_dir:
        console.print(f"\nLoading Letterboxd export from [cyan]{export_dir}[/cyan]...")
        try:
            watched = load_from_export(export_dir)
            console.print(f"  Loaded [green]{len(watched)}[/green] films from export.")
        except Exception as e:
            console.print(f"[red]Could not load export: {e}[/red]")
            watched = []
    else:
        console.print(f"\nFetching Letterboxd history for [cyan]{username}[/cyan] (RSS, ~50 recent)...")
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

    # β — only ask when both taste and intent signals are present
    has_intent = bool(references or mood_text)
    if has_intent and taste_vector is not None:
        beta = max(0.0, min(1.0, FloatPrompt.ask(
            "\n[bold]Intent vs taste balance β[/bold] (0=pure taste, 1=pure intent)",
            default=defaults.get("beta", 0.5),
        )))
    elif has_intent:
        beta = 1.0
    else:
        beta = 0.0

    # γ — only ask when both reference films and mood text are provided
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
    recommendations = ranker.recommend(query_vector, user_vector, watched_tmdb_ids, top_n=top_n, exploration=exploration)

    if not recommendations:
        console.print("[yellow]No recommendations found.[/yellow]")
        return

    # Fetch metadata for display
    meta_map = tmdb_client.get_metadata_batch([r.tmdb_id for r in recommendations])

    # Build avg ratings lookup (tmdb_id → avg MovieLens rating)
    avg_ratings: dict[int, float] = {}
    for rec in recommendations:
        ml_id = tmdb_to_ml.get(rec.tmdb_id)
        if ml_id is not None:
            avg = cf_model.movie_avg_ratings.get(str(ml_id))
            if avg is not None:
                avg_ratings[rec.tmdb_id] = avg

    # Generate LLM explanations before rendering the table
    explanations: dict[int, str] | None = None
    if use_llm:
        explainer = LLMExplainer(model=llm_config.get("model", "llama3"))
        ref_titles = [
            m.title
            for tmdb_id, _ in references
            for m in [tmdb_client.get_metadata(tmdb_id)]
            if m is not None
        ]
        console.print("Generating explanations...")
        explanations = {}
        for rec in recommendations:
            meta = meta_map.get(rec.tmdb_id)
            if meta:
                explanation = explainer.explain(meta, ref_titles, mood_text)
                if explanation:
                    explanations[rec.tmdb_id] = explanation

    display_results(recommendations, meta_map, cf_model.global_mean, avg_ratings, explanations)


def film_index_exists() -> bool:
    return (
        (INDEX_DIR / "films.faiss").exists()
        and (INDEX_DIR / "films_ids.npy").exists()
        and (INDEX_DIR / "svd_model.pkl").exists()
        and (INDEX_DIR / "tmdb_to_ml.json").exists()
    )


if __name__ == "__main__":
    main()
