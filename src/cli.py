"""next-film CLI — personal cinema recommender."""

import os

# The CLI only ever loads an already-cached embedding model, so run the Hugging Face stack
# offline: no hub round-trip (faster start) and no "unauthenticated requests" notice. Setup
# and reembed do NOT set this — they need to download models.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import sys
from pathlib import Path

import yaml
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import FloatPrompt, Prompt
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.enrichment.tmdb import FilmMetadata, TMDBClient
from src.llm.explainer import Explainer
from src.models.collaborative import CollaborativeModel
from src.models.embeddings import EmbeddingModel, FilmIndex, index_model_name
from src.models.hybrid import HybridRanker, Recommendation
from src.query.builder import QueryBuilder
from src.scrapers.letterboxd import fetch_watched, load_from_export
from src.search.film_search import FilmMatch, FilmSearcher

console = Console()
INDEX_DIR = Path(__file__).parent.parent / "data" / "index"
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def banner() -> None:
    console.print()
    console.print(
        Panel(
            "[bold]next-film[/bold]   [dim]· find your next film[/dim]",
            box=box.DOUBLE,
            border_style="bright_cyan",
            padding=(0, 3),
            expand=False,
        )
    )


def load_models() -> tuple[FilmIndex, CollaborativeModel, dict, dict[int, str]]:
    # The embedding model is loaded lazily (only when a mood is entered) — taste and reference
    # vectors come straight from the index, so most runs never need the 400 MB encoder.
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
    return film_index, cf_model, tmdb_to_ml, local_titles


def resolve_watched(
    watched: list, tmdb_client: TMDBClient
) -> tuple[set[int], dict[int, float]]:
    """Resolve Letterboxd films to TMDB ids via robust best-match, cached to disk.

    Resolution is deterministic and slow (one TMDB search per unseen film), so results are
    persisted to data/index/lb_resolved.json keyed by "title|year". Films that can't be
    resolved are cached as -1 so they aren't re-searched every run.
    """
    cache_path = INDEX_DIR / "lb_resolved.json"
    cache: dict[str, int] = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    watched_tmdb_ids: set[int] = set()
    rated_films: dict[int, float] = {}
    dirty = False
    for film in watched:
        key = f"{film.title}|{film.year}"
        if key in cache:
            tmdb_id = cache[key]
        else:
            resolved = tmdb_client.best_match(film.title, film.year)
            tmdb_id = resolved if resolved is not None else -1
            cache[key] = tmdb_id
            dirty = True
        if tmdb_id is None or tmdb_id < 0:
            continue
        watched_tmdb_ids.add(tmdb_id)
        if film.rating > 0:
            rated_films[tmdb_id] = film.rating

    if dirty:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache))
    return watched_tmdb_ids, rated_films


def prompt_reference_films(searcher: FilmSearcher) -> list[tuple[int, float]]:
    console.print("[bold]Reference films[/bold] [dim]— films to steer toward. Enter to skip / when done.[/dim]")
    found: list[FilmMatch] = []

    while True:
        query = Prompt.ask("  [cyan]film[/cyan]", default="", show_default=False)
        if not query.strip():
            break
        matches = searcher.search(query.strip())
        if not matches:
            console.print("  [yellow]no results — try another spelling.[/yellow]")
            continue
        if len(matches) == 1:
            film = matches[0]
        else:
            for i, m in enumerate(matches, 1):
                console.print(f"    [bold]{i}.[/bold] {m.title} [dim]({m.year})[/dim]")
            choice = Prompt.ask("  pick one", choices=[str(i) for i in range(1, len(matches) + 1)])
            film = matches[int(choice) - 1]
        console.print(f"  [green]✓[/green] [cyan]{film.title}[/cyan] [dim]({film.year})[/dim]")
        found.append(film)

    if not found:
        return []
    if len(found) == 1:
        return [(found[0].tmdb_id, 1.0)]

    console.print(f"  [dim]{len(found)} films — set relative weights (Enter = 1.0).[/dim]")
    references = []
    for film in found:
        while True:
            weight = FloatPrompt.ask(f"    weight for [cyan]{film.title}[/cyan]", default=1.0)
            if weight > 0:
                break
            console.print("    [yellow]must be positive.[/yellow]")
        references.append((film.tmdb_id, weight))
    return references


def _fit(cf_z: float | None) -> str:
    """The debiased 'people like you' nudge, as a coloured arrow."""
    if cf_z is None:
        return "[dim]·[/dim]"
    if cf_z >= 0.5:
        return "[bold green]▲▲[/bold green]"
    if cf_z > 0:
        return "[green]▲[/green]"
    if cf_z <= -0.5:
        return "[bold red]▼▼[/bold red]"
    return "[red]▼[/red]"


def display_results(
    recommendations: list[Recommendation],
    metadata: dict[int, FilmMetadata],
    avg_ratings: dict[int, float],
    explanations: dict[int, str],
    backend_note: str,
) -> None:
    table = Table(box=box.SIMPLE_HEAVY, show_lines=True, padding=(0, 1), expand=True)
    table.add_column("#", justify="right", style="bold bright_black", width=2)
    table.add_column("Film", ratio=3)
    table.add_column("Match", justify="center", width=6)
    table.add_column("Fit", justify="center", width=4)
    table.add_column("★", justify="center", width=4)
    table.add_column("Why", ratio=5, overflow="fold")

    for i, rec in enumerate(recommendations, 1):
        meta = metadata.get(rec.tmdb_id)
        title = meta.title if meta else f"TMDB:{rec.tmdb_id}"
        year = meta.year if meta and meta.year else None
        genres = ", ".join(meta.genres[:3]) if meta and meta.genres else ""
        film_cell = f"[bold cyan]{title}[/bold cyan]"
        if year:
            film_cell += f"  [dim]{year}[/dim]"
        if genres:
            film_cell += f"\n[dim]{genres}[/dim]"

        pct = max(0.0, rec.content_sim) * 100
        color = "green" if pct >= 65 else "yellow" if pct >= 50 else "bright_black"
        match_cell = f"[{color}]{pct:.0f}%[/{color}]"

        avg = avg_ratings.get(rec.tmdb_id)
        star_cell = f"[yellow]{avg:.1f}[/yellow]" if avg is not None else "[dim]·[/dim]"

        table.add_row(
            str(i), film_cell, match_cell, _fit(rec.cf_z), star_cell,
            explanations.get(rec.tmdb_id, ""),
        )

    console.print()
    console.print(table)
    console.print(
        f"[dim]Match = how close to your query · Fit ▲/▼ = viewers with your taste · "
        f"★ = community avg · explanations: {backend_note}[/dim]"
    )


def main() -> None:
    banner()
    config = load_config()

    if not film_index_exists():
        console.print("[red]Index not found. Run: python scripts/setup.py[/red]")
        sys.exit(1)

    with console.status("[cyan]loading models…", spinner="dots"):
        film_index, cf_model, tmdb_to_ml, local_titles = load_models()
    tmdb_client = TMDBClient(config["tmdb"]["api_key"])
    searcher = FilmSearcher(tmdb_client, local_titles)
    query_builder = QueryBuilder(embedder=None, film_index=film_index)  # encoder loaded lazily

    defaults = config.get("defaults", {})
    top_n = defaults.get("top_n", 10)
    exploration = float(defaults.get("exploration", 0.0))
    exp_cfg = config.get("explanations", {})
    exp_provider = exp_cfg.get("provider", "auto")
    exp_model = exp_cfg.get("model")

    # ---- Letterboxd history --------------------------------------------------------------
    lb_config = config["letterboxd"]
    export_dir = lb_config.get("export_dir")
    watched: list = []
    if export_dir:
        try:
            watched = load_from_export(export_dir)
        except Exception as e:
            console.print(f"[red]Could not load export: {e}[/red]")
    else:
        try:
            watched = fetch_watched(lb_config["username"])
        except Exception as e:
            console.print(f"[red]Could not fetch Letterboxd data: {e}[/red]")

    with console.status("[cyan]reading your Letterboxd history…", spinner="dots"):
        watched_tmdb_ids, rated_films = resolve_watched(watched, tmdb_client)
    console.print(
        f"[green]✓[/green] [bold]{len(watched_tmdb_ids)}[/bold] films watched, "
        f"[bold]{len(rated_films)}[/bold] rated."
    )

    taste_vector = query_builder.build_taste_vector(rated_films)
    n_profiles = int(defaults.get("n_taste_profiles", 6))
    taste_profiles = query_builder.build_taste_profiles(rated_films, n_profiles=n_profiles)
    if taste_vector is None:
        console.print("[yellow]No rated films found — recommendations will be based on intent only.[/yellow]")

    # ---- intent --------------------------------------------------------------------------
    console.rule("[bold bright_cyan]what are you after?", style="bright_black")
    references = prompt_reference_films(searcher)
    mood_text = Prompt.ask(
        "[bold]Mood[/bold] [dim]— free text, e.g. “slow and dread-soaked”. Enter to skip[/dim]",
        default="", show_default=False,
    ).strip() or None

    if not references and not mood_text and taste_vector is None:
        console.print("[red]Give me at least a reference film, a mood, or rate some films on Letterboxd.[/red]")
        sys.exit(1)

    has_intent = bool(references or mood_text)
    if has_intent and taste_vector is not None:
        beta = max(0.0, min(1.0, FloatPrompt.ask(
            "[bold]β[/bold] [dim]0 = my usual taste · 1 = only this mood[/dim]",
            default=defaults.get("beta", 0.5),
        )))
    elif has_intent:
        beta = 1.0
    else:
        beta = 0.0

    gamma = defaults.get("gamma", 0.6)
    if references and mood_text:
        gamma = max(0.0, min(1.0, FloatPrompt.ask(
            "[bold]γ[/bold] [dim]0 = only mood · 1 = only reference films[/dim]", default=gamma,
        )))

    # ---- query summary -------------------------------------------------------------------
    ref_titles = [
        m.title for tmdb_id, _ in references
        for m in [tmdb_client.get_metadata(tmdb_id)] if m is not None
    ]
    _print_query_summary(len(rated_films), ref_titles, mood_text, beta, has_intent, taste_vector)

    # Encoding a free-text mood is the only thing that needs the embedding model — load it now.
    if mood_text:
        with console.status("[cyan]loading text encoder…", spinner="dots"):
            query_builder.embedder = EmbeddingModel(index_model_name())

    query_vector = query_builder.build_query_vector(
        taste_vector=taste_vector, reference_films=references,
        mood_text=mood_text, beta=beta, gamma=gamma,
    )
    ml_rated = {tmdb_to_ml[t]: r for t, r in rated_films.items() if t in tmdb_to_ml}
    user_vector = cf_model.fold_in_user(ml_rated)

    # ---- recommend -----------------------------------------------------------------------
    with console.status("[cyan]ranking films…", spinner="dots"):
        ranker = HybridRanker(film_index, cf_model, tmdb_to_ml)
        recommendations = ranker.recommend(
            query_vector, user_vector, watched_tmdb_ids,
            taste_profiles=taste_profiles if beta < 1.0 else None,
            top_n=top_n, weights=defaults.get("weights") or {},
            diversity=float(defaults.get("diversity", 0.3)),
            candidate_pool=int(defaults.get("candidate_pool", 1000)),
            cf_retrieval=int(defaults.get("cf_retrieval", 0)),
            exploration=exploration,
        )
    if not recommendations:
        console.print("[yellow]No recommendations found.[/yellow]")
        return

    meta_map = tmdb_client.get_metadata_batch([r.tmdb_id for r in recommendations])
    avg_ratings = {}
    for rec in recommendations:
        ml_id = tmdb_to_ml.get(rec.tmdb_id)
        avg = cf_model.movie_avg_ratings.get(str(ml_id)) if ml_id is not None else None
        if avg is not None:
            avg_ratings[rec.tmdb_id] = avg

    # ---- explanations (grounded in your highly-rated films) ------------------------------
    liked = {}
    for tmdb_id, rating in rated_films.items():
        if rating < 4.0:
            continue
        vec = film_index.get_vector(tmdb_id)
        if vec is None:
            continue
        meta = tmdb_client.get_metadata(tmdb_id)
        if meta is not None:
            liked[tmdb_id] = (meta, rating, vec)

    explainer = Explainer(provider=exp_provider, model=exp_model, liked=liked)
    if explainer.requested_but_unavailable:
        console.print("[yellow]Ollama requested but no server on :11434 — using built-in explanations "
                      "(run `ollama serve` & `ollama pull llama3.2` to enable the neural writer).[/yellow]")
    backend_note = f"Ollama · {explainer.model}" if explainer.active == "ollama" else "built-in"

    with console.status("[cyan]writing the why…", spinner="dots"):
        explanations = {
            rec.tmdb_id: explainer.explain(meta_map[rec.tmdb_id], film_index.get_vector(rec.tmdb_id), mood_text)
            for rec in recommendations if rec.tmdb_id in meta_map
        }

    display_results(recommendations, meta_map, avg_ratings, explanations, backend_note)


def _print_query_summary(
    n_rated: int, ref_titles: list[str], mood_text: str | None,
    beta: float, has_intent: bool, taste_vector,
) -> None:
    lines = []
    if taste_vector is not None:
        lines.append(f"[bright_black]taste[/bright_black]  from your {n_rated} rated films")
    if ref_titles:
        lines.append(f"[bright_black]like[/bright_black]   {', '.join(ref_titles)}")
    if mood_text:
        lines.append(f"[bright_black]mood[/bright_black]   “{mood_text}”")
    if has_intent and taste_vector is not None:
        lines.append(f"[bright_black]blend[/bright_black]  β={beta:.1f}  ({int((1-beta)*100)}% taste / {int(beta*100)}% mood)")
    console.print(Panel("\n".join(lines), border_style="bright_black", box=box.ROUNDED,
                        title="[dim]searching for[/dim]", title_align="left", expand=False, padding=(0, 2)))


def film_index_exists() -> bool:
    return (
        (INDEX_DIR / "films.faiss").exists()
        and (INDEX_DIR / "films_ids.npy").exists()
        and (INDEX_DIR / "svd_model.pkl").exists()
        and (INDEX_DIR / "tmdb_to_ml.json").exists()
    )


if __name__ == "__main__":
    main()
