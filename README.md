# next-film

A personal movie recommender that combines collaborative filtering and semantic embeddings to suggest films you haven't seen yet — based on your Letterboxd history and today's intent.

## How it works

The system has two signals:

**Taste** — what you generally like, derived from your Letterboxd ratings. Films you rated above your personal average contribute more. This is a weighted average of content vectors, normalized to a unit vector in embedding space.

**Intent** — what you're in the mood for right now. You provide reference films (with optional weights), a free-text mood description, or both. These are blended with a γ parameter (0 = pure mood text, 1 = pure reference films).

At query time you set β (0–1) to control how much your current intent matters versus your long-term taste.

```
query_vector = normalize(β × intent_vector + (1−β) × taste_vector)
```

The query vector is then used to drive a two-stage pipeline:

1. **FAISS retrieval** — the query vector is used to find the 500 most content-similar unwatched films via cosine similarity over 384-dimensional sentence-transformer embeddings (all-MiniLM-L6-v2). β only affects this step — it shapes what gets retrieved, not how results are ranked.
2. **Collaborative filtering rerank** — those 500 candidates are scored independently using a user vector projected into the MovieLens 32M SVD latent space via confidence-weighted least-squares fold-in. The final ranking is purely by CF score. Films not covered by MovieLens fall back to FAISS similarity order, appended after the CF-ranked results.

**Confidence weighting** — during SVD training, each rating is scaled by `log(1 + number_of_ratings_for_that_film)`, normalized so a median-popularity film keeps weight 1.0. Popular films get stronger signal, so their latent vectors are estimated more precisely. The same weights apply during fold-in: your rating of a well-known film pulls your user vector harder than your rating of a film with 8 total ratings in MovieLens. Hidden gems are unaffected — their latent vectors still exist and they can still be recommended, they just don't distort your user vector.

An optional Ollama LLM layer can generate a 2-sentence explanation for each recommendation. The core system works entirely without it.

## Setup

**Requirements:** Python 3.10–3.13. Python 3.14 is not yet supported by several dependencies.

```bash
git clone git@github.com:yourname/next-film.git
cd next-film
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 1. Download MovieLens 32M manually

Download **ml-32m** from https://grouplens.org/datasets/movielens/32m/ and unzip it. Place the four CSV files in `data/movielens/`:

```
data/movielens/
├── links.csv
├── movies.csv
├── ratings.csv
└── tags.csv
```

MovieLens 32M covers **87,585 films** and **32 million ratings** from January 1995 to October 2023.

### 2. Configure

Edit `config.yaml`:

```yaml
letterboxd:
  username: your_letterboxd_username

tmdb:
  api_key: your_tmdb_api_key   # free at https://www.themoviedb.org/settings/api
```

### 3. Run setup

```bash
python scripts/setup.py
```

Setup steps (each is skipped on rerun if already done):
1. Reads `links.csv` to map MovieLens IDs → TMDB IDs directly (no search needed)
2. Reads `movies.csv` to build base metadata (title, year, genres) for all films instantly
3. Reads `tags.csv` to extract top user-applied tags per film and merge into embeddings
4. Fetches rich metadata from TMDB (plot, director, cast, keywords) — falls back to `movies.csv` data for any film that fails
5. Embeds all films with `all-MiniLM-L6-v2` and saves a FAISS index
6. Trains SVD on the full ratings matrix

Expected time on first run: **60–90 minutes**, almost entirely the TMDB metadata fetch (~87k individual API calls at ~40 req/s). The per-film JSON cache means rerunning is instant.

## Usage

```bash
python src/cli.py
```

The interactive session:

1. Your Letterboxd watch history is loaded (CSV export or RSS).
2. You optionally add reference films and assign weights to each.
3. You optionally describe a mood in free text (e.g. *"something slow and melancholic"*).
4. If both taste and intent are available, you set β.
5. If both reference films and mood text are provided, you set γ.
6. Top-N recommendations are displayed with scores.
7. If Ollama is configured, you can generate explanations.

## Full Letterboxd history (recommended)

The RSS feed returns only your ~50 most recent diary entries. For full history, download your Letterboxd export:

**letterboxd.com → Settings → Data → Export Your Data**

Unzip it, then set the path in `config.yaml`:

```yaml
letterboxd:
  username: your_username
  export_dir: /path/to/letterboxd_export
```

The export contains `ratings.csv` (all rated films) and `watched.csv` (all watched films including unrated). Both are read and merged. When `export_dir` is set it takes priority over RSS everywhere — both in `setup.py` and `cli.py`.

## Optional LLM explanations

Install and run [Ollama](https://ollama.com), then enable it in `config.yaml`:

```yaml
llm:
  enabled: true
  provider: ollama
  model: llama3
```

## Configuration reference

| Key | Default | Description |
|-----|---------|-------------|
| `letterboxd.username` | — | Your Letterboxd username |
| `letterboxd.export_dir` | — | Path to unzipped Letterboxd export (optional, recommended) |
| `tmdb.api_key` | — | TMDB API key (free) |
| `defaults.beta` | `0.5` | Intent vs taste balance (0 = pure taste, 1 = pure intent) |
| `defaults.gamma` | `0.6` | Reference films vs mood text (0 = pure mood, 1 = pure refs) |
| `defaults.top_n` | `10` | Number of results |
| `llm.enabled` | `false` | Enable Ollama explanations |
| `llm.model` | `llama3` | Ollama model name |

## Project structure

```
next-film/
├── config.yaml
├── scripts/
│   └── setup.py              # one-time: build FAISS index, train SVD
├── src/
│   ├── cli.py                # main entrypoint
│   ├── scrapers/
│   │   └── letterboxd.py     # RSS parser + CSV export loader
│   ├── enrichment/
│   │   └── tmdb.py           # TMDB API client with JSON caching
│   ├── models/
│   │   ├── collaborative.py  # SVD training + confidence-weighted user fold-in
│   │   ├── embeddings.py     # sentence-transformer encoding + FAISS index
│   │   └── hybrid.py         # two-stage FAISS → CF ranker
│   ├── query/
│   │   └── builder.py        # taste vector + query vector construction
│   ├── search/
│   │   └── film_search.py    # TMDB search + rapidfuzz fallback
│   └── llm/
│       └── explainer.py      # optional Ollama explanation layer
└── data/                     # created by setup.py, not committed
    ├── movielens/            # ml-32m CSV files (place here manually)
    ├── index/                # FAISS index, SVD model, ID maps
    └── cache/                # TMDB API response cache (per-film JSON)
```

## Limitations

- Collaborative filtering coverage depends on MovieLens 32M (87,585 films, January 1995 – October 2023). Films released after October 2023 have no CF score and fall back to embedding similarity.
- First-run setup makes ~87k individual TMDB API calls. The per-film JSON cache means re-runs are fast.

## Citation

This project uses the MovieLens 32M dataset:

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems (TiiS)* 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872

## Dependencies

| Package | Purpose |
|---------|---------|
| `sentence-transformers` | `all-MiniLM-L6-v2` embeddings |
| `faiss-cpu` | Approximate nearest-neighbour search |
| `scikit-learn` | Randomized SVD |
| `scipy` | Sparse ratings matrix |
| `pandas` | MovieLens data loading |
| `rapidfuzz` | Fuzzy film title fallback |
| `requests` | TMDB API + Letterboxd RSS |
| `rich` | Terminal UI |
| `pyyaml` | Config parsing |
| `ollama` | Optional LLM explanations |
