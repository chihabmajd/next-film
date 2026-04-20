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

The pipeline then runs in two stages:

1. **FAISS retrieval** — finds the 500 most content-similar unwatched films using cosine similarity over 384-dimensional sentence-transformer embeddings (all-MiniLM-L6-v2).
2. **Collaborative filtering rerank** — scores those candidates using a user vector projected into the MovieLens 25M SVD latent space via confidence-weighted least-squares fold-in. CF-ranked results appear first; films not in MovieLens follow sorted by embedding similarity.

**Confidence weighting** — during SVD training, each rating is scaled by `log(1 + number_of_ratings_for_that_film)`, normalized so a median-popularity film keeps weight 1.0. Popular films get stronger signal, so their latent vectors are estimated more precisely. The same weights apply during fold-in: your rating of a well-known film pulls your user vector harder than your rating of a film with 8 total ratings in MovieLens. Hidden gems are unaffected — their latent vectors still exist and they can still be recommended, they just don't distort your user vector.

An optional Ollama LLM layer can generate a 2-sentence explanation for each recommendation. The core system works entirely without it.

## Setup

**Requirements:** Python 3.10–3.13. Python 3.14 is not yet supported by several dependencies.

```bash
git clone https://github.com/yourname/next-film.git
cd next-film
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Edit `config.yaml`:

```yaml
letterboxd:
  username: your_letterboxd_username

tmdb:
  api_key: your_tmdb_api_key   # free at https://www.themoviedb.org/settings/api
```

Then run the one-time setup (downloads ~250 MB of data, fetches TMDB metadata, trains SVD, builds FAISS index — expect 30–90 minutes depending on your connection and machine):

```bash
python scripts/setup.py
```

## Usage

```bash
python src/cli.py
```

The interactive session:

1. Your Letterboxd watch history is fetched and resolved to TMDB IDs.
2. You optionally add reference films and assign weights to each.
3. You optionally describe a mood in free text (e.g. *"something slow and melancholic"*).
4. If both taste and intent are available, you set β.
5. If both reference films and mood text are provided, you set γ.
6. Top-N recommendations are displayed with scores.
7. If Ollama is configured, you can generate explanations.

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
│   └── setup.py              # one-time: download data, train SVD, build FAISS index
├── src/
│   ├── cli.py                # main entrypoint
│   ├── scrapers/
│   │   └── letterboxd.py     # Letterboxd RSS parser
│   ├── enrichment/
│   │   └── tmdb.py           # TMDB API client with JSON caching
│   ├── models/
│   │   ├── collaborative.py  # SVD training + user fold-in
│   │   ├── embeddings.py     # sentence-transformer encoding + FAISS index
│   │   └── hybrid.py         # two-stage FAISS → CF ranker
│   ├── query/
│   │   └── builder.py        # taste vector + query vector construction
│   ├── search/
│   │   └── film_search.py    # TMDB search + rapidfuzz fallback
│   └── llm/
│       └── explainer.py      # optional Ollama explanation layer
└── data/                     # created by setup.py, not committed
    ├── movielens/            # MovieLens 25M raw files
    ├── index/                # FAISS index, SVD model, ID maps
    └── cache/                # TMDB API response cache (per-film JSON)
```

## Limitations

- Letterboxd's RSS feed returns only the ~50 most recent diary entries. Taste vector quality improves with more rated films, but only recent activity is available without a full export.
- Collaborative filtering coverage depends on MovieLens 25M (62,000 films). Films absent from MovieLens still appear in recommendations but are ranked by embedding similarity only.
- First-run setup makes tens of thousands of TMDB API requests. The per-film JSON cache means re-runs are fast.

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
