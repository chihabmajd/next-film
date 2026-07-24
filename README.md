# next-film

A personal movie recommender that combines collaborative filtering and semantic embeddings to suggest films you haven't seen yet — based on your Letterboxd history and today's intent.

## How it works

### Two prediction logics

At its core the system combines two different ways of guessing what you'll like, because each covers the other's blind spot:

- **Content (semantic embeddings)** — looks at the films *themselves*. Every film becomes a vector built from its text (director, genre, themes, a little plot); nearby vectors mean similar films. This finds films that *resemble* what you like and works for **any** film with metadata — including obscure and brand-new ones — but it only sees surface style and topic. It can't tell that fans of a quiet Korean drama also tend to love a loud American thriller; nothing in the text links them.
- **Collaborative filtering** — ignores what films are *about* and looks only at rating patterns across 32M MovieLens ratings. If people who love X also love Y, then X and Y sit close in *taste space* even across genres. This captures the "people like you also loved…" correlations content is blind to — but only for films with enough ratings, and only once you've rated enough yourself to be placed well.

next-film uses **content for retrieval** (gather candidates near your taste) and to compute a match score, and **collaborative filtering to adjust the ranking** by predicted affinity, then blends the two. With a sparse rating history the collaborative half is weak, so content carries most of the weight until you've rated more.

### Aiming the content query: taste vs intent

Separately from the two logics above, the *content* query is aimed by combining two signals:

**Taste** — what you generally like, derived from your Letterboxd ratings. It's *two-sided* and *multi-modal*:

- Two-sided: films you rated above your personal average pull your taste vector toward them; films you rated *below* your average push it away (down-weighted, since a dislike is a noisier signal than a like). Rating your own average contributes nothing. This uses your whole history — the earlier version discarded everything you disliked, throwing away half the information in your ratings.
- Multi-modal: your likes are also clustered into a few **taste profiles** (e.g. noir, screwball comedy, slow art films) instead of one averaged centroid. A single average of an eclectic history collapses into a mushy vector that retrieves generic acclaimed films; keeping the modes separate lets retrieval pull candidates for *each* facet of your taste.

**Intent** — what you're in the mood for right now. You provide reference films (with optional weights), a free-text mood description, or both. These are blended with a γ parameter (0 = pure mood text, 1 = pure reference films).

At query time you set β (0–1) to control how much your current intent matters versus your long-term taste.

```
query_vector = normalize(β × intent_vector + (1−β) × taste_vector)
```

### The ranking pipeline

The query vector then drives a four-stage pipeline that puts both logics to work:

1. **Retrieval** — FAISS gathers unwatched candidates via cosine similarity over 768-dimensional sentence-transformer embeddings (`all-mpnet-base-v2`), from the query vector *and* from each taste profile. The profiles only widen recall — the query vector still governs relevance at scoring time — so off-intent candidates surface but score low.
2. **Debiased collaborative filtering** — each candidate that exists in MovieLens 32M is scored with a user vector projected into the SVD latent space via confidence-weighted least-squares fold-in, then **z-scored within the candidate set**. Raw CF scores carry a popularity offset (well-loved films score high for *everyone*); z-scoring removes that offset and turns CF into a relative "more/less for you than the pool average" signal. Films outside MovieLens simply get a neutral CF contribution — they compete on content instead of being buried at the bottom, which is where the old pipeline dumped every arthouse and post-2023 film.
3. **Blend** — content similarity (also z-scored across the pool) and the CF signal are combined, minus a **popularity penalty**, so obvious blockbusters don't dominate a cinephile's list:

   ```
   score = w_content · z(content) + w_cf · z(CF) − w_pop · popularity
   ```

   Both standardized signals are clipped to ±2.5 σ so a single heavy-tailed CF outlier can't swamp the blend (the failure mode that quietly re-created the mainstream bias). The weights are configurable.
4. **Diversify (MMR)** — the top slice is re-ranked with Maximal Marginal Relevance, trading each pick's score against its similarity to films already chosen, so you don't get five near-identical entries from the same franchise or director. Strength is set by `diversity`.

**Confidence weighting** — during SVD training, each rating is scaled by `log(1 + number_of_ratings_for_that_film)`, normalized so a median-popularity film keeps weight 1.0, so popular films' latent vectors are estimated more precisely. The same weights apply during fold-in. Note this only shapes *how the CF vectors are estimated* — the popularity penalty in stage 3 is what keeps popular films from dominating the final ranking.

**Honest scores** — results show a **Match** percentage (content similarity to your query), a **For you** arrow (the debiased CF nudge, ↑/↓ relative to the candidate pool), and the film's **Community** average. There is no fabricated "predicted rating" — the old star prediction was an uncalibrated dot product dressed up as `x/5`.

Every recommendation comes with a **grounded "Why"** — the film you love it's closest to and what they share (director, themes, or register), plus the collaborative signal and mood overlap. This is data-driven and instant, with no dependencies. If a local Ollama server is running it can write the blurb instead; the built-in explainer is used otherwise.

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
5. Embeds all films with `all-mpnet-base-v2` (768-dim) and saves a FAISS index
6. Trains SVD on the full ratings matrix

Expected time on first run: **5-15 minutes**, almost entirely the TMDB metadata fetch (~87k individual API calls at ~40 req/s). The per-film JSON cache means rerunning is instant.

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
6. Top-N recommendations are displayed with a **Match** %, a **For you** arrow, and the **Community** average.
7. Each pick includes a **Match %**, a **Fit** arrow (viewers with your taste), the **community** average, and a grounded **Why**.

On first run, resolving your Letterboxd history to TMDB ids is cached to `data/index/lb_resolved.json`, so subsequent runs skip the lookups.

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

## Explanations (the "Why" column)

By default the "Why" is written by a built-in, data-driven explainer — no dependencies, always on. It anchors each pick to the film you rated highly that it most resembles, and names what they share.

To have a local LLM write the blurbs instead, install and run [Ollama](https://ollama.com), pull a small model, and leave `provider: auto` (or force `ollama`):

```bash
ollama pull llama3.2      # any chat model works
ollama serve              # if not already running
```

```yaml
explanations:
  provider: auto          # auto | heuristic | ollama
  model:                  # optional; blank = first installed Ollama model
```

`auto` uses Ollama when a server answers on `:11434`, otherwise the built-in explainer. If you set `provider: ollama` but no server is reachable, the CLI says so and falls back — it never fails silently.

## Configuration reference

| Key | Default | Description |
|-----|---------|-------------|
| `letterboxd.username` | — | Your Letterboxd username |
| `letterboxd.export_dir` | — | Path to unzipped Letterboxd export (optional, recommended) |
| `tmdb.api_key` | — | TMDB API key (free) |
| `defaults.beta` | `0.5` | Intent vs taste balance (0 = pure taste, 1 = pure intent) |
| `defaults.gamma` | `0.6` | Reference films vs mood text (0 = pure mood, 1 = pure refs) |
| `defaults.top_n` | `10` | Number of results |
| `defaults.n_taste_profiles` | `6` | Max taste facets to cluster your likes into for retrieval |
| `defaults.diversity` | `0.3` | MMR re-rank strength (0 = off, ~0.5 = aggressively varied) |
| `defaults.candidate_pool` | `1000` | Films retrieved by content before ranking |
| `defaults.cf_retrieval` | `0` | Collaborative retrieval channel size (0 = off; see Evaluation) |
| `defaults.weights.content` | `1.0` | Weight of content relevance in the blend |
| `defaults.weights.cf` | `0.6` | Weight of the debiased collaborative-filtering signal |
| `defaults.weights.popularity` | `0.15` | Popularity penalty — raise to push harder toward non-obvious films |

## Evaluation

`scripts/evaluate.py` measures recommendation quality offline so ranking changes can be compared with numbers instead of eyeballed. It uses **leave-N-out**: from your Letterboxd likes it repeatedly hides a random handful, builds taste from the rest, ranks all unwatched candidates, and reports how often the hidden films resurface.

```bash
python scripts/evaluate.py                    # sweep preset ranking configs on the current index
python scripts/evaluate.py --index-dir data/index/backup-YYYYMMDD-HHMMSS   # A/B a different index
```

Metrics (averaged over folds × held-out films): **recall@K** (fraction of held-out likes in the top K) and **MRR** (mean reciprocal rank). This is how the `all-mpnet-base-v2` upgrade was validated (~2× recall@10 over the old `all-MiniLM-L6-v2` index) and how the collaborative retrieval channel was found *not* to help at ~120 ratings and left off by default. With a sparse rating history, absolute recall is low — content and CF can only predict so much from few examples.
| `explanations.provider` | `auto` | `auto` (Ollama if reachable, else built-in), `heuristic`, or `ollama` |
| `explanations.model` | — | Optional Ollama model name; blank = first installed |

## Project structure

```
next-film/
├── config.yaml
├── scripts/
│   ├── setup.py              # one-time: build FAISS index, train SVD
│   ├── reembed.py            # rebuild the index after a model/blob change (cache-only)
│   └── evaluate.py           # offline leave-N-out evaluation (recall@K, MRR)
├── src/
│   ├── cli.py                # main entrypoint
│   ├── scrapers/
│   │   └── letterboxd.py     # RSS parser + CSV export loader
│   ├── enrichment/
│   │   └── tmdb.py           # TMDB API client with JSON caching
│   ├── models/
│   │   ├── collaborative.py  # SVD training + confidence-weighted user fold-in
│   │   ├── embeddings.py     # sentence-transformer encoding + FAISS index
│   │   └── hybrid.py         # blended ranker: retrieval → debiased CF → blend → MMR
│   ├── query/
│   │   └── builder.py        # two-sided taste vector + taste profiles + query vector
│   ├── search/
│   │   └── film_search.py    # TMDB search + rapidfuzz fallback
│   └── llm/
│       └── explainer.py      # "Why" text: built-in data-driven + optional Ollama
└── data/                     # created by setup.py, not committed
    ├── movielens/            # ml-32m CSV files (place here manually)
    ├── index/                # FAISS index, SVD model, ID maps
    └── cache/                # TMDB API response cache (per-film JSON)
```

## Limitations

- Collaborative filtering coverage depends on MovieLens 32M (87,585 films, January 1995 – October 2023). Films released after October 2023 get a neutral CF contribution and are ranked on content relevance alone.
- First-run setup makes ~87k individual TMDB API calls. The per-film JSON cache means re-runs are fast.
- The content embedding is `all-mpnet-base-v2` (768-dim) over a **taste-forward text blob** — director, genre and theme lead; the plot is trimmed so it no longer dominates; the film's own title is excluded (a title is an identifier, not a descriptor, and embedding it causes lexical collisions — e.g. every unrelated film named "Parasite" clustering together). It is still a general-purpose semantic model, so "content similarity" approximates *sensibility* but doesn't capture it perfectly. Changing the model (set `content.model`) or the blob requires rebuilding the index with `python scripts/reembed.py` — a cache-only rebuild that backs up the previous index. The index records which model built it so the query encoder always matches.

## Citation

This project uses the MovieLens 32M dataset:

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems (TiiS)* 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872

## Dependencies

| Package | Purpose |
|---------|---------|
| `sentence-transformers` | `all-mpnet-base-v2` embeddings |
| `faiss-cpu` | Approximate nearest-neighbour search |
| `scikit-learn` | Randomized SVD |
| `scipy` | Sparse ratings matrix |
| `pandas` | MovieLens data loading |
| `rapidfuzz` | Fuzzy film title fallback |
| `requests` | TMDB API + Letterboxd RSS |
| `rich` | Terminal UI |
| `pyyaml` | Config parsing |
| `ollama` | Optional LLM explanations |
