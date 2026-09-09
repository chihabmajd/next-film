# next-film

A movie recommender over a Letterboxd history: content embeddings for retrieval,
collaborative filtering for re-ranking, combined into a single score.

No recommendation library is used. The truncated SVD relies on scikit-learn's
`randomized_svd` and the content vectors come from a pretrained sentence-transformer;
the user fold-in, the popularity debiasing and the MMR re-ranking are derived and
implemented here.

## How it works

Two signals, covering each other's blind spot:

- **Content** — each film is embedded from its metadata (director, genres, themes,
  trimmed plot) with `all-mpnet-base-v2`. Works for any film with metadata, including
  obscure and recent ones, but only sees topic and style.
- **Collaborative** — SVD over 32M MovieLens ratings. Captures cross-genre taste
  correlations that text cannot, but only for films with enough ratings.

The query vector blends long-term taste with current intent:

```
query = normalize(β · intent + (1 − β) · taste)
```

Taste is two-sided (films rated above your average pull toward them, films below push
away, down-weighted) and clustered into a few facets rather than averaged into one
centroid. Intent is reference films, a free-text mood, or both, mixed by γ.

Ranking, in four stages:

1. **Retrieval** — FAISS cosine search from the query vector and from each taste facet.
2. **Collaborative score** — a user vector is projected into the latent space by
   confidence-weighted regularized least squares, then z-scored within the candidate set
   to strip the popularity offset. Films absent from MovieLens contribute 0.
3. **Blend** — `score = w_c · z(content) + w_cf · z(CF) − w_pop · popularity`, both
   standardized terms clipped to ±2.5σ.
4. **Diversify** — MMR re-rank against already-picked films.

Results show a match percentage, the collaborative nudge, the community average, and a
grounded explanation anchored on the film you rated highly that the pick most resembles.

## Setup

Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download [MovieLens 32M](https://grouplens.org/datasets/movielens/32m/) and place
`links.csv`, `movies.csv`, `ratings.csv`, `tags.csv` in `data/movielens/`.

Copy `config.example.yaml` to `config.yaml` and set your Letterboxd username and a
[TMDB API key](https://www.themoviedb.org/settings/api).

```bash
python scripts/setup.py     # builds metadata, embeds the catalog, trains the SVD
python src/cli.py
```

Setup takes 5–15 minutes on first run, almost entirely the TMDB metadata fetch. Each
step is skipped on rerun. The Letterboxd RSS feed only returns the last ~50 entries; for
full history, export your data (Settings → Data → Export) and set `letterboxd.export_dir`.

## Configuration

| Key | Default | Description |
|-----|---------|-------------|
| `defaults.beta` | `0.5` | Intent vs taste (0 = pure taste, 1 = pure intent) |
| `defaults.gamma` | `0.6` | Reference films vs mood text |
| `defaults.top_n` | `10` | Number of results |
| `defaults.n_taste_profiles` | `6` | Taste facets used for retrieval |
| `defaults.diversity` | `0.3` | MMR strength (0 = off) |
| `defaults.candidate_pool` | `1000` | Candidates retrieved before ranking |
| `defaults.cf_retrieval` | `0` | Collaborative retrieval channel (0 = off) |
| `defaults.weights.{content,cf,popularity}` | `1.0 / 0.6 / 0.15` | Blend weights |
| `content.model` | `all-mpnet-base-v2` | Changing this requires `scripts/reembed.py` |
| `explanations.provider` | `auto` | `auto`, `heuristic`, or `ollama` |

## Limitations

- MovieLens 32M covers 87,585 films to October 2023. Later releases are ranked on
  content alone.
- First-run setup makes ~87k TMDB calls; responses are cached per film.
- Film titles are excluded from the embedded text, since a title is an identifier rather
  than a descriptor and embedding it collides unrelated films that share a name.

## Layout

```
scripts/   setup.py, reembed.py, evaluate.py
src/       cli.py, scrapers/, enrichment/, models/, query/, search/, llm/
data/      movielens/, index/, cache/   (not committed)
```

## Citation

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and
Context. *ACM TiiS* 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872
