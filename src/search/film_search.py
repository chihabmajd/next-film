from dataclasses import dataclass

from rapidfuzz import process, fuzz

from src.enrichment.tmdb import TMDBClient


@dataclass
class FilmMatch:
    tmdb_id: int
    title: str
    year: int
    director: str


class FilmSearcher:
    def __init__(self, client: TMDBClient, local_titles: dict[int, str] | None = None):
        self.client = client
        # local_titles: {tmdb_id: "Title (YYYY)"} for rapidfuzz fallback
        self.local_titles = local_titles or {}

    def search(self, query: str) -> list[FilmMatch]:
        results = self.client.search(query)
        if results:
            return self._to_matches(results[:4])

        # rapidfuzz fallback against local index
        if self.local_titles:
            fuzzy_hits = process.extract(
                query,
                self.local_titles,
                scorer=fuzz.WRatio,
                score_cutoff=80,
                limit=4,
            )
            matches = []
            for _, _, tmdb_id in fuzzy_hits:
                meta = self.client.get_metadata(tmdb_id)
                if meta:
                    matches.append(FilmMatch(
                        tmdb_id=meta.tmdb_id,
                        title=meta.title,
                        year=meta.year,
                        director=meta.director,
                    ))
            return matches

        return []

    def _to_matches(self, results: list[dict]) -> list[FilmMatch]:
        matches = []
        for r in results:
            year = int(r.get("release_date", "0000")[:4]) if r.get("release_date") else 0
            matches.append(FilmMatch(
                tmdb_id=r["id"],
                title=r.get("title", ""),
                year=year,
                director="",
            ))
        return matches
