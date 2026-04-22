import json
import time
from pathlib import Path
from dataclasses import dataclass, field

import requests


CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache"
BASE_URL = "https://api.themoviedb.org/3"


@dataclass
class FilmMetadata:
    tmdb_id: int
    title: str
    year: int
    genres: list[str] = field(default_factory=list)
    plot: str = ""
    director: str = ""
    cast: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    def to_text_blob(self) -> str:
        parts = [f"{self.title} ({self.year})."]
        if self.director:
            parts.append(f"Directed by {self.director}.")
        if self.plot:
            parts.append(self.plot)
        if self.genres:
            parts.append(f"Genres: {', '.join(self.genres)}.")
        if self.keywords:
            parts.append(f"Keywords: {', '.join(self.keywords[:15])}.")
        if self.cast:
            parts.append(f"Cast: {', '.join(self.cast[:5])}.")
        return " ".join(parts)


class TMDBClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def search(self, title: str, year: int | None = None) -> list[dict]:
        params: dict = {"query": title, "language": "en-US"}
        if year:
            params["year"] = year
        data = self._get("/search/movie", params)
        return data.get("results", [])

    def get_metadata(self, tmdb_id: int) -> FilmMetadata | None:
        cache_path = CACHE_DIR / f"{tmdb_id}.json"
        if cache_path.exists():
            return self._from_cache(cache_path)

        details = self._get(f"/movie/{tmdb_id}", {"append_to_response": "credits,keywords"})
        if not details:
            return None

        metadata = self._parse(details)
        cache_path.write_text(json.dumps(details, ensure_ascii=False))
        return metadata

    def get_metadata_batch(self, tmdb_ids: list[int], delay: float = 0.025) -> dict[int, FilmMetadata]:
        results = {}
        for tmdb_id in tmdb_ids:
            cached = (CACHE_DIR / f"{tmdb_id}.json").exists()
            meta = self.get_metadata(tmdb_id)
            if meta:
                results[tmdb_id] = meta
            if not cached:
                time.sleep(delay)  # only rate-limit actual HTTP requests
        return results

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = BASE_URL + path
        p = {"api_key": self.api_key, **(params or {})}
        for attempt in range(5):
            try:
                response = requests.get(url, params=p, timeout=15)
            except requests.RequestException:
                if attempt == 4:
                    return {}
                time.sleep(2 ** attempt)
                continue
            if response.status_code == 404:
                return {}
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 10))
                time.sleep(retry_after)
                continue
            response.raise_for_status()
            return response.json()
        return {}

    def _parse(self, data: dict) -> FilmMetadata:
        year = 0
        release = data.get("release_date", "")
        if release:
            try:
                year = int(release[:4])
            except ValueError:
                pass

        credits = data.get("credits", {})
        director = ""
        for crew in credits.get("crew", []):
            if crew.get("job") == "Director":
                director = crew.get("name", "")
                break

        cast = [m["name"] for m in credits.get("cast", [])[:5] if "name" in m]
        genres = [g["name"] for g in data.get("genres", [])]
        keywords = [k["name"] for k in data.get("keywords", {}).get("keywords", [])]

        return FilmMetadata(
            tmdb_id=data["id"],
            title=data.get("title", ""),
            year=year,
            genres=genres,
            plot=data.get("overview", ""),
            director=director,
            cast=cast,
            keywords=keywords,
        )

    def _from_cache(self, path: Path) -> FilmMetadata:
        data = json.loads(path.read_text())
        return self._parse(data)
