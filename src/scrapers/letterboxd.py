import csv
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import requests


@dataclass
class WatchedFilm:
    title: str
    year: int
    rating: float  # 0.5–5.0, or 0.0 if unrated


def fetch_watched(username: str) -> list[WatchedFilm]:
    """Fetch via Letterboxd RSS — returns only the ~50 most recent diary entries."""
    url = f"https://letterboxd.com/{username}/rss/"
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    return _parse_rss(response.text)


def load_from_export(export_dir: str | Path) -> list[WatchedFilm]:
    """Load full watch history from a Letterboxd CSV export directory.

    Download your export at letterboxd.com → Settings → Data → Export Your Data.
    The zip contains ratings.csv (rated films) and watched.csv (all watched films).
    Both are read and merged; ratings.csv takes precedence for rating values.
    """
    export_dir = Path(export_dir)
    ratings_path = export_dir / "ratings.csv"
    watched_path = export_dir / "watched.csv"

    films: dict[tuple[str, int], WatchedFilm] = {}

    # watched.csv: Date, Name, Year, Letterboxd URI
    if watched_path.exists():
        with watched_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                title = row.get("Name", "").strip()
                year = _parse_year(row.get("Year", ""))
                if title:
                    films[(title, year)] = WatchedFilm(title=title, year=year, rating=0.0)

    # ratings.csv: Date, Name, Year, Letterboxd URI, Rating
    if ratings_path.exists():
        with ratings_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                title = row.get("Name", "").strip()
                year = _parse_year(row.get("Year", ""))
                rating = _parse_rating_str(row.get("Rating", ""))
                if title:
                    films[(title, year)] = WatchedFilm(title=title, year=year, rating=rating)

    return list(films.values())


def _parse_year(raw: str) -> int:
    try:
        return int(raw.strip())
    except ValueError:
        return 0


def _parse_rating_str(raw: str) -> float:
    try:
        return float(raw.strip())
    except ValueError:
        return 0.0


def _parse_rss(xml_text: str) -> list[WatchedFilm]:
    root = ET.fromstring(xml_text)
    films = []
    for item in root.findall(".//item"):
        title_el = item.find("title")
        if title_el is None or title_el.text is None:
            continue
        raw_title = title_el.text.strip()
        title, year = _parse_title_year(raw_title)
        if title is None:
            continue
        rating = _parse_rss_rating(item)
        films.append(WatchedFilm(title=title, year=year, rating=rating))
    return films


def _parse_title_year(raw: str) -> tuple[str | None, int]:
    match = re.match(r"^(.+),\s*(\d{4})", raw)
    if match:
        return match.group(1).strip(), int(match.group(2))
    return None, 0


def _parse_rss_rating(item: ET.Element) -> float:
    rating_el = item.find("{https://a.letterboxd.com/ns/1.0/}memberRating")
    if rating_el is not None and rating_el.text:
        try:
            return float(rating_el.text)
        except ValueError:
            pass
    return 0.0
