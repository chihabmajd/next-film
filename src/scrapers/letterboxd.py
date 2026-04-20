import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import requests


@dataclass
class WatchedFilm:
    title: str
    year: int
    rating: float  # 0.5–5.0, or 0.0 if unrated


def fetch_watched(username: str) -> list[WatchedFilm]:
    url = f"https://letterboxd.com/{username}/rss/"
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    return _parse_rss(response.text)


def _parse_rss(xml_text: str) -> list[WatchedFilm]:
    root = ET.fromstring(xml_text)

    films = []
    for item in root.findall(".//item"):
        title_el = item.find("title")
        if title_el is None or title_el.text is None:
            continue

        # title format: "Film Name, YYYY - rating★"
        raw_title = title_el.text.strip()
        title, year = _parse_title_year(raw_title)
        if title is None:
            continue

        rating = _parse_rating(item)
        films.append(WatchedFilm(title=title, year=year, rating=rating))

    return films


def _parse_title_year(raw: str) -> tuple[str | None, int]:
    # Letterboxd RSS title: "Film Name, YYYY"
    match = re.match(r"^(.+),\s*(\d{4})", raw)
    if match:
        return match.group(1).strip(), int(match.group(2))
    return None, 0


def _parse_rating(item: ET.Element) -> float:
    # Letterboxd uses <letterboxd:memberRating> in the RSS
    rating_el = item.find("{https://a.letterboxd.com/ns/1.0/}memberRating")
    if rating_el is not None and rating_el.text:
        try:
            return float(rating_el.text)
        except ValueError:
            pass
    return 0.0
