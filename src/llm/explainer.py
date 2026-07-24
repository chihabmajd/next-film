"""Explanations for why each film was recommended.

Two backends:
  * "heuristic" (default) — data-driven, instant, no dependencies. It grounds each pick in
    *your* history: the film you love it's closest to, what they share (director / themes /
    register), the collaborative signal, and mood overlap. Always available.
  * "ollama" — a local LLM writes the blurb, used only when an Ollama server is actually
    reachable. Never silently swallowed: if it's unreachable we say so and fall back.

Provider "auto" uses Ollama when reachable, otherwise the heuristic.
"""

from __future__ import annotations

import urllib.request

import numpy as np

from src.enrichment.tmdb import FilmMetadata


def ollama_available(host: str = "http://localhost:11434", timeout: float = 0.6) -> tuple[bool, list[str]]:
    """Return (reachable, [model names]) without importing or blocking for long."""
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as r:
            import json
            data = json.loads(r.read().decode())
        models = [m.get("name", "") for m in data.get("models", [])]
        return True, models
    except Exception:
        return False, []


def _stars(rating: float) -> str:
    full = int(rating)
    half = "½" if rating - full >= 0.5 else ""
    return f"{full}{half}★"


class Explainer:
    def __init__(
        self,
        provider: str = "auto",
        model: str | None = None,
        liked: dict[int, tuple[FilmMetadata, float, np.ndarray]] | None = None,
    ) -> None:
        """
        liked: {tmdb_id: (metadata, your_rating, embedding_vector)} for films you rated highly —
               the material the heuristic explainer grounds its reasons in.
        """
        self.provider = provider
        self.model = model
        self._ollama = None
        self.active = "heuristic"          # the backend actually in use: "heuristic" or "ollama"
        self.requested_but_unavailable = False  # True iff provider=ollama but no server answered

        # Stack the liked films' vectors once so _closest_liked is a single matmul, not a loop.
        self._liked = list((liked or {}).values())  # [(FilmMetadata, rating, vector)]
        self._liked_matrix = (
            np.stack([vec for _, _, vec in self._liked]) if self._liked else None
        )

        if provider in ("auto", "ollama"):
            reachable, models = ollama_available()
            if reachable and models:
                self.model = model if (model and model in models) else models[0]
                self.active = "ollama"
            elif provider == "ollama":
                self.requested_but_unavailable = True

    # ---- public -------------------------------------------------------------------------
    def explain(self, film: FilmMetadata, vec: np.ndarray | None, mood_text: str | None) -> str:
        if self.active == "ollama":
            out = self._ollama_explain(film, mood_text)
            if out:
                return out
        return self._heuristic_explain(film, vec, mood_text)

    # ---- heuristic ----------------------------------------------------------------------
    def _heuristic_explain(self, film: FilmMetadata, vec: np.ndarray | None, mood_text: str | None) -> str:
        anchor = self._closest_liked(vec)
        bits: list[str] = []

        if anchor is not None:
            meta, rating = anchor
            shared_dir = film.director and film.director == meta.director
            shared_kw = self._overlap(film.keywords, meta.keywords)
            shared_genre = self._overlap(film.genres, meta.genres)
            lead = f"Closest to your {_stars(rating)} [italic]{meta.title}[/italic]"
            if shared_dir:
                bits.append(f"{lead} — both directed by {film.director}.")
            elif shared_kw:
                bits.append(f"{lead} — shared threads: {', '.join(shared_kw[:3])}.")
            elif shared_genre:
                bits.append(f"{lead} — same register: {', '.join(shared_genre[:2])}.")
            else:
                bits.append(f"{lead}.")
        elif film.director:
            bits.append(f"A {', '.join(film.genres[:2])} film by {film.director}.")
        elif film.genres:
            bits.append(f"{', '.join(film.genres[:2])}.")

        if mood_text:
            mood_hits = self._mood_overlap(film, mood_text)
            if mood_hits:
                bits.append(f"Fits your mood via {', '.join(mood_hits[:2])}.")

        return " ".join(bits) if bits else "Strong match for your query."

    def _closest_liked(self, vec: np.ndarray | None) -> tuple[FilmMetadata, float] | None:
        if vec is None or self._liked_matrix is None:
            return None
        i = int(np.argmax(self._liked_matrix @ vec))
        meta, rating, _ = self._liked[i]
        return meta, rating

    @staticmethod
    def _overlap(a: list[str], b: list[str]) -> list[str]:
        bl = {x.lower() for x in b}
        seen, out = set(), []
        for x in a:
            xl = x.lower()
            if xl in bl and xl not in seen:
                seen.add(xl)
                out.append(x)
        return out

    @staticmethod
    def _mood_overlap(film: FilmMetadata, mood_text: str) -> list[str]:
        words = {w.strip(".,!?;:").lower() for w in mood_text.split() if len(w) > 3}
        hits = []
        for kw in film.keywords + film.genres:
            if any(w in kw.lower() or kw.lower() in w for w in words):
                hits.append(kw)
        return hits

    # ---- ollama -------------------------------------------------------------------------
    def _ollama_explain(self, film: FilmMetadata, mood_text: str | None) -> str | None:
        try:
            if self._ollama is None:
                import ollama
                self._ollama = ollama
            liked_titles = ", ".join(m.title for m, _, _ in self._liked[:6])
            prompt = (
                "In one vivid sentence, tell the user why they'll like this film. "
                f"Films they love: {liked_titles}. "
                f"{'Mood: ' + mood_text + '. ' if mood_text else ''}"
                f"Recommendation: {film.to_text_blob()}"
            )
            resp = self._ollama.chat(
                model=self.model, messages=[{"role": "user", "content": prompt}]
            )
            return resp["message"]["content"].strip().replace("\n", " ")
        except Exception:
            return None
