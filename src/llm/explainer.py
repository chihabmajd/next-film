from src.enrichment.tmdb import FilmMetadata


class LLMExplainer:
    def __init__(self, model: str = "llama3"):
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            import ollama
            self._client = ollama
        return self._client

    def explain(
        self,
        film: FilmMetadata,
        reference_titles: list[str],
        mood_text: str | None,
    ) -> str | None:
        try:
            client = self._get_client()
        except ImportError:
            return None

        context_parts = []
        if reference_titles:
            context_parts.append(f"Reference films the user likes: {', '.join(reference_titles)}.")
        if mood_text:
            context_parts.append(f"Mood: {mood_text}.")

        prompt = (
            f"In 2 sentences, explain why '{film.title} ({film.year})' is a good recommendation. "
            f"{' '.join(context_parts)} "
            f"Film info: {film.to_text_blob()}"
        )

        try:
            response = client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response["message"]["content"].strip()
        except Exception:
            return None
