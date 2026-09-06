import os
from pathlib import Path

# Silence transformers logging before import, else it prints a load-report table.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import logging
import warnings

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path(__file__).parent.parent.parent / "data" / "index"

# The building model is persisted to embedding_model.txt: a mismatched encoder
# makes the geometry meaningless.
DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"
MODEL_FILE = INDEX_DIR / "embedding_model.txt"


def index_model_name() -> str:
    """The embedding model the FAISS index was built with (falls back to the default)."""
    return MODEL_FILE.read_text().strip() if MODEL_FILE.exists() else DEFAULT_MODEL


class EmbeddingModel:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or DEFAULT_MODEL
        try:
            from transformers.utils import logging as hf_logging
            hf_logging.set_verbosity_error()
        except Exception:
            pass
        self.model = SentenceTransformer(self.model_name)
        # get_embedding_dimension() is typed int | None in stubs; 768 for all-mpnet-base-v2
        self.dim: int = int(self.model.get_embedding_dimension() or 768)

    def encode(self, texts: list[str] | str) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        vecs = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.array(vecs, dtype=np.float32)


class FilmIndex:
    def __init__(self) -> None:
        self.index: faiss.IndexFlatIP | None = None
        self.id_map: list[int] = []         # position → tmdb_id
        self._pos_map: dict[int, int] = {}  # tmdb_id → position (O(1) lookup)

    def build(self, vectors: np.ndarray, tmdb_ids: list[int]) -> None:
        dim = int(vectors.shape[1])
        self.index = faiss.IndexFlatIP(dim)  # type: ignore[call-arg]
        self.index.add(vectors)              # type: ignore[call-arg]
        self.id_map = list(tmdb_ids)
        self._pos_map = {tid: pos for pos, tid in enumerate(self.id_map)}

    def save(self, name: str = "films", model_name: str | None = None) -> None:
        assert self.index is not None, "Index not built"
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(INDEX_DIR / f"{name}.faiss"))
        np.save(str(INDEX_DIR / f"{name}_ids.npy"), np.array(self.id_map))
        if model_name:
            MODEL_FILE.write_text(model_name)

    def load(self, name: str = "films", index_dir: Path | None = None) -> None:
        d = index_dir or INDEX_DIR
        self.index = faiss.read_index(str(d / f"{name}.faiss"))
        self.id_map = np.load(str(d / f"{name}_ids.npy")).tolist()
        self._pos_map = {tid: pos for pos, tid in enumerate(self.id_map)}

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        assert self.index is not None, "Index not loaded"
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)
        scores, positions = self.index.search(query_vector, k)  # type: ignore[call-arg]
        results = []
        for pos, score in zip(positions[0], scores[0]):
            if pos >= 0:
                results.append((self.id_map[pos], float(score)))
        return results

    def is_built(self) -> bool:
        return (INDEX_DIR / "films.faiss").exists()

    def has(self, tmdb_id: int) -> bool:
        return tmdb_id in self._pos_map

    def get_vector(self, tmdb_id: int) -> np.ndarray | None:
        assert self.index is not None, "Index not loaded"
        pos = self._pos_map.get(tmdb_id)
        if pos is None:
            return None
        vec = np.empty(self.index.d, dtype=np.float32)  # type: ignore[attr-defined]
        self.index.reconstruct(pos, vec)                 # type: ignore[call-arg]
        return vec

    def get_vectors(self, tmdb_ids: list[int]) -> tuple[np.ndarray, list[int]]:
        """Reconstruct vectors into one (N, d) matrix.

        Returns (matrix, kept_ids), in order, with ids absent from the index dropped.
        """
        assert self.index is not None, "Index not loaded"
        positions, kept = [], []
        for tid in tmdb_ids:
            pos = self._pos_map.get(tid)
            if pos is not None:
                positions.append(pos)
                kept.append(tid)
        out = np.empty((len(positions), self.index.d), dtype=np.float32)  # type: ignore[attr-defined]
        for row, pos in enumerate(positions):
            self.index.reconstruct(pos, out[row])  # type: ignore[call-arg]
        return out, kept


def embed_and_index(
    embedder: "EmbeddingModel",
    film_index: FilmIndex,
    ids: list[int],
    texts: list[str],
    batch_size: int = 256,
) -> None:
    """Encode `texts` in batches, build the index and save it."""
    from rich.progress import track

    vecs = [
        embedder.encode(texts[i: i + batch_size])
        for i in track(range(0, len(texts), batch_size), description="Embedding")
    ]
    film_index.build(np.vstack(vecs), ids)
    film_index.save(model_name=embedder.model_name)
