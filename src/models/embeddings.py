from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path(__file__).parent.parent.parent / "data" / "index"
MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingModel:
    def __init__(self) -> None:
        self.model = SentenceTransformer(MODEL_NAME)
        # get_embedding_dimension() is typed int | None in stubs; default 384 for all-MiniLM-L6-v2
        self.dim: int = int(self.model.get_embedding_dimension() or 384)

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

    def save(self, name: str = "films") -> None:
        assert self.index is not None, "Index not built"
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(INDEX_DIR / f"{name}.faiss"))
        np.save(str(INDEX_DIR / f"{name}_ids.npy"), np.array(self.id_map))

    def load(self, name: str = "films") -> None:
        self.index = faiss.read_index(str(INDEX_DIR / f"{name}.faiss"))
        self.id_map = np.load(str(INDEX_DIR / f"{name}_ids.npy")).tolist()
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

    def get_vector(self, tmdb_id: int) -> np.ndarray | None:
        assert self.index is not None, "Index not loaded"
        pos = self._pos_map.get(tmdb_id)
        if pos is None:
            return None
        vec = np.empty(self.index.d, dtype=np.float32)  # type: ignore[attr-defined]
        self.index.reconstruct(pos, vec)                 # type: ignore[call-arg]
        return vec
