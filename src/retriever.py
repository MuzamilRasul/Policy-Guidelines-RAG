
import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class PolicyRetriever:
    """Semantic search over policy chunks using FAISS."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_folder: str | Path = ".cache/huggingface",
    ) -> None:
        self.model = SentenceTransformer(
            model_name,
            cache_folder=str(cache_folder),
        )
        self.index = None
        self.chunks: list[dict[str, Any]] = []

    def build_index(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            raise ValueError("Cannot build an index with no chunks.")

        texts = [chunk["text"] for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        embeddings = np.asarray(embeddings, dtype=np.float32)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.chunks = chunks

        print(f"FAISS index built with {len(chunks)} chunks.")

    def save_index(self, folder: str | Path = "storage") -> None:
        if self.index is None:
            raise RuntimeError("Build the index before saving.")

        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(folder / "policy.index"))

        with (folder / "chunks.json").open("w", encoding="utf-8") as file:
            json.dump(self.chunks, file, ensure_ascii=False, indent=2)

        print(f"Index and metadata saved to: {folder.resolve()}")

    def load_index(self, folder: str | Path = "storage") -> None:
        folder = Path(folder)
        index_path = folder / "policy.index"
        chunks_path = folder / "chunks.json"

        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                "Saved index files are missing. Build and save the index first."
            )

        self.index = faiss.read_index(str(index_path))

        with chunks_path.open("r", encoding="utf-8") as file:
            self.chunks = json.load(file)

        if self.index.ntotal != len(self.chunks):
            raise ValueError("Index and chunk metadata counts do not match.")

        print(f"Loaded {len(self.chunks)} chunks from: {folder.resolve()}")

    def search(
        self,
        question: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        if self.index is None:
            raise RuntimeError("Load or build the index before searching.")

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_embedding = self.model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        query_embedding = np.asarray(query_embedding, dtype=np.float32)

        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query_embedding, k)

        results = []

        for score, position in zip(scores[0], indices[0]):
            if position < 0:
                continue

            result = dict(self.chunks[int(position)])
            result["similarity"] = float(score)
            results.append(result)

        return results
