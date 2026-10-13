
import re
from typing import Any


def chunk_documents(
    documents: list[dict[str, Any]],
    chunk_size: int = 700,
    overlap: int = 1,
) -> list[dict[str, Any]]:
    """
    Create sentence-aware chunks while preserving source metadata.

    chunk_size: approximate maximum characters per chunk.
    overlap: number of sentences carried into the next chunk.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    chunks: list[dict[str, Any]] = []

    for document in documents:
        text = document.get("text", "").strip()
        if not text:
            continue

        # Split on sentence-ending punctuation followed by whitespace.
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", text)
            if sentence.strip()
        ]

        if not sentences:
            continue

        current: list[str] = []

        def save_chunk(sentences_to_save: list[str]) -> None:
            chunk_text = " ".join(sentences_to_save).strip()
            if chunk_text:
                chunks.append({
                    "chunk_id": len(chunks),
                    "source": document.get("source", "unknown"),
                    "page": document.get("page"),
                    "text": chunk_text,
                })

        for sentence in sentences:
            # Flush the current chunk before adding a sentence that
            # would make it too large.
            candidate = " ".join(current + [sentence])

            if current and len(candidate) > chunk_size:
                save_chunk(current)

                # Carry forward a few complete sentences as overlap.
                current = current[-overlap:] if overlap else []

            current.append(sentence)

            # Keep a very long individual sentence intact rather than
            # cutting it in the middle.
            if len(" ".join(current)) >= chunk_size:
                save_chunk(current)
                current = current[-overlap:] if overlap else []

        if current:
            save_chunk(current)

    print(f"Created {len(chunks)} text chunks.")
    return chunks

