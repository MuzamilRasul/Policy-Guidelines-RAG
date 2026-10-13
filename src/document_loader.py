
from pathlib import Path
from typing import Any

import pymupdf


def load_pdf(file_path: str | Path) -> list[dict[str, Any]]:
    """
    Extract text from a PDF while preserving its source and page number.

    Returns one record per page containing:
    - source: PDF filename
    - page: 1-based page number
    - text: extracted page text
    """
    path = Path(file_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, received: {path.suffix}")

    pages: list[dict[str, Any]] = []

    with pymupdf.open(path) as document:
        if document.is_encrypted:
            raise ValueError(f"The PDF is password-protected: {path.name}")

        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            # Skip blank pages, which can occur in some documents.
            if not text:
                continue

            pages.append(
                {
                    "source": path.name,
                    "page": page_number,
                    "text": text,
                }
            )

    if not pages:
        raise ValueError(
            f"No extractable text found in {path.name}. "
            "It may be scanned and require OCR."
        )

    return pages


def load_documents(folder_path: str | Path) -> list[dict[str, Any]]:
    """Load all PDF documents from a folder."""
    folder = Path(folder_path)

    if not folder.exists() or not folder.is_dir():
        raise NotADirectoryError(f"Invalid document folder: {folder}")

    pdf_files = sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {folder.resolve()}. "
            "Add at least one policy or guidelines PDF."
        )

    all_pages: list[dict[str, Any]] = []
    errors: list[str] = []

    for pdf_file in pdf_files:
        try:
            pages = load_pdf(pdf_file)
            all_pages.extend(pages)
            print(f"Loaded {pdf_file.name}: {len(pages)} text pages")
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append(f"{pdf_file.name}: {exc}")

    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise RuntimeError(
            "Some PDFs could not be loaded:\n" + details
        )

    print(f"Total extracted pages: {len(all_pages)}")
    return all_pages
