# utils/preprocessing.py

import re
import pypdf


def extract_text_from_pdf(uploaded_file) -> list[dict]:
    """Extract text page-by-page from a PDF file."""
    pdf_reader = pypdf.PdfReader(uploaded_file)
    pages = []
    for i, page in enumerate(pdf_reader.pages):
        pages.append({
            "page": i + 1,
            "text": page.extract_text() or "",
        })
    return pages


def normalize_encoding(text: str) -> str:
    """Normalize to UTF-8, dropping undecodable bytes."""
    return text.encode("utf-8", "ignore").decode("utf-8")


def clean_text(text: str) -> str:
    """
    Basic noise reduction:
    - Collapse multiple whitespace/newlines into a single space
    - Strip leading/trailing whitespace
    """
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """
    Split text into overlapping chunks by word boundary.
    Uses a sliding window so context isn't lost at chunk edges.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(words):
            break
        start += chunk_size - overlap
    return chunks


def preprocess_pdf(uploaded_file, doc_name: str | None = None) -> list[dict]:
    """
    Full pipeline: extract → normalize → clean → chunk.

    Returns a list of chunk dicts:
        { "content": str, "page": int }

    doc_name is accepted for forward-compatibility but metadata
    is attached at the vector-store level, not here.
    """
    pages = extract_text_from_pdf(uploaded_file)
    all_chunks = []

    for page_data in pages:
        normalized = normalize_encoding(page_data["text"])
        cleaned    = clean_text(normalized)

        if not cleaned:
            continue

        chunks = chunk_text(cleaned)
        for chunk in chunks:
            all_chunks.append({
                "content": chunk,
                "page":    page_data["page"],
            })

    return all_chunks