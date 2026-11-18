import pymupdf as fitz  # PyMuPDF


def load_pdf(path):
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    return pages


def chunk_pages(pages, chunk_size=150, overlap=30, source="doc.pdf"):
    chunks = []
    step = chunk_size - overlap
    for p in pages:
        words = p["text"].split()
        for start in range(0, len(words), step):
            piece = words[start:start + chunk_size]
            if len(piece) < 20:
                continue
            chunks.append({"text": " ".join(piece), "page": p["page"], "source": source})
    return chunks