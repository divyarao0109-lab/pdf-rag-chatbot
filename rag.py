import re
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_pages(file):
    """Return [{'page': n, 'text': ...}, ...]. 'file' is a path or file-like object."""
    reader = PdfReader(file)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"[ \t]+", " ", text)        # collapse repeated spaces
        text = re.sub(r"\n{3,}", "\n\n", text)     # collapse blank lines
        text = text.strip()
        if text:
            pages.append({"page": i, "text": text})
    return pages


def chunk_pages(pages, chunk_size=1000, chunk_overlap=200):
    """Split each page into chunks, keeping the page number for citations."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = []
    for p in pages:
        for piece in splitter.split_text(p["text"]):
            chunks.append({"text": piece, "page": p["page"]})
    return chunks