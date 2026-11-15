import re
from pypdf import PdfReader
import numpy as np
from sentence_transformers import SentenceTransformer
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

    
_MODEL = None


def get_model(name="all-MiniLM-L6-v2"):
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(name)
    return _MODEL


def embed_chunks(chunks):
    texts = [c["text"] for c in chunks]
    return get_model().encode(texts, normalize_embeddings=True)


def retrieve(question, chunks, chunk_emb, k=4):
    """Return the k chunks most similar to the question, with scores."""
    q = get_model().encode([question], normalize_embeddings=True)[0]
    scores = chunk_emb @ q
    top = np.argsort(scores)[::-1][:k]
    return [{**chunks[i], "score": float(scores[i])} for i in top]