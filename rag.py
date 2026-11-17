import re
import numpy as np
import ollama
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Embedding settings -------------------------------------------------
EMBEDDING_BACKEND = "ollama"            # "ollama" or "sbert"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
SBERT_MODEL = "all-MiniLM-L6-v2"
_SBERT = None


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


def embed_texts(texts, kind="document"):
    """Return a (len(texts), dim) array of unit-length embeddings."""
    global _SBERT
    if EMBEDDING_BACKEND == "sbert":
        if _SBERT is None:
            from sentence_transformers import SentenceTransformer
            _SBERT = SentenceTransformer(SBERT_MODEL)
        return _SBERT.encode(texts, normalize_embeddings=True)

    # nomic-embed-text works best with these task prefixes
    prefix = "search_document: " if kind == "document" else "search_query: "
    resp = ollama.embed(model=OLLAMA_EMBED_MODEL, input=[prefix + t for t in texts])
    arr = np.array(resp["embeddings"], dtype="float32")
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def embed_chunks(chunks):
    return embed_texts([c["text"] for c in chunks], kind="document")


def retrieve(question, chunks, chunk_emb, k=4):
    """Return the k chunks most similar to the question, with scores."""
    q = embed_texts([question], kind="query")[0]
    scores = chunk_emb @ q
    top = np.argsort(scores)[::-1][:k]
    return [{**chunks[i], "score": float(scores[i])} for i in top]


SYSTEM_PROMPT = (
    "You are an assistant that answers questions about a PDF document. "
    "Use ONLY the context provided. "
    "If the answer is not in the context, reply exactly: "
    "'I could not find this in the document.' "
    "Cite the page number(s) you used, like (Page 3)."
)


def build_prompt(question, hits):
    context = "\n\n".join(f"[Page {h['page']}]\n{h['text']}" for h in hits)
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


def stream_answer(question, hits, temperature=0.2, model="llama3.2"):
    """Yield the answer piece by piece so Streamlit can show it as it is written."""
    stream = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, hits)},
        ],
        options={"temperature": temperature, "num_ctx": 4096},
        stream=True,
    )
    for part in stream:
        yield part["message"]["content"]


def spread_chunks(chunks, n=8):
    """For 'summarize' questions: take chunks spread evenly across the document."""
    step = max(1, len(chunks) // n)
    return [{**c, "score": 1.0} for c in chunks[::step][:n]]