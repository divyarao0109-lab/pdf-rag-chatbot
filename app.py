import io
import streamlit as st
from rag import extract_pages, chunk_pages, embed_chunks, retrieve

st.set_page_config(page_title="PDF Chatbot")
st.title("PDF Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_data(show_spinner="Reading and embedding the PDF...")
def process(pdf_bytes, chunk_size, chunk_overlap):
    pages = extract_pages(io.BytesIO(pdf_bytes))
    chunks = chunk_pages(pages, chunk_size, chunk_overlap)
    emb = embed_chunks(chunks) if chunks else None
    return pages, chunks, emb


def show_hits(hits):
    for h in hits:
        with st.expander(f"Page {h['page']} | similarity {h['score']:.2f}"):
            st.text(h["text"])


with st.sidebar:
    st.header("Settings")
    pdf = st.file_uploader("Upload a PDF", type="pdf")
    chunk_size = st.slider("Chunk size", 200, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk overlap", 0, 500, 200, 50)
    top_k = st.slider("Top-k chunks", 1, 10, 4)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

pages, chunks, emb = [], [], None
if pdf:
    pages, chunks, emb = process(pdf.getvalue(), chunk_size, chunk_overlap)
    if not pages:
        st.error("No text found. This may be a scanned PDF (images only).")
    else:
        st.sidebar.success(f"{pdf.name}: {len(pages)} pages, {len(chunks)} chunks")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        show_hits(m.get("hits", []))

if question := st.chat_input("Ask something about your PDF"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    if not chunks:
        answer, hits = "Please upload a PDF first.", []
    else:
        hits = retrieve(question, chunks, emb, top_k)
        answer = f"Top {len(hits)} matching sections (the LLM answer comes in Phase 7):"

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "hits": hits}
    )
    with st.chat_message("assistant"):
        st.markdown(answer)
        show_hits(hits)