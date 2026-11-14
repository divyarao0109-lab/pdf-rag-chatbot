import io
import streamlit as st
from rag import extract_pages, chunk_pages

st.set_page_config(page_title="PDF Chatbot")
st.title("PDF Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_data(show_spinner="Reading PDF...")
def process(pdf_bytes, chunk_size, chunk_overlap):
    pages = extract_pages(io.BytesIO(pdf_bytes))
    return pages, chunk_pages(pages, chunk_size, chunk_overlap)


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

chunks = []
if pdf:
    pages, chunks = process(pdf.getvalue(), chunk_size, chunk_overlap)
    if not pages:
        st.error("No text found. This may be a scanned PDF (images only).")
    else:
        st.sidebar.success(f"{pdf.name}: {len(pages)} pages, {len(chunks)} chunks")
        with st.expander("Preview chunks"):
            for c in chunks[:5]:
                st.caption(f"Page {c['page']}")
                st.text(c["text"])

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if question := st.chat_input("Ask something about your PDF"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    answer = "Placeholder: the RAG answer will appear here."
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)