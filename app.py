import streamlit as st

st.set_page_config(page_title="PDF Chatbot")
st.title("PDF Chatbot")

# Chat history survives reruns via session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar: upload and controls
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
    if pdf:
        st.success(f"Loaded: {pdf.name}")

# Show past messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# New question
if question := st.chat_input("Ask something about your PDF"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    answer = "Placeholder: the RAG answer will appear here."
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)