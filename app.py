# PDF chatbot: RAG pipeline + Streamlit web app in one file
# Run with:  python -m streamlit run app.py


# Importing the required packages

import io  # Reading the uploaded PDF from memory
import re  # Cleaning the extracted text
import numpy as np  # Numerical computations
import ollama  # Talking to the local embedding model and the LLM
import streamlit as st  # Building the web app
from pypdf import PdfReader  # Extracting the text from the PDF
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Splitting the text into chunks
from sklearn.cluster import KMeans  # K-Means clustering algorithm

# This section imports all the necessary libraries for the chatbot.
# It includes tools for reading the PDF (pypdf), splitting the text (langchain),
# creating embeddings and answers (ollama), clustering the chunks (KMeans)
# and building the web page (streamlit).


# Setting the parameters

app_name = "PDF Chatbot"  # change after you choose a name
embedding_backend = "ollama"  # "ollama" or "sbert"
ollama_embed_model = "nomic-embed-text"
sbert_model = "all-MiniLM-L6-v2"
llm_model = "llama3.1"  # Run "ollama pull llama3.1" once before using it
random_state = 42  # Seed to get the same clusters every time
ollama_hint = "Could not reach the language model. Is Ollama running?"
sbert_embedder = None  # Loaded only when the sbert backend is used


# Loading the PDF and splitting it into chunks

# Reading the text of every page of the PDF, 'file' is a path or a file-like object

def extract_pages(file):

    reader = PdfReader(file)
    pages = []

    for i, page in enumerate(reader.pages, start = 1):
        text = page.extract_text() or ""  # Pages with no text give an empty string
        text = re.sub(r"[ \t]+", " ", text)  # Collapse repeated spaces
        text = re.sub(r"\n{3,}", "\n\n", text)  # Collapse blank lines
        text = text.strip()

        if text:  # Skipping the empty pages
            pages.append({"page": i, "text": text})

    return pages


# Splitting each page into chunks, the page number is kept for the citations

def chunk_pages(pages, chunk_size = 1000, chunk_overlap = 200):

    # Splitting the text into chunks of the given size
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size, chunk_overlap = chunk_overlap
    )

    chunks = []
    for p in pages:
        for piece in text_splitter.split_text(p["text"]):
            chunks.append({"text": piece, "page": p["page"]})  # Page number is kept for the citations

    return chunks


# Creating the embeddings and searching the chunks

# Converting the texts into an array of unit-length embeddings of shape (len(texts), dim)

def embed_texts(texts, kind = "document", batch_size = 32):

    global sbert_embedder

    if embedding_backend == "sbert":
        if sbert_embedder is None:
            from sentence_transformers import SentenceTransformer

            sbert_embedder = SentenceTransformer(sbert_model)
        return sbert_embedder.encode(texts, normalize_embeddings = True)

    # nomic-embed-text works best with these task prefixes
    prefix = "search_document: " if kind == "document" else "search_query: "

    vectors = []
    for i in range(0, len(texts), batch_size):  # Embedding the texts in small batches
        batch = [prefix + t for t in texts[i : i + batch_size]]
        vectors.extend(ollama.embed(model = ollama_embed_model, input = batch)["embeddings"])

    embeddings = np.array(vectors, dtype = "float32")

    # Scaling every vector to unit length, so that the dot product gives the cosine similarity
    return embeddings / np.linalg.norm(embeddings, axis = 1, keepdims = True)


# Creating the embedding of every chunk

def embed_chunks(chunks):

    return embed_texts([c["text"] for c in chunks], kind = "document")


# Returning the k chunks most similar to the question, along with their scores

def retrieve(question, chunks, chunk_embeddings, k = 6):

    question_embedding = embed_texts([question], kind = "query")[0]

    # Cosine similarity between the question and every chunk
    scores = chunk_embeddings @ question_embedding

    # Positions of the k highest scores
    top_idxs = np.argsort(scores)[::-1][:k]

    return [{**chunks[i], "score": float(scores[i])} for i in top_idxs]


# Returning the chunks which are relevant to the question, but different from each other (MMR)

def retrieve_mmr(question, chunks, chunk_embeddings, k = 6, fetch_k = 15, lambda_mult = 0.5):

    question_embedding = embed_texts([question], kind = "query")[0]
    similarities = chunk_embeddings @ question_embedding

    # Starting with the fetch_k most similar chunks, and picking the best one first
    candidates = list(np.argsort(similarities)[::-1][:fetch_k])
    selected = [candidates.pop(0)]

    while candidates and len(selected) < k:
        best_idx, best_score = None, -1e9

        for c in candidates:
            # How similar the candidate is to the chunks already selected
            redundancy = max(chunk_embeddings[c] @ chunk_embeddings[s] for s in selected)
            score = lambda_mult * similarities[c] - (1 - lambda_mult) * redundancy

            if score > best_score:
                best_idx, best_score = c, score

        selected.append(best_idx)
        candidates.remove(best_idx)

    return [{**chunks[i], "score": float(similarities[i])} for i in selected]


# Summarizing the whole document using K-Means clustering

# K-Means splits the n chunks into K clusters (basically, centroids) by minimizing the
# within cluster sum-of-squares (euclidean distance between the centroid and the
# corresponding chunks of a cluster). Instead of giving the whole document to the LLM,
# only the chunk closest to the centre of each cluster is given.

# Returning the chunk(s) closest to the centre of each cluster, in the order of the document

def get_cluster_chunks(chunks, chunk_embeddings, n_clusters = 10, num_closest = 1):

    # A small document does not need clustering, so all the chunks are used
    if len(chunks) <= n_clusters * num_closest:
        return [{**c, "score": 1.0} for c in chunks]

    # Creating a KMeans clustering model with the given number of clusters
    kmeans_model = KMeans(n_clusters = n_clusters, random_state = random_state, n_init = 10)
    kmeans_model.fit(chunk_embeddings)

    # Cluster label (0 to n_clusters - 1) of every chunk
    cluster_labels = kmeans_model.predict(chunk_embeddings)

    selected_idxs = []
    for cluster_no in range(n_clusters):
        # Positions of the chunks which belong to this cluster
        member_idxs = np.where(cluster_labels == cluster_no)[0]

        # Distance between the centroid of the cluster and each chunk of the cluster
        distances = np.linalg.norm(
            chunk_embeddings[member_idxs] - kmeans_model.cluster_centers_[cluster_no], axis = 1
        )

        # The chunks with the smallest distances are the closest to the centroid
        closest_idxs = member_idxs[np.argsort(distances)[:num_closest]]
        selected_idxs.extend(closest_idxs.tolist())

    # Keeping the selected chunks in the original order of the document
    selected_idxs = sorted(set(selected_idxs))

    print(f"Number of Clusters: {n_clusters}, Chunks selected for the summary: {len(selected_idxs)}")

    # Score 1.0 tells show_hits() not to print a similarity value
    return [{**chunks[i], "score": 1.0} for i in selected_idxs]


# Prompts and LLM generation

qa_system = (
    "You are an assistant that answers questions about a PDF document. "
    "Use ONLY the context provided. "
    "If the answer is not in the context, reply exactly: "
    "'I could not find this in the document.' "
    "Cite the page number(s) you used, like (Page 3)."
)

summary_system = (
    "You are given key excerpts selected from across a long document, in "
    "document order, each labelled with its page number. Write a clear, "
    "coherent summary of the whole document in 6 to 10 bullet points, "
    "mentioning page numbers where helpful, like (Page 3). "
    "Use only the given excerpts."
)

summary_words = ("summar", "overview", "main points", "key points", "tl;dr", "gist")

greeting_words = (
    "hello", "hi", "hii", "hey", "namaste", "good morning", "good afternoon",
    "good evening", "how are you", "thanks", "thank you", "bye", "goodbye",
)


# Checking whether the question asks for a summary of the whole document

def is_summary_request(question):

    q = question.lower()
    return any(w in q for w in summary_words)  # True if any summary word is present


# Checking whether the message is only a conversation starter like "hello"

def is_greeting(question):

    q = re.sub(r"[^a-z ]", "", question.lower()).strip()  # Removing the punctuation

    # Only short messages are greetings, so a real question starting with "hi" is not missed
    return len(q.split()) <= 4 and any(q == w or q.startswith(w + " ") for w in greeting_words)


# Replying to the conversation starter without searching the PDF

def get_greeting_reply(question, has_pdf):

    q = question.lower()

    if "thank" in q:
        return "You're welcome! Ask me anything else about your document."

    if "bye" in q:
        return "Goodbye! Come back whenever you want to chat about a PDF."

    if has_pdf:
        return "Hello! Your PDF is ready. Ask me a question about it, or ask me to summarize it."

    return "Hello! I'm your PDF assistant. Upload a PDF in the sidebar, then ask me questions about it or ask for a summary."


# Joining the chunks and the question into one prompt

def build_prompt(question, hits):

    context = "\n\n".join(f"[Page {h['page']}]\n{h['text']}" for h in hits)
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


# Answering the question, yielding the answer piece by piece so that Streamlit can show it as it is written

def stream_answer(question, hits, temperature = 0.2, model = None):

    stream = ollama.chat(
        model = model or llm_model,
        messages = [
            {"role": "system", "content": qa_system},
            {"role": "user", "content": build_prompt(question, hits)},
        ],
        options = {"temperature": temperature, "num_ctx": 8192},
        stream = True,
    )

    for part in stream:
        yield part["message"]["content"]


# Giving all the selected chunks to the LLM in one prompt, and yielding the summary piece by piece

def stream_summary(hits, temperature = 0.2, model = None, max_chars = 14000):

    # Joining the selected chunks with their page numbers
    body = "\n\n".join(f"[Page {h['page']}]\n{h['text']}" for h in hits)[:max_chars]

    stream = ollama.chat(
        model = model or llm_model,
        messages = [
            {"role": "system", "content": summary_system},
            {"role": "user", "content": f"Excerpts:\n{body}\n\nSummary:"},
        ],
        options = {"temperature": temperature, "num_ctx": 8192},
        stream = True,
    )

    for part in stream:
        yield part["message"]["content"]


# Streamlit web app

# Reading the PDF, creating the chunks and their embeddings (the result is cached by Streamlit)

@st.cache_data(show_spinner = "Reading and embedding the PDF...")
def process(pdf_bytes, chunk_size, chunk_overlap):

    pages = extract_pages(io.BytesIO(pdf_bytes))
    chunks = chunk_pages(pages, chunk_size, chunk_overlap)

    print(f"Number of Pages: {len(pages)}, Number of Chunks: {len(chunks)}")

    chunk_embeddings = embed_chunks(chunks) if chunks else None

    return pages, chunks, chunk_embeddings


# Displaying the citations, i.e. the chunks (and their pages) which were used for the answer

def show_hits(hits, enabled = True):

    if not hits or not enabled:
        return

    # Page labels in ascending order of the page number
    labels = sorted({str(h["page"]) for h in hits}, key = lambda s: int(s.split("-")[0]))

    with st.expander("Citations: pages " + ", ".join(labels)):
        for h in hits:
            score = "" if h["score"] >= 1.0 else f" | similarity {h['score']:.2f}"
            st.caption(f"Page {h['page']}{score}")
            st.text(h["text"])


def main():

    st.set_page_config(page_title = app_name, page_icon = "📄")
    st.title(app_name)
    st.caption("Ask questions about your PDF. Answers use only the document and cite page numbers.")

    # Creating the chat history when the app starts
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar with the document upload and the parameters
    with st.sidebar:
        st.header("Document")
        pdf = st.file_uploader("Upload a PDF", type = "pdf")
        stats = st.empty()

        with st.expander("Chunking"):
            chunk_size = st.slider("Chunk size", 200, 2000, 1000, 100)
            chunk_overlap = st.slider("Chunk overlap", 0, 500, 200, 50)

        with st.expander("Retrieval"):
            search_mode = st.selectbox("Method", ["Similarity", "MMR"])
            top_k = st.slider("Top-k chunks", 1, 10, 6)
            min_score = st.slider("Min similarity", 0.0, 0.8, 0.30, 0.05)

        with st.expander("Summary (k-means)"):
            n_clusters = st.slider("Clusters", 3, 20, 10)
            num_closest = st.slider("Chunks per cluster", 1, 3, 1)

        with st.expander("Generation"):
            temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
            show_citations = st.toggle("Show citations", value = True)

        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

    # A new PDF means a new conversation
    if pdf and st.session_state.get("pdf_name") != pdf.name:
        st.session_state.pdf_name = pdf.name
        st.session_state.messages = []

    # Reading the PDF, creating the chunks and their embeddings
    pages, chunks, chunk_embeddings = [], [], None
    if pdf:
        pages, chunks, chunk_embeddings = process(pdf.getvalue(), chunk_size, chunk_overlap)

        if not pages:
            st.error("No text found. This may be a scanned PDF (images only).")
        else:
            with stats.container():
                c1, c2 = st.columns(2)
                c1.metric("Pages", len(pages))
                c2.metric("Chunks", len(chunks))

    if not chunks:
        st.info("Upload a PDF in the sidebar to start chatting.")

    # Displaying the earlier messages of the conversation
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            show_hits(m.get("hits", []), show_citations)

    # Getting the question from the user
    if question := st.chat_input("Ask something about your PDF"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            hits = []

            if is_greeting(question) and not is_summary_request(question):
                # Conversation starters like "hello" are answered without searching the PDF
                answer = get_greeting_reply(question, len(chunks) > 0)
                st.markdown(answer)

            elif not chunks:
                answer = "Please upload a PDF first."
                st.markdown(answer)

            elif is_summary_request(question):
                # Summary of the whole document using K-Means clustering
                try:
                    with st.spinner("Clustering the document..."):
                        hits = get_cluster_chunks(chunks, chunk_embeddings, n_clusters, num_closest)
                    answer = st.write_stream(stream_summary(hits, temperature))
                except Exception as e:
                    hits = []
                    answer = f"{ollama_hint} ({e})"
                    st.error(answer)
                show_hits(hits, show_citations)

            else:
                # Answer to the question using the most similar chunks
                if search_mode == "MMR":
                    hits = retrieve_mmr(question, chunks, chunk_embeddings, top_k)
                else:
                    hits = retrieve(question, chunks, chunk_embeddings, top_k)

                if hits[0]["score"] < min_score:
                    answer = "I could not find this in the document."
                    st.markdown(answer)
                else:
                    try:
                        answer = st.write_stream(stream_answer(question, hits, temperature))
                    except Exception as e:
                        answer = f"{ollama_hint} ({e})"
                        st.error(answer)
                show_hits(hits, show_citations)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "hits": hits}
        )


if __name__ == "__main__":
    main()