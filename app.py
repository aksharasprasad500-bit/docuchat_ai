"""
DocuChat AI — Chat with your PDF using Retrieval-Augmented Generation (RAG)

Pipeline:
1. Extract text from an uploaded PDF
2. Split text into overlapping chunks
3. Embed chunks locally with SentenceTransformers (all-MiniLM-L6-v2)
4. Store embeddings in a FAISS vector index for fast similarity search
5. On each question: embed the question, retrieve top-k relevant chunks,
   pass them as context to an LLM (via Groq's free/fast API) to generate
   a grounded answer.

Run:
    pip install -r requirements.txt
    export GROQ_API_KEY="your_key_here"   # free key from https://console.groq.com
    streamlit run app.py
"""

import os
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq

st.set_page_config(page_title="DocuChat AI", page_icon="📄", layout="wide")

# ---------------------------------------------------------------------------
# Cached resources (loaded once per session, not on every rerun)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY") or st.session_state.get("groq_api_key")
    if not api_key:
        return None
    return Groq(api_key=api_key)


# ---------------------------------------------------------------------------
# Core RAG functions
# ---------------------------------------------------------------------------

def extract_text_from_pdf(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150):
    """Split text into overlapping chunks so context isn't cut mid-idea."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def build_faiss_index(chunks, embedder):
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings


def retrieve_relevant_chunks(question, chunks, index, embedder, top_k=4):
    q_embedding = embedder.encode([question]).astype("float32")
    distances, indices = index.search(q_embedding, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]


def generate_answer(question, context_chunks, client):
    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"""You are a strict document-QA assistant. You must answer using ONLY the
context below, extracted from a user-uploaded document.

Rules:
1. First, check whether the context actually contains information that answers the question.
2. If it does NOT, respond with EXACTLY this and nothing else:
   "I couldn't find that in the document. This document appears to be about: [give a one-line
   description of what the retrieved context actually covers]."
3. If it DOES contain the answer, answer clearly and concisely using only that context.
4. Never use outside knowledge. Never guess. Never pad a weak match into a full answer.

Context:
{context}

Question: {question}"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.title("📄 DocuChat AI")
st.caption("Upload a PDF and ask questions about it — powered by RAG (retrieval-augmented generation)")

with st.sidebar:
    st.header("⚙️ Setup")
    if not os.environ.get("GROQ_API_KEY"):
        key_input = st.text_input("Groq API Key", type="password",
                                   help="Get a free key at console.groq.com")
        if key_input:
            st.session_state["groq_api_key"] = key_input
    else:
        st.success("Groq API key loaded from environment")

    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown(
        "1. PDF text is extracted & chunked\n"
        "2. Chunks are embedded locally (MiniLM)\n"
        "3. Your question retrieves the top matching chunks (FAISS)\n"
        "4. An LLM answers using only that retrieved context"
    )

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    if "processed_file" not in st.session_state or st.session_state["processed_file"] != uploaded_file.name:
        with st.spinner("Reading and indexing document..."):
            embedder = load_embedder()
            raw_text = extract_text_from_pdf(uploaded_file)

            if not raw_text.strip():
                st.error("Couldn't extract text from this PDF (it may be a scanned image). Try another file.")
                st.stop()

            chunks = chunk_text(raw_text)
            index, _ = build_faiss_index(chunks, embedder)

            st.session_state["chunks"] = chunks
            st.session_state["index"] = index
            st.session_state["processed_file"] = uploaded_file.name
            st.session_state["chat_history"] = []

        st.success(f"Indexed {len(st.session_state['chunks'])} chunks from '{uploaded_file.name}'")

    # Chat history
    for msg in st.session_state.get("chat_history", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a question about the document...")

    if question:
        st.session_state["chat_history"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        client = get_groq_client()
        if not client:
            st.error("Please enter your Groq API key in the sidebar (free at console.groq.com).")
            st.stop()

        embedder = load_embedder()
        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant context and generating answer..."):
                relevant_chunks = retrieve_relevant_chunks(
                    question, st.session_state["chunks"], st.session_state["index"], embedder
                )
                answer = generate_answer(question, relevant_chunks, client)
                st.markdown(answer)

                with st.expander("🔍 View retrieved context (what the model used)"):
                    for i, chunk in enumerate(relevant_chunks, 1):
                        st.markdown(f"**Chunk {i}:** {chunk[:300]}...")

        st.session_state["chat_history"].append({"role": "assistant", "content": answer})
else:
    st.info("👆 Upload a PDF to get started. Try a resume, research paper, or any document you have handy.")