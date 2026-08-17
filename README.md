# DocuChat AI — Chat with your PDF (RAG)

A Retrieval-Augmented Generation (RAG) app that lets you upload any PDF and ask
questions about it in natural language. Answers are grounded in the document's
actual content — not hallucinated — because the LLM only sees the specific
chunks retrieved as relevant to your question.

## How it works

1. **Extract** — Text is pulled from the uploaded PDF (`PyPDF2`).
2. **Chunk** — Text is split into overlapping ~800-character chunks so context
   isn't cut mid-sentence.
3. **Embed** — Each chunk is converted into a vector using a local
   sentence-transformer model (`all-MiniLM-L6-v2`) — no API cost for this step.
4. **Index** — Vectors are stored in a `FAISS` index for fast similarity search.
5. **Retrieve** — When you ask a question, it's embedded too, and the top-k most
   similar chunks are pulled from the index.
6. **Generate** — Those chunks + your question are sent to an LLM (Groq's
   `llama-3.1-8b-instant`, free tier, very low latency) which answers using
   only that retrieved context.

This is the same core architecture used in production RAG systems (customer
support bots, internal knowledge-base search, legal/medical document Q&A).

## Setup 

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Get a free Groq API key: https://console.groq.com  (no credit card needed)
export GROQ_API_KEY="your_key_here"     # Mac/Linux
# set GROQ_API_KEY=your_key_here        # Windows cmd

# 3. Run the app
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`. Upload a PDF and
start asking questions.

## Deploying

Push this repo to GitHub, then deploy for free on
[Streamlit Community Cloud](https://streamlit.io/cloud):
1. Connect your GitHub repo
2. Set `GROQ_API_KEY` under app secrets
3. Deploy — you'll get a public URL to put on your resume/LinkedIn

## Tech stack

`Python` · `Streamlit` · `FAISS` (vector search) · `Sentence-Transformers`
(embeddings) · `Groq API` (LLM inference) · `PyPDF2`

## Possible extensions 

- Support multiple file formats (docx, txt, web pages)
- Persist the FAISS index to disk so you don't re-embed on every restart
- Add source citations with page numbers
- Swap FAISS for a hosted vector DB (Pinecone/Chroma) for multi-user scale
- Add conversation memory so follow-up questions understand prior context

---

### Resume bullet points 

- *Built a RAG-based document Q&A application using Python, FAISS, and the
  Groq LLM API, enabling users to query PDF content in natural language with
  answers grounded in retrieved context.*
- *Designed and implemented a full retrieval pipeline (text chunking, local
  embedding generation, vector similarity search, LLM-based answer synthesis)
  and deployed it as an interactive Streamlit web app.*
