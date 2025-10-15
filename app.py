# app.py
import os
import streamlit as st
from dotenv import load_dotenv
import openai
import PyPDF2
import tempfile
import numpy as np
from datetime import datetime
import math

# ---- Config ----
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

st.set_page_config(page_title="Knowledge-base Search Engine", layout="wide")
st.title("Knowledge-base Search Engine 🔎 ")
st.markdown(
    """
Upload PDF / TXT documents, the app creates embeddings, and answers user queries by retrieving the most relevant text chunks and asking the LLM to synthesize a concise answer.
**Educational/demo only** — verify outputs before using in production.
"""
)

if not OPENAI_API_KEY:
    st.sidebar.error("OPENAI_API_KEY not set. Add a .env file or set environment variable.")

# ---- Utilities ----
def extract_text_from_pdf(file_bytes):
    try:
        reader = PyPDF2.PdfReader(file_bytes)
        text_pages = []
        for p in range(len(reader.pages)):
            page = reader.pages[p]
            text_pages.append(page.extract_text() or "")
        return "\n".join(text_pages)
    except Exception as e:
        return ""

def chunk_text(text, chunk_size=800, overlap=100):
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
        if start < 0:
            start = 0
    return [c for c in chunks if c]

def embed_texts(texts, model="text-embedding-3-small"):
    # returns list of vectors (floats)
    resp = openai.Embedding.create(model=model, input=texts)
    return [r["embedding"] for r in resp["data"]]

def cosine_similarity_matrix(a, b):
    # a: (n, d) array, b: (m, d) array -> (n, m) similarity
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    # avoid div by zero
    a_norm[a_norm==0] = 1e-12
    b_norm[b_norm==0] = 1e-12
    sim = (a @ b.T) / (a_norm * b_norm.T)
    return sim

def top_k_indices(sim_scores, k=5):
    # sim_scores: 1D array
    idx = np.argsort(-sim_scores)[:k]
    return idx

def build_qa_prompt(query, contexts):
    joined = "\n\n---\n\n".join([f"Source {i+1}:\n{c}" for i, c in enumerate(contexts)])
    prompt = f"""
You are an assistant that answers user questions using ONLY the provided source excerpts. Provide a short, factual, and well-structured answer to the user's question. If the information is not present in the sources, say you don't have enough information and suggest where to look or what clarifying info is needed.

User question:
\"\"\"{query}\"\"\"

Sources:
{joined}

Instructions:
- Use the sources to support your answer. Quote or reference the source number when appropriate.
- Keep the answer concise (3-6 sentences).
- If the sources disagree, summarize both viewpoints and say the disagreement exists.
- Provide a short list of citations like: [Source 1], [Source 3].
- Do NOT hallucinate facts outside the sources.
- End with a one-line suggestion for next steps (e.g., "Check the original document or ask for clarification").
"""
    return prompt

def synthesize_answer(prompt, model="gpt-4o-mini", temperature=0.0, max_tokens=450):
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions using provided sources. Do not hallucinate."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
    )
    return resp.choices[0].message.content.strip()

# ---- Persistence in session ----
if "docs" not in st.session_state:
    st.session_state.docs = []          # list of dicts: {name, text}
if "chunks" not in st.session_state:
    st.session_state.chunks = []        # list of chunk texts
if "embeddings" not in st.session_state:
    st.session_state.embeddings = []    # list of vectors
if "meta" not in st.session_state:
    st.session_state.meta = []          # parallel metadata for chunks (doc name, chunk id)
if "index_time" not in st.session_state:
    st.session_state.index_time = None

# ---- Upload UI ----
st.sidebar.header("Ingest documents")
uploaded_files = st.sidebar.file_uploader("Upload PDF or TXT files (multiple)", accept_multiple_files=True, type=["pdf", "txt"])
chunk_size = st.sidebar.number_input("Chunk size (chars)", min_value=200, max_value=2000, value=800, step=100)
overlap = st.sidebar.number_input("Chunk overlap (chars)", min_value=0, max_value=500, value=100, step=50)
embed_model = st.sidebar.selectbox("Embedding model", ["text-embedding-3-small", "text-embedding-3-large"], index=0)
llm_model = st.sidebar.selectbox("Response model", ["gpt-4o-mini","gpt-3.5-turbo"], index=0)
top_k = st.sidebar.slider("Context passages (top k)", 1, 10, 4)
temp = st.sidebar.slider("LLM temperature", 0.0, 1.0, 0.0, 0.05)
ingest_btn = st.sidebar.button("Ingest & Build Index")

# ---- Ingestion ----
if ingest_btn and uploaded_files:
    new_docs = []
    for f in uploaded_files:
        raw = None
        if f.type == "application/pdf" or f.name.lower().endswith(".pdf"):
            raw = extract_text_from_pdf(f)
        elif f.type.startswith("text") or f.name.lower().endswith(".txt"):
            raw = f.getvalue().decode("utf-8")
        else:
            raw = ""
        if not raw:
            st.warning(f"No text extracted from {f.name}")
            continue
        new_docs.append({"name": f.name, "text": raw})
    if not new_docs:
        st.error("No valid text extracted from uploaded files.")
    else:
        # chunk all new docs and embed
        all_new_chunks = []
        meta = []
        for d in new_docs:
            doc_chunks = chunk_text(d["text"], chunk_size=int(chunk_size), overlap=int(overlap))
            for i, c in enumerate(doc_chunks):
                all_new_chunks.append(c)
                meta.append({"doc": d["name"], "chunk_id": i})
        if not all_new_chunks:
            st.error("No chunks produced (maybe files empty).")
        else:
            with st.spinner("Creating embeddings (may take time for many chunks)..."):
                try:
                    vectors = embed_texts(all_new_chunks, model=embed_model)
                except Exception as e:
                    st.error(f"Embedding error: {e}")
                    vectors = []
            # store
            st.session_state.docs.extend(new_docs)
            base_index = len(st.session_state.chunks)
            st.session_state.chunks.extend(all_new_chunks)
            st.session_state.embeddings.extend(vectors)
            st.session_state.meta.extend(meta)
            st.session_state.index_time = datetime.utcnow().isoformat() + "Z"
            st.success(f"Ingested {len(new_docs)} files, {len(all_new_chunks)} chunks added.")
else:
    if st.session_state.index_time:
        st.sidebar.markdown(f"Index last built: **{st.session_state.index_time}**")
    else:
        st.sidebar.info("No index built yet. Upload files and click 'Ingest & Build Index'.")

# ---- Query UI ----
st.subheader("Ask a question")
query = st.text_input("Enter your question about the uploaded documents:")
query_btn = st.button("Search & Answer")

if query_btn:
    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY not configured.")
    elif not query.strip():
        st.warning("Please enter a question.")
    elif not st.session_state.chunks:
        st.warning("No documents indexed. Upload and ingest documents first.")
    else:
        with st.spinner("Running retrieval and synthesis..."):
            try:
                q_vec = embed_texts([query], model=embed_model)[0]
                # compute similarities
                sims = cosine_similarity_matrix(np.array(st.session_state.embeddings), np.array([q_vec]))[:,0]
                top_idx = top_k_indices(sims, k=top_k)
                contexts = [st.session_state.chunks[i] for i in top_idx]
                contexts_meta = [st.session_state.meta[i] for i in top_idx]
                # build prompt
                prompt = build_qa_prompt(query, contexts)
                answer = synthesize_answer(prompt, model=llm_model, temperature=float(temp))
                # show results
                st.markdown("### Answer")
                st.write(answer)
                st.markdown("### Retrieved passages (context snippets used)")
                for i, (c, m) in enumerate(zip(contexts, contexts_meta)):
                    st.markdown(f"**Source {i+1}** — `{m['doc']}` — chunk {m['chunk_id']}")
                    st.write(c[:800] + ("..." if len(c) > 800 else ""))
            except Exception as e:
                st.error(f"Error during retrieval/synthesis: {e}")

# ---- Show indexed docs summary ----
st.markdown("---")
st.subheader("Indexed documents & stats")
colA, colB = st.columns(2)
with colA:
    st.write(f"Documents ingested: **{len(st.session_state.docs)}**")
    for d in st.session_state.docs:
        st.write("- ", d["name"])
with colB:
    st.write(f"Total chunks: **{len(st.session_state.chunks)}**")
    st.write(f"Embedding dim: **{len(st.session_state.embeddings[0]) if st.session_state.embeddings else 'N/A'}**")

st.markdown("---")
st.caption("This is a demo RAG implementation (in-memory). For production, persist embeddings and use a vector DB (FAISS, Milvus, Pinecone). Avoid exposing API keys and add authentication.")
