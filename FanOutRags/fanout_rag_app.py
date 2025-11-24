import os
import uuid
import json
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv

import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec

# ========= LOAD ENV =========
load_dotenv()

# ========= CONFIG =========
EMBEDDING_MODEL = "text-embedding-004"
CHAT_MODEL = "gemini-2.0-flash"   # or "gemini-1.5-flash"
PINECONE_INDEX_DIM = 768          # "text-embedding-004" outputs 768-d vectors
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

# Defaults for fan-out RAG params (overridable via UI)
DEFAULT_FANOUT_K = 5
DEFAULT_TOP_K_PER_SUBQ = 4
DEFAULT_MAX_UNIQUE_CHUNKS = 12
MAX_CONTEXT_CHARS = 18000
PARALLEL_RETRIEVAL = True

# ========= HELPERS =========
@st.cache_resource
def get_gemini_client():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set")
    genai.configure(api_key=api_key)
    return genai

@st.cache_resource
def get_pinecone_index():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY not set")
    index_name = os.getenv("PINECONE_INDEX_NAME", "pdf-rag-index")
    pc = Pinecone(api_key=api_key)
    # Create index if missing
    if index_name not in [i["name"] for i in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=PINECONE_INDEX_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    return pc.Index(index_name)

def extract_text_from_pdf(file_bytes) -> str:
    reader = PdfReader(file_bytes)
    return "\n".join([page.extract_text() or "" for page in reader.pages])

def chunk_text(text: str, size=1000, overlap=200) -> List[str]:
    # normalize whitespace and split into overlapping windows
    text = " ".join(text.replace("\n", " ").split())
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += max(1, size - overlap)
    return chunks

def embed_batch(genai_client, texts: List[str]) -> List[List[float]]:
    # sequential to avoid rate-limit surprises; thread this if needed
    embeddings = []
    for t in texts:
        resp = genai_client.embed_content(model=EMBEDDING_MODEL, content=t)
        embeddings.append(resp["embedding"])
    return embeddings

def hash_text(s: str) -> str:
    return hashlib.sha256((" ".join(s.split())).encode("utf-8")).hexdigest()

def truncate_contexts_by_chars(contexts: List[str], max_chars: int) -> List[str]:
    total = 0
    result = []
    for c in contexts:
        if total + len(c) > max_chars:
            break
        result.append(c)
        total += len(c)
    return result

def build_system_prompt() -> str:
    # Strong system prompt for RAG with fan-out queries and strict grounding rules.
    return (
        "You are a focused RAG assistant. Follow these rules strictly:\n"
        "1) Answer ONLY using the provided CONTEXT chunks. If the answer is not present, say: Not found in document.\n"
        "2) Be concise and precise. Prefer bullet points if listing items.\n"
        "3) Quote short, relevant snippets from the context when helpful.\n"
        "4) Do not invent information or speculate.\n"
        "5) If multiple chunks conflict, say so and summarize the discrepancy.\n"
        "6) If the user asks for information beyond the context, respond with: Not found in document.\n"
        "7) If the user’s query is ambiguous, ask a brief clarifying question before answering.\n\n"
        "Examples:\n\n"
        "User: What is the warranty period for the XYZ Pro?\n"
        "Context snippets:\n"
        "- Chunk A: 'The XYZ Pro includes a standard 12-month warranty covering manufacturing defects.'\n"
        "- Chunk B: 'Warranty claims must be submitted within the warranty period via the service portal.'\n"
        "Assistant: The warranty period for the XYZ Pro is 12 months. (Source: 'standard 12-month warranty')\n\n"
        "User: Does this document specify EU safety certifications for the product?\n"
        "Context snippets:\n"
        "- Chunk A: 'This manual covers installation and maintenance.'\n"
        "Assistant: Not found in document.\n\n"
        "User: Are there any differences between Model A and Model B regarding battery life?\n"
        "Context snippets:\n"
        "- Chunk A: 'Model A: up to 8 hours typical use.'\n"
        "- Chunk B: 'Model B: up to 10 hours typical use.'\n"
        "Assistant: Yes. Model A offers up to 8 hours, while Model B offers up to 10 hours of typical use. "
        "(Sources: 'Model A: up to 8 hours', 'Model B: up to 10 hours')\n"
    )

def build_user_prompt(user_question: str, unique_contexts: List[str]) -> str:
    context_text = "\n\n---\n\n".join(unique_contexts)
    return (
        f"CONTEXT (use only this information):\n{context_text}\n\n"
        f"QUESTION:\n{user_question}\n\n"
        "Return a helpful answer grounded in the context. If not present, say: Not found in document."
    )

def generate_subqueries(genai_client, question: str, k: int = 5) -> List[str]:
    # Produce a compact JSON array of sub-queries
    sys_msg = (
        "You generate 3-8 short, search-friendly sub-queries that help retrieve relevant passages. "
        "Rules: Return only a JSON array of strings. Keep each sub-query under 15 words. "
        "Do not add explanations or trailing text."
    )
    prompt = f"{sys_msg}\n\nUser question: {question}\n\nNumber of sub-queries requested: {k}"
    model = genai.GenerativeModel(model_name=CHAT_MODEL)
    resp = model.generate_content(prompt)
    text = (resp.text or "").strip()

    subs: List[str] = []
    try:
        if text.startswith("```"):
            stripped = text.strip("`")
            first_newline = stripped.find("\n")
            text = stripped[first_newline + 1 :].strip() if first_newline != -1 else stripped.strip()
        parsed = json.loads(text)
        if isinstance(parsed, list):
            subs = [str(s).strip() for s in parsed if str(s).strip()]
    except Exception:
        pass

    if not subs:
        candidates = [p.strip("-• \t") for p in text.split("\n") if p.strip()]
        if not candidates:
            candidates = [question]
        subs = candidates[:k]

    # Deduplicate and ensure original question is included
    dedup = []
    seen = set()
    for s in [question] + subs:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            dedup.append(s)
        if len(dedup) >= max(1, k):
            break
    return dedup

def pinecone_query(index, vector: List[float], top_k: int, doc_id_filter: Optional[str]) -> List[Dict[str, Any]]:
    # Build Pinecone metadata filter correctly
    flt = {"doc_id": doc_id_filter} if doc_id_filter else None
    # If you prefer operator style, you can use: flt = {"doc_id": {"\$eq": doc_id_filter}}
    res = index.query(vector=vector, top_k=top_k, include_metadata=True, filter=flt)
    out = []
    for m in res.matches or []:
        out.append({
            "id": m["id"],
            "score": m["score"],
            "text": m["metadata"].get("text", ""),
            "doc_id": m["metadata"].get("doc_id")
        })
    return out

def fanout_retrieve_unique(
    genai_client,
    index,
    user_question: str,
    doc_id_filter: Optional[str],
    fanout_k: int,
    top_k_per_subq: int,
    max_unique_chunks: int,
) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    # 1) expand sub-queries
    subqs = generate_subqueries(genai_client, user_question, fanout_k)

    # 2) embed each sub-query
    sub_embeddings = embed_batch(genai_client, subqs)

    # 3) query Pinecone per sub-query (parallel optional)
    results: List[Dict[str, Any]] = []
    if PARALLEL_RETRIEVAL:
        with ThreadPoolExecutor(max_workers=min(8, len(subqs))) as ex:
            futs = [
                ex.submit(pinecone_query, index, sub_embeddings[i], top_k_per_subq, doc_id_filter)
                for i in range(len(subqs))
            ]
            for f in as_completed(futs):
                try:
                    results.extend(f.result())
                except Exception as e:
                    st.warning(f"Retrieval error: {e}")
    else:
        for i in range(len(subqs)):
            results.extend(pinecone_query(index, sub_embeddings[i], top_k_per_subq, doc_id_filter))

    # 4) dedupe by text hash; keep best-scoring first
    results_sorted = sorted(results, key=lambda r: r.get("score", 0.0), reverse=True)
    unique_texts = []
    seen_hashes = set()
    for r in results_sorted:
        t = r.get("text", "")
        h = hash_text(t)
        if t and h not in seen_hashes:
            unique_texts.append(t)
            seen_hashes.add(h)
        if len(unique_texts) >= max_unique_chunks:
            break

    # 5) trim by char budget
    unique_texts = truncate_contexts_by_chars(unique_texts, MAX_CONTEXT_CHARS)
    return unique_texts, results_sorted, subqs

def upsert_chunks(index, doc_id: str, chunks: List[str], embeddings: List[List[float]], batch_size: int = 50):
    vectors = [
        {"id": f"{doc_id}_{i}", "values": embeddings[i], "metadata": {"text": chunks[i], "doc_id": doc_id}}
        for i in range(len(chunks))
    ]
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i:i+batch_size])

def build_answer(genai_client, question: str, contexts: List[str]) -> str:
    system_instruction = build_system_prompt()
    user_prompt = build_user_prompt(question, contexts)
    model = genai.GenerativeModel(model_name=CHAT_MODEL, system_instruction=system_instruction)
    resp = model.generate_content(user_prompt)
    return (resp.text or "").strip()

# ========= STREAMLIT UI =========
def main():
    st.title("🔎 Fan-Out RAG – Gemini + Pinecone (uv)")
    st.caption("Query expansion → fan-out retrieval → dedupe → grounded answer")

    try:
        genai_client = get_gemini_client()
        pinecone_index = get_pinecone_index()
    except Exception as e:
        st.error(f"Configuration error: {e}")
        return

    if "doc_id" not in st.session_state:
        st.session_state.doc_id = None

    with st.expander("Ingestion settings", expanded=False):
        size = st.number_input("Chunk size", 500, 3000, 1000, step=100)
        overlap = st.number_input("Chunk overlap", 0, 500, 200, step=20)

    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded and st.button("Process PDF"):
        try:
            text = extract_text_from_pdf(uploaded)
            chunks = chunk_text(text, size=size, overlap=overlap)
            if not chunks:
                st.warning("No text extracted from the PDF.")
                return
            doc_id = str(uuid.uuid4())
            embeddings = embed_batch(genai_client, chunks)
            upsert_chunks(pinecone_index, doc_id, chunks, embeddings)
            st.session_state.doc_id = doc_id
            st.success(f"PDF processed! doc_id={doc_id} | {len(chunks)} chunks")
        except Exception as e:
            st.error(f"Ingestion error: {e}")

    st.divider()

    if st.session_state.doc_id:
        st.subheader("Ask a question about your document")
        question = st.text_input("Enter your question")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            fanout_k = st.number_input("Sub-queries (fan-out)", 1, 10, DEFAULT_FANOUT_K, step=1)
        with c2:
            topk_per = st.number_input("Top-K per sub-query", 1, 10, DEFAULT_TOP_K_PER_SUBQ, step=1)
        with c3:
            max_unique = st.number_input("Max unique chunks", 1, 50, DEFAULT_MAX_UNIQUE_CHUNKS, step=1)
        with c4:
            restrict_to_doc = st.checkbox("Restrict to this document only", value=True)

        if st.button("Run Fan-Out RAG"):
            if not question.strip():
                st.warning("Please enter a question.")
                return
            with st.spinner("Expanding, retrieving, deduping, and answering..."):
                try:
                    contexts, all_results, subqs = fanout_retrieve_unique(
                        genai_client=genai_client,
                        index=pinecone_index,
                        user_question=question,
                        doc_id_filter=st.session_state.doc_id if restrict_to_doc else None,
                        fanout_k=fanout_k,
                        top_k_per_subq=topk_per,
                        max_unique_chunks=max_unique,
                    )

                    st.markdown("**Generated sub-queries:**")
                    st.write(subqs)

                    st.markdown(f"**Unique contexts selected:** {len(contexts)}")
                    with st.expander("Preview contexts", expanded=False):
                        for i, c in enumerate(contexts):
                            st.markdown(f"Chunk {i+1}")
                            st.code(c)

                    if not contexts:
                        st.warning("No relevant chunks found. Try rephrasing your question.")
                        return

                    answer = build_answer(genai_client, question, contexts)
                    st.subheader("Answer")
                    st.write(answer)
                except Exception as e:
                    st.error(f"RAG error: {e}")
    else:
        st.info("Upload and process a PDF to start asking questions.")

if __name__ == "__main__":
    main()