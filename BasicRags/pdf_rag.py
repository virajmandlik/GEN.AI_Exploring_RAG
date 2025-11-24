import os
import uuid
from typing import List

import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv

import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# ========= CONFIG =========
EMBEDDING_MODEL = "text-embedding-004"
CHAT_MODEL = "gemini-2.0-flash"   # or gemini-1.5-flash

PINECONE_INDEX_DIM = 768
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"


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

    if index_name not in [i["name"] for i in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=PINECONE_INDEX_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION,
            ),
        )

    return pc.Index(index_name)


def extract_text_from_pdf(file_bytes) -> str:
    reader = PdfReader(file_bytes)
    return "\n".join([page.extract_text() or "" for page in reader.pages])


def chunk_text(text: str, size=1000, overlap=200) -> List[str]:
    text = " ".join(text.replace("\n", " ").split())
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def embed_texts(genai_client, texts: List[str]):
    resp = genai_client.embed_content(
        model=EMBEDDING_MODEL,
        content=texts
    )
    return resp['embedding']


def embed_batch(genai_client, texts: List[str]):
    embeddings = []
    for t in texts:
        resp = genai_client.embed_content(
            model=EMBEDDING_MODEL,
            content=t
        )
        embeddings.append(resp['embedding'])
    return embeddings


def build_prompt(question, contexts):
    context_text = "\n\n".join([c for c in contexts])
    return f"""
You are a helpful assistant. Answer ONLY using the provided context.

CONTEXT:
{context_text}

QUESTION: {question}

If answer not in context, say "Not found in document".
"""


# ========= STREAMLIT UI =========
def main():
    st.title("📚 Basic RAG – Gemini + Pinecone (uv)")
    st.write("Upload a PDF and ask questions about it")

    try:
        genai_client = get_gemini_client()
        pinecone_index = get_pinecone_index()
    except Exception as e:
        st.error(f"Configuration error: {e}")
        return

    if "doc_id" not in st.session_state:
        st.session_state.doc_id = None

    uploaded = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded and st.button("Process PDF"):
        text = extract_text_from_pdf(uploaded)
        chunks = chunk_text(text)

        doc_id = str(uuid.uuid4())
        embeddings = embed_batch(genai_client, chunks)

        # store in pinecone
        vectors = [
            {
                "id": f"{doc_id}_{i}",
                "values": embeddings[i],
                "metadata": {"text": chunks[i], "doc_id": doc_id}
            }
            for i in range(len(chunks))
        ]

        for i in range(0, len(vectors), 50):
            pinecone_index.upsert(vectors=vectors[i:i+50])

        st.session_state.doc_id = doc_id
        st.success("PDF processed!")

    if st.session_state.doc_id:
        question = st.text_input("Ask a question")

        if st.button("Ask"):
            q_emb = embed_batch(genai_client, [question])[0]

            res = pinecone_index.query(
                vector=q_emb,
                top_k=5,
                include_metadata=True,
                filter={"doc_id": {"$eq": st.session_state.doc_id}}
            )

            contexts = [m["metadata"]["text"] for m in res.matches]

            prompt = build_prompt(question, contexts)

            model = genai.GenerativeModel(CHAT_MODEL)
            answer = model.generate_content(prompt).text

            st.subheader("Answer:")
            st.write(answer)


if __name__ == "__main__":
    main()
