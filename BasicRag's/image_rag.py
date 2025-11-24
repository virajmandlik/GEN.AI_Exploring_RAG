import os
import uuid
from typing import List

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

EMBEDDING_MODEL = "text-embedding-004"
VISION_MODEL = "gemini-2.0-flash"
CHAT_MODEL = "gemini-2.0-flash"

PINECONE_INDEX_DIM = 768
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"


@st.cache_resource
def get_gemini():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY missing")

    genai.configure(api_key=api_key)
    return genai


@st.cache_resource
def get_pinecone():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY missing")

    index_name = os.getenv("PINECONE_INDEX_NAME", "image-rag-index")

    pc = Pinecone(api_key=api_key)

    if index_name not in [i["name"] for i in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=PINECONE_INDEX_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )

    return pc.Index(index_name)


def extract_text_from_image(genai_client, image):
    model = genai_client.GenerativeModel("gemini-2.0-flash")

    resp = model.generate_content(
        [
            "Extract ALL visible text from this image. Return plain text only.",
            image
        ]
    )

    return resp.text or ""



def chunk_text(text, size=500, overlap=100):
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+size])
        start += size - overlap
    return chunks


def embed_batch(genai_client, texts):
    embeddings = []
    for t in texts:
        resp = genai_client.embed_content(
            model=EMBEDDING_MODEL,
            content=t
        )
        embeddings.append(resp['embedding'])
    return embeddings


def build_prompt(question, contexts):
    ctx = "\n\n".join(contexts)
    return f"""
Use ONLY this extracted text to answer.

CONTEXT:
{ctx}

QUESTION: {question}

If not found, say "Not found in image text".
"""


def main():
    st.title("🖼️ Image RAG – Gemini + Pinecone (uv)")
    st.write("Upload an image and ask questions about its text")

    genai_client = get_gemini()
    pinecone_index = get_pinecone()

    if "img_doc" not in st.session_state:
        st.session_state.img_doc = None

    image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if image_file and st.button("Process Image"):
        image = Image.open(image_file)

        text = extract_text_from_image(genai_client, image)

        chunks = chunk_text(text)

        doc_id = str(uuid.uuid4())
        embeddings = embed_batch(genai_client, chunks)

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

        st.session_state.img_doc = doc_id
        st.success("Image processed!")

    if st.session_state.img_doc:
        question = st.text_input("Ask about the image text")

        if st.button("Ask"):
            q_emb = embed_batch(genai_client, [question])[0]

            res = pinecone_index.query(
                vector=q_emb,
                top_k=5,
                include_metadata=True,
                filter={"doc_id": {"$eq": st.session_state.img_doc}}
            )

            contexts = [m["metadata"]["text"] for m in res.matches]

            prompt = build_prompt(question, contexts)

            model = genai_client.GenerativeModel(CHAT_MODEL)
            answer = model.generate_content(prompt).text

            st.subheader("Answer:")
            st.write(answer)


if __name__ == "__main__":
    main()
