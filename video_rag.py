import os
import uuid
import tempfile

import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
from moviepy import VideoFileClip

load_dotenv()

# =============== CONFIG ==================
EMBED_MODEL = "text-embedding-004"
TRANSCRIBE_MODEL = "gemini-2.0-flash"
CHAT_MODEL = "gemini-2.0-flash"

PINECONE_INDEX_DIM = 768
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"


# =============== CLIENTS ==================
@st.cache_resource
def get_gemini():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set in environment/.env")
    genai.configure(api_key=api_key)
    return genai


@st.cache_resource
def get_pinecone():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY not set in environment/.env")

    index_name = os.getenv("PINECONE_INDEX_NAME", "video-rag-index")
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


# =============== UTILS ==================
def extract_audio(video_path):
    clip = VideoFileClip(video_path)

    temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

    # MoviePy 2.x compatible call
    clip.audio.write_audiofile(temp_audio)

    clip.close()
    return temp_audio



def chunk_text(text: str, size: int = 800, overlap: int = 200):
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def embed_texts(genai_client, texts):
    """Embed a list of strings, return list of vectors."""
    embeddings = []
    for t in texts:
        if not t.strip():
            continue
        resp = genai_client.embed_content(
            model=EMBED_MODEL,
            content=t,
        )
        embeddings.append(resp["embedding"])
    return embeddings


def transcribe_audio_stream(genai_client, audio_path: str):
    """
    Stream transcription from Gemini.
    Yields incremental text chunks as they arrive.
    """
    model = genai_client.GenerativeModel(TRANSCRIBE_MODEL)

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    # stream=True gives partial chunks
    stream = model.generate_content(
        [
            "Transcribe this audio into text. Return only the transcript.",
            {"mime_type": "audio/wav", "data": audio_bytes},
        ],
        stream=True,
    )

    for chunk in stream:
        if hasattr(chunk, "text") and chunk.text:
            yield chunk.text


def build_prompt(question: str, contexts):
    ctx = "\n\n".join(contexts)
    return f"""
You are a helpful assistant answering questions about the CONTENT of a video.
You are given chunks of transcript text. Use ONLY this text to answer.

CONTEXT:
{ctx}

QUESTION: {question}

If the answer is not clearly present in the context, say: "Not mentioned in the video."
"""


# =============== STREAMLIT APP ==================
def main():
    st.set_page_config(page_title="Video RAG – Gemini + Pinecone", layout="wide")
    st.title("🎥 Video RAG – Gemini + Pinecone")
    st.write("Upload a video and ask questions about what was said in it.")

    try:
        genai_client = get_gemini()
        pinecone_index = get_pinecone()
    except Exception as e:
        st.error(f"Configuration error: {e}")
        return

    if "video_doc_id" not in st.session_state:
        st.session_state.video_doc_id = None
    if "video_transcript" not in st.session_state:
        st.session_state.video_transcript = ""

    video = st.file_uploader("Upload Video", type=["mp4", "mkv", "mov", "mpeg4"])

    # ---------- PROCESS VIDEO ----------
    if video and st.button("Process Video"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video.read())
            video_path = tmp.name

        doc_id = str(uuid.uuid4())
        st.session_state.video_doc_id = doc_id
        st.session_state.video_transcript = ""

        with st.status("Processing video...", expanded=True) as status:
            st.write("📥 Extracting audio from video...")
            audio_path = extract_audio(video_path)
            st.write("✅ Audio extracted.")
            st.write("📝 Transcribing audio with Gemini (streaming)...")

            accumulated_text = ""
            chunk_count = 0

            # Stream transcription, update Pinecone as text comes in
            for partial_text in transcribe_audio_stream(genai_client, audio_path):
                accumulated_text += partial_text
                st.session_state.video_transcript = accumulated_text

                st.write(partial_text)  # show partial transcript

                # Take just the latest portion for embedding to avoid re-embedding everything
                chunks = chunk_text(accumulated_text)
                if not chunks:
                    continue

                last_chunk = chunks[-1]
                vectors = embed_texts(genai_client, [last_chunk])
                if not vectors:
                    continue

                pinecone_index.upsert(
                    vectors=[
                        {
                            "id": f"{doc_id}_{chunk_count}",
                            "values": vectors[0],
                            "metadata": {"text": last_chunk, "doc_id": doc_id},
                        }
                    ]
                )
                chunk_count += 1
                status.update(
                    label=f"Processing video... stored {chunk_count} chunks so far.",
                    state="running",
                )

            status.update(label="✅ Video processed and indexed!", state="complete")
            st.success("Video processed! You can now ask questions below.")

    # ---------- QA SECTION ----------
    st.header("Ask questions about this video")

    if not st.session_state.get("video_doc_id"):
        st.info("Upload and process a video first.")
        return

    question = st.text_input("Your question about the video:")

    if question and st.button("Ask"):
        if not st.session_state.get("video_transcript"):
            st.warning("Transcript still empty. Try processing the video first.")
            return

        with st.spinner("Retrieving relevant parts from Pinecone and generating answer..."):
            q_vec = embed_texts(genai_client, [question])[0]

            res = pinecone_index.query(
                vector=q_vec,
                top_k=5,
                include_metadata=True,
                filter={"doc_id": {"$eq": st.session_state.video_doc_id}},
            )

            contexts = [m["metadata"]["text"] for m in res.matches] if res.matches else []

            if not contexts:
                st.warning("No relevant chunks found for this question (maybe transcript still short).")
                return

            prompt = build_prompt(question, contexts)
            model = genai_client.GenerativeModel(CHAT_MODEL)
            answer = model.generate_content(prompt).text

            st.subheader("Answer")
            st.write(answer)

            with st.expander("Show retrieved transcript chunks"):
                for i, m in enumerate(res.matches):
                    st.markdown(f"**Chunk {i} – Score: {m['score']:.4f}**")
                    st.write(m["metadata"]["text"])
                    st.markdown("---")


if __name__ == "__main__":
    main()
