import streamlit as st
from openai import OpenAI
from tinydb import TinyDB
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from streamlit_mic_recorder import mic_recorder
from faster_whisper import WhisperModel
import tempfile


# ---------------- PAGE ----------------
st.set_page_config(page_title="PRO Engineering Tutor", page_icon="🤖")

st.title("🤖 PRO Engineering Tutor")
st.write("Memory + Multi-PDF Brain + Streaming + Voice AI")

# ---------------- OPENROUTER ----------------
client = OpenAI(
    api_key=st.secrets["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1"
)

# ---------------- MEMORY ----------------
db = TinyDB("memory.json")

if "messages" not in st.session_state:
    data = db.all()

    if data:
        st.session_state.messages = data[0]["messages"]
    else:
        st.session_state.messages = [
            {
                "role": "system",
                "content": "You are an expert engineering tutor. Explain clearly with examples."
            }
        ]

# ---------------- LOAD EMBEDDING MODEL ----------------
if "embed_model" not in st.session_state:
    with st.spinner("Loading AI brain... first run takes ~20 seconds ⏳"):
        st.session_state.embed_model = SentenceTransformer('all-MiniLM-L6-v2')

embed_model = st.session_state.embed_model

# ---------------- LOAD WHISPER ----------------
if "whisper_model" not in st.session_state:
    with st.spinner("Loading speech recognition model..."):
        st.session_state.whisper_model = WhisperModel("base")

whisper_model = st.session_state.whisper_model


# ---------------- VECTOR DB ----------------
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
    st.session_state.text_chunks = []


# ---------------- MULTI PDF UPLOAD ----------------
uploaded_files = st.file_uploader(
    "📚 Upload Engineering PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    all_text = ""

    for file in uploaded_files:
        reader = PdfReader(file)

        for page in reader.pages:
            text = page.extract_text()
            if text:
                all_text += text

    chunks = [all_text[i:i+500] for i in range(0, len(all_text), 500)]
    embeddings = embed_model.encode(chunks)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    st.session_state.vector_db = index
    st.session_state.text_chunks = chunks

    st.success("✅ PDFs uploaded! Ask questions from them.")


# ---------------- DISPLAY OLD CHAT ----------------
for msg in st.session_state.messages[1:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# =====================================================
# 🎤 VOICE ASSISTANT
# =====================================================

st.divider()
st.subheader("🎤 Voice Assistant")

audio = mic_recorder(start_prompt="🎤 Speak", stop_prompt="Stop recording")

voice_prompt = None

if audio:

    st.audio(audio["bytes"])

    # Save temp audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio["bytes"])
        audio_path = f.name

    # Transcribe speech
    segments, _ = whisper_model.transcribe(audio_path)

    spoken_text = ""
    for segment in segments:
        spoken_text += segment.text

    st.success(f"You said: {spoken_text}")

    voice_prompt = spoken_text


# ---------------- TEXT INPUT ----------------
text_prompt = st.chat_input("Ask anything about engineering...")

prompt = voice_prompt if voice_prompt else text_prompt


# =====================================================
# 🤖 AI RESPONSE
# =====================================================

if prompt:

    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    context = ""

    # SEARCH PDF
    if st.session_state.vector_db:

        query_embedding = embed_model.encode([prompt])
        D, I = st.session_state.vector_db.search(np.array(query_embedding), k=3)

        context = "\n".join(
            [st.session_state.text_chunks[i] for i in I[0]]
        )

        final_prompt = f"""
Use this PDF context when relevant:

{context}

Question:
{prompt}
"""
    else:
        final_prompt = prompt

    messages_for_api = st.session_state.messages + [
        {"role": "user", "content": final_prompt}
    ]

    # ================= STREAMING =================
    with st.chat_message("assistant"):

        message_placeholder = st.empty()
        full_reply = ""

        stream = client.chat.completions.create(
            model="anthropic/claude-3-haiku",
            messages=messages_for_api,
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                full_reply += chunk.choices[0].delta.content
                message_placeholder.markdown(full_reply)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_reply}
    )

    # ---------------- SAVE MEMORY ----------------
    db.truncate()
    db.insert({"messages": st.session_state.messages})

