import streamlit as st
from openai import OpenAI
from tinydb import TinyDB
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import tempfile

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="PRO Engineering Tutor",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 PRO Engineering Tutor")
st.write("Memory + Multi-PDF Brain + Streaming AI")

# ---------------- OPENROUTER CLIENT ----------------
client = OpenAI(
    api_key=st.secrets["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1"
)

# ---------------- MEMORY ----------------
db = TinyDB("memory.json")

if "messages" not in st.session_state:
    saved = db.all()
    if saved:
        st.session_state.messages = saved[0]["messages"]
    else:
        st.session_state.messages = []

# ---------------- SHOW CHAT HISTORY ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- USER INPUT ----------------
user_input = st.chat_input("Ask anything...")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_reply = ""

        stream = client.chat.completions.create(
            model="anthropic/claude-3-haiku",
            messages=st.session_state.messages,
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                full_reply += chunk.choices[0].delta.content
                placeholder.markdown(full_reply)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_reply}
    )

    # ---------------- SAVE MEMORY ----------------
    db.truncate()
    db.insert({"messages": st.session_state.messages})
