import streamlit as st
from openai import OpenAI
from tinydb import TinyDB
from pypdf import PdfReader

# ---------------- PAGE ----------------
st.set_page_config(page_title="AI Engineering Tutor", layout="centered")
st.title("🤖 AI Engineering Tutor")
st.caption("Memory + Chat (OpenRouter)")

# ---------------- OPENROUTER CLIENT ----------------
client = OpenAI(
    api_key=st.secrets["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1"
)

# ---------------- MEMORY ----------------
db = TinyDB("memory.json")

if "messages" not in st.session_state:
    saved = db.all()
    st.session_state.messages = saved[0]["messages"] if saved else []

# ---------------- CHAT UI ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask anything about AI / ML / Python")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

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

    # save memory
    db.truncate()
    db.insert({"messages": st.session_state.messages})
