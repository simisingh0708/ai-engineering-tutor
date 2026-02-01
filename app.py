import streamlit as st
from openai import OpenAI
from tinydb import TinyDB

# ---------------- PAGE ----------------
st.set_page_config(page_title="AI Engineering Tutor", layout="wide")
st.title("🤖 AI Engineering Tutor")

# ---------------- OPENROUTER ----------------
client = OpenAI(
    api_key=st.secrets["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1"
)

# ---------------- MEMORY ----------------
db = TinyDB("memory.json")

if "messages" not in st.session_state:
    data = db.all()
    st.session_state.messages = data[0]["messages"] if data else []

# ---------------- CHAT HISTORY ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- USER INPUT ----------------
prompt = st.chat_input("Ask anything about engineering or AI...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_reply = ""

        response = client.chat.completions.create(
            model="anthropic/claude-3-haiku",
            messages=st.session_state.messages,
            stream=True,
        )

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                full_reply += chunk.choices[0].delta.content
                placeholder.markdown(full_reply)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_reply}
    )

    db.truncate()
    db.insert({"messages": st.session_state.messages})
