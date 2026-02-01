import streamlit as st
import requests
from tinydb import TinyDB

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="PRO Engineering Tutor",
    layout="centered"
)

st.title("🤖 PRO Engineering Tutor")
st.caption("Streamlit + OpenRouter (Claude)")

# ---------------- API SETUP ----------------
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

API_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}

# ---------------- MEMORY ----------------
db = TinyDB("memory.json")

if "messages" not in st.session_state:
    stored = db.all()
    if stored:
        st.session_state.messages = stored[0]["messages"]
    else:
        st.session_state.messages = []

# ---------------- CHAT HISTORY ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- USER INPUT ----------------
prompt = st.chat_input("Ask me anything...")

if prompt:
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    payload = {
        "model": "anthropic/claude-3-haiku",
        "messages": st.session_state.messages,
        "temperature": 0.7,
    }

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_reply = ""

        response = requests.post(API_URL, headers=HEADERS, json=payload)

        if response.status_code == 200:
            data = response.json()
            full_reply = data["choices"][0]["message"]["content"]
            placeholder.markdown(full_reply)
        else:
            placeholder.error("❌ API Error")

    st.session_state.messages.append(
        {"role": "assistant", "content": full_reply}
    )

    db.truncate()
    db.insert({"messages": st.session_state.messages})
