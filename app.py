import streamlit as st
import pandas as pd
import os
from src.chatbot import ask

st.set_page_config(page_title="Chain Mind", page_icon="🔗", layout="wide")
st.title("Chain Mind")
st.caption("Your LangChain documentation assistant")

with st.sidebar:
  st.header("About")
  st.write("An intelligent chatbot for LangChain documentation using hybrid RAG retrieval.")
  st.divider()
  if os.path.exists("qa/qa_dataset.csv"):
      df = pd.read_csv("qa/qa_dataset.csv")
      st.metric("Q/A Pairs", len(df))
  raw_files = len(os.listdir("data/raw")) if os.path.exists("data/raw") else 0
  st.metric("Pages Scraped", raw_files)
  st.divider()
  if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

if "messages" not in st.session_state:
      st.session_state.messages = []

for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.write(message["content"])
    if "source" in message:
      st.caption(f"Source: {message['source']}")

if query := st.chat_input("Ask about LangChain..."):
  st.session_state.messages.append({"role": "user", "content": query})
  with st.chat_message("user"):
    st.write(query)

  history = [{"role": message["role"], "content": message["content"]} for message in st.session_state.messages[:-1]]
  answer, source = ask(query, history)

  st.session_state.messages.append({"role": "assistant", "content": answer, "source": source})
  with st.chat_message("assistant"):
    st.write(answer)
    st.caption(f"Source: {source}")
