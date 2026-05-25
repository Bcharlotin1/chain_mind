import os
from groq import Groq
from dotenv import load_dotenv
from src.retriever import retrieve

SYSTEM_PROMPT = """You are an expert assistant for LangChain documentation.
You help developers understand and use LangChain effectively.
Answer questions based only on the provided context.
If the context doesn't contain enough information, say so honestly.
Keep answers clear and concise."""

model = "llama-3.3-70b-versatile"

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

def ask(query, chat_history=[]):
  retrieval = retrieve(query)

  if retrieval["method"] == "off_topic":
    return "I can only answer questions about LangChain documentation.", "N/A"

  if retrieval["method"] == "qa_match":
    context = retrieval["answer"]
    source = retrieval["source"]
  else:
    context = retrieval["context"]
    source = ", ".join(set(retrieval["sources"]))

  messages = chat_history + [
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]

  response = client.chat.completions.create(
        model= model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            *messages,
        ]
    )

  answer = response.choices[0].message.content
  return answer, source
