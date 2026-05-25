import os
import json
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

model = SentenceTransformer("all-MiniLM-L6-v2")
INDEX_PATH = "data/faiss.index"

def build_vector_store (raw_dir="data/raw"):
  splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
  all_chunks = []
  all_metadata = []

  for filename in os.listdir(raw_dir):
    if filename.endswith(".json"):
      with open(f"{raw_dir}/{filename}") as file:
        data = json.load(file)

        chunks = splitter.split_text(data["content"])
        for chunk in chunks:
          all_chunks.append(chunk)
          all_metadata.append({"source": data["url"]})

        print(f"Chunked {len(chunks)} chunks from {filename}")

  embeddings = model.encode(all_chunks, show_progress_bar=True)
  embeddings = np.array(embeddings).astype("float32")

  dimension = embeddings.shape[1]
  index = faiss.IndexFlatL2(dimension)
  index.add(embeddings)

  faiss.write_index(index, INDEX_PATH)
  print(f"FAISS index saved to {INDEX_PATH}")

  with open("data/metadata.pkl", "wb") as file:
    pickle.dump({"chunks": all_chunks, "metadata": all_metadata}, file)
  print("Metadata saved.")

if __name__ == "__main__":
  build_vector_store()


