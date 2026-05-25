import pickle
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

QA_PATH = "qa/qa_dataset.csv"
INDEX_PATH = "data/faiss.index"
METADATA_PATH = "data/metadata.pkl"
THRESHOLD = 0.80

class Retriever:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.read_index(INDEX_PATH)

        with open(METADATA_PATH, "rb") as file:
            self.store = pickle.load(file)

        self.df = pd.read_csv(QA_PATH)
        self.question_embeddings = self.model.encode(self.df["question"].tolist())
        self.question_norms = np.linalg.norm(self.question_embeddings, axis=1)

    def search_qa(self, query):
        query_embedding = self.model.encode([query])[0]
        query_norm = np.linalg.norm(query_embedding)

        scores = []
        for i, question_embedding in enumerate(self.question_embeddings):
            dot_product = np.dot(query_embedding, question_embedding)
            similarity = dot_product / (query_norm * self.question_norms[i])
            scores.append(similarity)

        best_index = scores.index(max(scores))
        best_score = scores[best_index]

        if best_score >= THRESHOLD:
            return self.df.iloc[best_index]["answer"], self.df.iloc[best_index]["source_page"], best_score
        return None, None, best_score

    def search_vector_store(self, query, n_result=3):
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")

        _distance, indices = self.index.search(query_embedding, n_result)
        chunks = [self.store["chunks"][i] for i in indices[0]]
        sources = [self.store["metadata"][i]["source"] for i in indices[0]]

        return chunks, sources

    def retrieve(self, query):
        answer, source, score = self.search_qa(query)
        if answer:
            return {"answer": answer, "source": source, "method": "qa_match", "score": score}

        chunks, sources = self.search_vector_store(query)
        context = "\n\n".join(chunks)
        return {"context": context, "sources": sources, "method": "vector_search"}


retriever = Retriever()

def retrieve(query):
    return retriever.retrieve(query)
