# Chain Mind

An intelligent RAG (Retrieval-Augmented Generation) chatbot for LangChain documentation. Ask questions about LangChain and get answers backed by real documentation.

---

## What It Does

Chain Mind uses a **hybrid retrieval** system to answer questions:

1. **Q/A Search first** — searches a curated Q/A dataset for a close match using cosine similarity
2. **Vector search fallback** — if no good match is found, searches scraped LangChain docs using FAISS
3. **LLM answer generation** — sends the retrieved context to Groq (Llama 3.3) to generate a natural language answer

Every answer includes a **source citation** so you know where the information came from.

---

## Tech Stack

- **Python 3.14**
- **Streamlit** — web interface
- **Groq (Llama 3.3)** — LLM for answer generation
- **FAISS** — vector index for semantic search
- **sentence-transformers** — text embeddings (`all-MiniLM-L6-v2`)
- **LangChain** — text splitting utilities
- **BeautifulSoup** — web scraping
- **pandas** — Q/A dataset management

---

## Setup

**1. Clone the repo and navigate into the project:**
```bash
cd chain_mind
```

**2. Create and activate a virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Create a `.env` file with your Groq API key:**
```
GROQ_API_KEY=your_key_here
```
Get a free API key at [console.groq.com](https://console.groq.com)

**5. Scrape the docs and build the vector index:**
```bash
python src/scraper.py
python src/vector_store.py
```

**6. Run the app:**
```bash
streamlit run app.py
```

---

## Usage

- Type any question about LangChain in the chat input
- The app will search the Q/A dataset first, then fall back to vector search
- Each answer shows a **source citation** indicating where the answer came from
- Use the **Clear Chat** button in the sidebar to reset the conversation
- The sidebar shows stats: number of Q/A pairs and pages scraped
