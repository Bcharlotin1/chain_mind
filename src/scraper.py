import requests
import json
import os
from bs4 import BeautifulSoup

PAGES = [
      "https://docs.langchain.com/oss/python/langchain/overview",
      "https://docs.langchain.com/oss/python/langchain/quickstart",
      "https://docs.langchain.com/oss/python/langchain/agents",
      "https://docs.langchain.com/oss/python/langchain/models",
      "https://docs.langchain.com/oss/python/langchain/messages",
      "https://docs.langchain.com/oss/python/langchain/tools",
      "https://docs.langchain.com/oss/python/langchain/retrieval",
      "https://docs.langchain.com/oss/python/langchain/rag",
      "https://docs.langchain.com/oss/python/langchain/knowledge-base",
      "https://docs.langchain.com/oss/python/langchain/context-engineering",
      "https://docs.langchain.com/oss/python/langchain/short-term-memory",
      "https://docs.langchain.com/oss/python/langchain/long-term-memory",
      "https://docs.langchain.com/oss/python/langchain/structured-output",
      "https://docs.langchain.com/oss/python/langchain/streaming",
      "https://docs.langchain.com/oss/python/langchain/middleware",
  ]

def scrape_page(url):
  response = requests.get(url)

  soup = BeautifulSoup(response.text, "html.parser")
  for tag in soup(["nav", "footer", "script", "style"]):
    tag.decompose()

  text = soup.get_text(separator="\n", strip=True)

  return{"url": url, "content": text}

def scrape_all(output_dir="data/raw"):
  os.makedirs(output_dir, exist_ok=True)

  for i, url in enumerate(PAGES):
    pageData = scrape_page(url)
    filename = f"{output_dir}/page_{i+1}.json"

    with open(filename, "w") as file :
      json.dump(pageData, file, indent=2)
      print(f"Saved: {filename}")

