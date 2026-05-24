import requests
from bs4 import BeautifulSoup
import json
import os

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
