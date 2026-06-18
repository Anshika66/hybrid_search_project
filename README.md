# 🚀 Hybrid Search Engine
### AI-Powered Semantic + Keyword Search using BM25, FAISS & Sentence Transformers

![Python](https://img.shields.io/badge/Python-3.9-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-orange)
![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-Embeddings-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

A production-style **Hybrid Search Engine** that combines **traditional keyword search (BM25)** with **semantic vector search (FAISS + Sentence Transformers)** to deliver highly relevant search results.

Unlike a traditional search engine that only matches keywords, this project understands the **meaning behind a query**, making it suitable for AI-powered document retrieval and Retrieval-Augmented Generation (RAG) applications.

---

# ✨ Features

- 🔍 Keyword Search using BM25
- 🧠 Semantic Search using Sentence Transformers
- ⚡ Fast Vector Search using FAISS
- 🔀 Hybrid Search using Reciprocal Rank Fusion (RRF)
- 🚀 REST API built with FastAPI
- 📂 Offline Index Generation Pipeline
- 📊 Supports thousands of documents efficiently
- 📦 Clean, modular, production-ready architecture

---

# 🏗️ Architecture

```mermaid
graph TD
    User([User Query]) --> Split(( ))
    
    Split --> BM25[BM25 Retriever<br>Keyword Search]
    Split --> FAISS[FAISS Retriever<br>Semantic Vector Search]
    
    BM25 --> RRF{Reciprocal Rank Fusion<br>Combine & Re-rank}
    FAISS --> RRF
    
    RRF --> Ranked[Final Ranked Results]
    Ranked --> API[FastAPI Response]

    classDef input fill:#f97316,stroke:#ea580c,stroke-width:2px,color:#fff
    classDef retrieve fill:#3b82f6,stroke:#2563eb,stroke-width:2px,color:#fff
    classDef fusion fill:#a855f7,stroke:#9333ea,stroke-width:2px,color:#fff
    classDef output fill:#22c55e,stroke:#16a34a,stroke-width:2px,color:#fff
    
    class User input
    class BM25,FAISS retrieve
    class RRF fusion
    class Ranked,API output

