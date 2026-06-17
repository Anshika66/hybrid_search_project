"""Compatibility import shim.

This repository's retriever implementations live in the `retrieval` package.
Some environments/snippets expect `from retriever import FAISSRetriever`.

Keep this file lightweight and re-export the public retriever classes.
"""

from retrieval.faiss_retriever import FAISSRetriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever

__all__ = [
    "FAISSRetriever",
    "BM25Retriever",
    "HybridRetriever",
]

