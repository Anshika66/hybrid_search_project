import os

# -----------------------------
# Reduce CPU & Memory Usage
# -----------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import traceback
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# -----------------------------
# Project Path
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from retrieval.hybrid_retriever import HybridRetriever

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="Hybrid Search Engine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Lazy Loading
# -----------------------------
engine = None


def get_engine():
    global engine

    if engine is None:
        print("=" * 60)
        print("Loading Hybrid Search Engine...")
        print("=" * 60)

        engine = HybridRetriever()

        print("=" * 60)
        print("Hybrid Search Engine Loaded Successfully")
        print("=" * 60)

    return engine


# -----------------------------
# Search API
# -----------------------------
@app.get("/api/search")
async def search(q: str = Query(..., min_length=1)):

    try:

        engine = get_engine()

        # Search only once
        bm25_hits = engine.bm25.search(q, top_k=5)

        faiss_hits = engine.faiss.search(q, top_k=5)

        fused_hits = engine.reciprocal_rank_fusion(
            bm25_hits,
            faiss_hits,
            top_k=5
        )

        processed_bm25 = []

        for h in bm25_hits:

            processed_bm25.append({

                "document_id": int(h["document_id"]),

                "score": float(h["score"]),

                "description": h["description"][:500],

                "tags": h["tags"]

            })

        processed_faiss = []

        for h in faiss_hits:

            processed_faiss.append({

                "document_id": int(h["document_id"]),

                "distance": float(h["distance"]),

                "description": h["description"][:500],

                "tags": h["tags"]

            })

        processed_fused = []

        for h in fused_hits:

            processed_fused.append({

                "rank": int(h["rank"]),

                "document_id": int(h["document_id"]),

                "rrf_score": float(h["rrf_score"]),

                "description": h["description"][:500],

                "tags": h["tags"]

            })

        return {

            "bm25": processed_bm25,

            "faiss": processed_faiss,

            "fused": processed_fused

        }

    except Exception as e:

        print("\n")
        print("=" * 70)
        print("SEARCH API ERROR")
        print("=" * 70)
        traceback.print_exc()
        print("=" * 70)

        return {
            "success": False,
            "error": str(e)
        }


# -----------------------------
# Root Endpoint
# -----------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

HTML_FILE = os.path.join(CURRENT_DIR, "index.html")


@app.get("/")
async def home():

    if os.path.exists(HTML_FILE):
        return FileResponse(HTML_FILE)

    return {

        "message": "Hybrid Search Engine API Running",

        "status": "Frontend not found.",

        "expected_location": HTML_FILE

    }


# -----------------------------
# Health Check
# -----------------------------
@app.get("/health")
async def health():

    return {
        "status": "healthy"
    }
