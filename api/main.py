import os
# Force macOS safety patches immediately before heavy libraries initialize
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import traceback
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# System path tracking to easily find 'retrieval' modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from retrieval.hybrid_retriever import HybridRetriever

app = FastAPI(title="Hybrid Search Engine API")

# Allow frontend to communicate smoothly across ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global initialization of the hybrid engine
engine = HybridRetriever()

@app.get("/api/search")
async def search(q: str = Query(..., min_length=1)):
    try:
        # 1. Execute the search pipelines
        bm25_hits = engine.bm25.search(q, top_k=5)
        faiss_hits = engine.faiss.search(q, top_k=5)
        fused_hits = engine.search(q, top_k=5)
        
        # 2. Package BM25 Column with absolute key safety
        processed_bm25 = []
        for h in bm25_hits:
            doc_id = h.get("document_id", h.get("id", -1))
            score = h.get("score", h.get("distance", 0.0))
            processed_bm25.append({
                "document_id": int(doc_id),
                "score": float(score),
                "description": h.get("description", "No description string available."),
                "tags": h.get("tags", "None")
            })

        # 3. Package FAISS Column with absolute key safety
        processed_faiss = []
        for h in faiss_hits:
            doc_id = h.get("document_id", h.get("id", -1))
            distance = h.get("distance", h.get("score", 0.0))
            processed_faiss.append({
                "document_id": int(doc_id),
                "distance": float(distance),
                "description": h.get("description", "No description string available."),
                "tags": h.get("tags", "None")
            })

        # 4. Package Fused RRF Column with absolute key safety
        processed_fused = []
        for idx, h in enumerate(fused_hits):
            doc_id = h.get("document_id", h.get("id", -1))
            rrf_score = h.get("rrf_score", h.get("score", 0.0))
            rank = h.get("rank", idx + 1)
            processed_fused.append({
                "rank": int(rank),
                "document_id": int(doc_id),
                "rrf_score": float(rrf_score),
                "description": h.get("description", "No description string available."),
                "tags": h.get("tags", "None")
            })
        
        return {
            "bm25": processed_bm25,
            "faiss": processed_faiss,
            "fused": processed_fused
        }

    except Exception as e:
        # If anything breaks, print the exact file line number in your terminal console!
        print("\n" + "="*60 + "\nCRITICAL BACKEND CRASH DETECTED\n" + "="*60)
        traceback.print_exc()
        print("="*60 + "\n")
        return {"error": f"Internal Search Engine Error: {str(e)}"}

# ------------------------------------------------------------------------------
# DYNAMIC PATH RESOLUTION: Forces FastAPI to find index.html inside the api/ folder
# ------------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
HTML_FILE_PATH = os.path.join(CURRENT_DIR, "index.html")

@app.get("/")
async def read_index():
    if os.path.exists(HTML_FILE_PATH):
        return FileResponse(HTML_FILE_PATH)
    else:
        return {
            "status": "API is online, but frontend UI file is missing.",
            "expected_file_location": HTML_FILE_PATH,
            "solution": "Make sure your index.html file is saved inside the 'api' directory right next to main.py!"
        }
