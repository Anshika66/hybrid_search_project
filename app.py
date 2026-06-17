import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"



import sys
import os
import time

# Force system path tracking
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import streamlit as st
from retrieval.hybrid_retriever import HybridRetriever

st.set_page_config(page_title="Enterprise Search Portal", layout="wide")
st.title("⚡ Enterprise Hybrid Search Engine Dashboard")
st.write("Querying 2,173 records across decoupled BM25 and FAISS indices via Reciprocal Rank Fusion.")

@st.cache_resource
def load_search_engine():
    return HybridRetriever()

with st.spinner("⏳ Loading system indices into local memory cache..."):
    try:
        engine = load_search_engine()
    except Exception as e:
        st.error(f"🚨 Initialization Error: {e}")
        st.stop()

search_input = st.text_input("🔎 Input search keyword phrases or conceptual tags:", placeholder="Type a concept (e.g., machine learning, database, gaming)...")

# Helper function to truncate huge text descriptions so they don't crash the browser websocket
def safe_text(text, max_chars=300):
    text_str = str(text)
    if len(text_str) > max_chars:
        return text_str[:max_chars] + "..."
    return text_str

if search_input:
    start_time = time.perf_counter()
    
    # Execute modular queries
    bm25_hits = engine.bm25.search(search_input, top_k=5)
    faiss_hits = engine.faiss.search(search_input, top_k=5)
    fused_hits = engine.search(search_input, top_k=5)
    
    latency_ms = (time.perf_counter() - start_time) * 1000
    st.metric(label="Search Latency Profile", value=f"{latency_ms:.2f} ms")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🔍 Keyword Track (BM25)")
        for item in bm25_hits:
            with st.expander(f"📄 Doc ID: {item['document_id']} | Score: {item['score']:.2f}", expanded=True):
                st.write(safe_text(item['description']))
                st.caption(f"🏷️ Tags: {safe_text(item['tags'], max_chars=100)}")
                
    with col2:
        st.subheader("🧠 Concept Track (FAISS)")
        for item in faiss_hits:
            with st.container(border=True):
                st.markdown(f"**Doc ID: {item['document_id']}** *(Distance: {item['distance']:.4f})*")
                st.write(safe_text(item['description']))
                st.caption(f"🏷️ Tags: {safe_text(item['tags'], max_chars=100)}")
                
    with col3:
        st.subheader("🏆 Unified Rankings (RRF)")
        for item in fused_hits:
            with st.container(border=True):
                st.markdown(f"🔥 **Rank {item['rank']}** *(Score: {item['rrf_score']:.4f})*")
                st.write(safe_text(item['description']))
                st.caption(f"🏷️ Tags: {safe_text(item['tags'], max_chars=100)}")