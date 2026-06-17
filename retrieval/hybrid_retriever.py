from retrieval.bm25_retriever import BM25Retriever
from retrieval.faiss_retriever import FAISSRetriever

class HybridRetriever:

    def __init__(self):
        print("=" * 60)
        print("INITIALIZING HYBRID RETRIEVER")
        print("=" * 60)

        self.bm25 = BM25Retriever()
        self.faiss = FAISSRetriever()

        print("Hybrid Retriever Ready")
        print("=" * 60)

    def search(self, query, top_k=5):
        print("\nSearching using BM25...")
        bm25_results = self.bm25.search(query, top_k=top_k)

        print("Searching using FAISS...")
        faiss_results = self.faiss.search(query, top_k=top_k)

        return self.reciprocal_rank_fusion(
            bm25_results,
            faiss_results,
            top_k
        )

    def reciprocal_rank_fusion(
        self,
        bm25_results,
        faiss_results,
        top_k=5,
        k=60,
    ):
        scores = {}
        documents = {}

        # -------------------------
        # BM25 Contribution
        # -------------------------
        for result in bm25_results:
            doc_id = result["document_id"]
            # EXTRACTING THE TRUE RETRIEVER RANK:
            retriever_rank = result["rank"] 
            
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + retriever_rank)
            documents[doc_id] = result

        # -------------------------
        # FAISS Contribution
        # -------------------------
        for result in faiss_results:
            doc_id = result["document_id"]
            # EXTRACTING THE TRUE RETRIEVER RANK:
            retriever_rank = result["rank"]
            
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + retriever_rank)
            documents[doc_id] = result

        # -------------------------
        # Sort and Rank Fusion Unification
        # -------------------------
        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        final_results = []

        for rank, (doc_id, score) in enumerate(
            ranked[:top_k],
            start=1
        ):
            document = documents[doc_id]

            final_results.append({
                "rank": rank,
                "document_id": doc_id,
                "rrf_score": score,
                "description": document["description"],
                "tags": document["tags"]
            })

        return final_results
    


