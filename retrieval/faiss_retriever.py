import pickle
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

from utils.config import (
    FAISS_INDEX_PATH,
    DOCUMENTS_PATH,
    MODEL_NAME,
)


class FAISSRetriever:
    """
    FAISS-based Semantic Retriever
    """

    def __init__(self):

        print("=" * 60)
        print("INITIALIZING FAISS RETRIEVER")
        print("=" * 60)

        # -----------------------------
        # Load FAISS Index
        # -----------------------------
        print("Loading FAISS index...")

        self.index = faiss.read_index(str(FAISS_INDEX_PATH))

        print("FAISS index loaded successfully.")

        # -----------------------------
        # Load Documents
        # -----------------------------
        print("Loading documents...")

        with open(DOCUMENTS_PATH, "rb") as file:
            self.documents = pickle.load(file)

        print(f"Loaded {len(self.documents)} documents.")

        # -----------------------------
        # Load Embedding Model
        # -----------------------------
        print("Loading Sentence Transformer...")

        self.model = SentenceTransformer(MODEL_NAME)

        print("Sentence Transformer loaded.")

        print("=" * 60)
        print("Retriever Ready")
        print("=" * 60)

    def search(self, query, top_k=5):

        print("\n----------------------------")
        print("Starting Semantic Search")
        print("----------------------------")

        print("User Query :", query)

        # -----------------------------
        # Generate Query Embedding
        # -----------------------------
        print("\nStep 1 : Encoding Query...")

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False
        )

        print("Embedding generated.")

        query_embedding = query_embedding.astype(np.float32)

        print("Embedding Shape :", query_embedding.shape)
        print("Embedding Type  :", query_embedding.dtype)

        # -----------------------------
        # FAISS Search
        # -----------------------------
        print("\nStep 2 : Searching FAISS Index...")

        distances, indices = self.index.search(query_embedding, top_k)

        print("FAISS Search Completed")

        print("Distances :", distances)
        print("Indices   :", indices)

        # -----------------------------
        # Prepare Results
        # -----------------------------
        results = []

        print("\nStep 3 : Preparing Results...")

        for rank, (doc_id, distance) in enumerate(
            zip(indices[0], distances[0]),
            start=1,
        ):

            if doc_id == -1:
                continue

            results.append(
                {
                    "rank": rank,
                    "document_id": int(doc_id),
                    "distance": float(distance),
                    "description": self.documents[doc_id]["description"],
                    "tags": self.documents[doc_id]["tags"],
                }
            )

        print(f"{len(results)} documents returned.")

        return results
    


