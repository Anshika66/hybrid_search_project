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

    def __init__(self):

        print("=" * 60)
        print("INITIALIZING FAISS RETRIEVER")
        print("=" * 60)

        print("Loading FAISS Index...")
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))
        print("FAISS Index Loaded.")

        print("Loading Documents...")
        with open(DOCUMENTS_PATH, "rb") as f:
            self.documents = pickle.load(f)
        print(f"{len(self.documents)} documents loaded.")

        # Lazy loading
        self.model = None

        print("Sentence Transformer will load on first search.")

        print("=" * 60)
        print("Retriever Ready")
        print("=" * 60)

    def load_model(self):

        if self.model is None:
            print("\nLoading Sentence Transformer...")
            self.model = SentenceTransformer(MODEL_NAME)
            print("Sentence Transformer Loaded.")

    def search(self, query, top_k=5):

        self.load_model()

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False
        ).astype(np.float32)

        distances, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for rank, (doc_id, distance) in enumerate(
            zip(indices[0], distances[0]),
            start=1,
        ):

            if doc_id == -1:
                continue

            results.append(
                {
                    "rank": int(rank),
                    "document_id": int(doc_id),
                    "distance": float(distance),
                    "description": self.documents[doc_id]["description"],
                    "tags": self.documents[doc_id]["tags"],
                }
            )

        return results
