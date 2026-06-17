import pickle
import numpy as np

from utils.config import BM25_INDEX_PATH, DOCUMENTS_PATH


class BM25Retriever:
    """
    BM25-based keyword retriever.
    Loads the pre-built BM25 index and returns the top-k matching documents.
    """

    def __init__(self):
        print("Loading BM25 Index...")

        # Load BM25 index and corpus
        with open(BM25_INDEX_PATH, "rb") as file:
            self.corpus, self.bm25 = pickle.load(file)

        # Load original documents
        with open(DOCUMENTS_PATH, "rb") as file:
            self.documents = pickle.load(file)

        print(f"Loaded {len(self.documents)} documents successfully.")

    def search(self, query, top_k=5):
        """
        Search documents using BM25.

        Parameters:
            query (str): User search query
            top_k (int): Number of documents to return

        Returns:
            list: Ranked search results
        """

        # Convert query into tokens
        query_tokens = query.lower().split()

        # Calculate BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Sort scores in descending order
        ranked_indices = np.argsort(scores)[::-1]

        results = []

        # Get Top-K documents
        for rank, idx in enumerate(ranked_indices[:top_k], start=1):

            results.append({
                "rank": rank,
                "document_id": idx,
                "score": float(scores[idx]),
                "description": self.documents[idx]["description"],
                "tags": self.documents[idx]["tags"]
            })

        return results
    



