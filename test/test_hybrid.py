from retrieval.hybrid_retriever import HybridRetriever


def main():

    retriever = HybridRetriever()

    while True:

        query = input("\nEnter your query (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        results = retriever.search(query)

        print("\n")
        print("=" * 90)
        print("HYBRID SEARCH RESULTS")
        print("=" * 90)

        for result in results:

            print(f"\nRank : {result['rank']}")
            print(f"Document ID : {result['document_id']}")
            print(f"RRF Score : {result['rrf_score']:.6f}")

            print("\nDescription:")
            print(result["description"])

            print("\nTags:")
            print(result["tags"])

            print("-" * 90)


if __name__ == "__main__":
    main()