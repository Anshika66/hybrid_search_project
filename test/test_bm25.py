from retrieval.bm25_retriever import BM25Retriever


def main():

    retriever = BM25Retriever()

    while True:

        query = input("\nEnter your search query (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        results = retriever.search(query, top_k=5)

        print("\nTop Results\n")

        for result in results:

            print("=" * 80)

            print(f"Rank : {result['rank']}")
            print(f"Score : {result['score']:.4f}")

            print("\nDescription:")
            print(result["description"])

            print("\nTags:")
            print(result["tags"])

            print()


if __name__ == "__main__":
    main()