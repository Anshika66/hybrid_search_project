from retrieval.faiss_retriever import FAISSRetriever


def main():

    retriever = FAISSRetriever()

    while True:

        query = input("\nEnter your search query (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        try:

            results = retriever.search(query, top_k=5)

            print("\n")
            print("=" * 80)
            print("TOP SEMANTIC RESULTS")
            print("=" * 80)

            for result in results:

                print(f"\nRank        : {result['rank']}")
                print(f"Document ID : {result['document_id']}")
                print(f"Distance    : {result['distance']:.4f}")

                print("\nDescription:")
                print(result["description"])

                print("\nTags:")
                print(result["tags"])

                print("-" * 80)

        except Exception as e:

            print("\nERROR OCCURRED")
            print(type(e))
            print(e)


if __name__ == "__main__":
    main()