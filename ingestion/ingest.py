import pickle
import faiss
import pandas as pd

from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from utils.config import (
    DATASET_PATH,
    INDEX_DIR,
    BM25_INDEX_PATH,
    FAISS_INDEX_PATH,
    DOCUMENTS_PATH,
    MODEL_NAME,
)


def run_ingestion():

    print("="* 60)
    print("HYBRID SEARCH ONLINE INGESTION STARTED")
    print("="*60)


    # step 1 : load dataset 
    print("\nStep1 : Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset loaded with {len(df)} records.")


    # step 2 : cleaning the dataset 
    print("\nstep2 : cleaning the dataset...")
    df["description"] = df["description"].fillna("").astype(str)
    df["tags"] = df["tags"].fillna("").astype(str)

    df["search_text"] = (
        df["description"] + " " + df["tags"]
    )

    corpus = df["search_text"].tolist()

    # step 3 : creating index directory if not exists
    print("\nStep3 : creating index directory if not exists...")
    INDEX_DIR.mkdir(exist_ok=True)


    # step 4 : saving documents 
    print("\nStep4 : saving documents...")
    documents = df.to_dict(orient="records")        

    with open(DOCUMENTS_PATH, "wb") as f:
        pickle.dump(documents, f)
    
    print(f"Documents saved ")

    # step 5 : creating and saving BM25 index
    print("\nstep 5 : creating and saving BM25 index ...")
    tokenized_corpus = [doc.lower().split() for doc in corpus]

    bm25 = BM25Okapi(tokenized_corpus)

    with open(BM25_INDEX_PATH , "wb") as f :
        pickle.dump((corpus , bm25) , f)

    print("bm25 index saved successfully")


    # step 6 : loading sentence transformer model 
    print("\nstep 6 : loading sentence transformer model...")
    model = SentenceTransformer(MODEL_NAME)

    # step 7 : generating embeddings 

    print("\nstep 7 : generating embeddings...")    
    embeddings = model.encode(
        corpus , 
        batch_size = 8,
        convert_to_numpy = True,
        show_progress_bar = True
    ).astype("float32") 

    print("Embedding shape :" , embeddings.shape)

    # step 8 : building faiss index 
    print("\nstep 8 : building faiss index...") 
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index , str(FAISS_INDEX_PATH))
    print("vector_index.faiss saved")


    # done 
    print("\nOffline Ingestion Completed Successfully")

    print("\nGenerated Files")

    print(DOCUMENTS_PATH)

    print(BM25_INDEX_PATH)

    print(FAISS_INDEX_PATH)


if __name__ == "__main__":
    run_ingestion()





