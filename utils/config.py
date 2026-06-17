from pathlib import Path


#root directoy
BASE_DIR = Path(__file__).resolve().parent.parent 

#dataset 
DATASET_PATH = BASE_DIR / 'dataset' / "datasets.csv"


#index folder 
INDEX_DIR = BASE_DIR / 'indexes'

#saved files 
BM25_INDEX_PATH = INDEX_DIR / "bm25_index.pkl"
FAISS_INDEX_PATH = INDEX_DIR / "vector_index.faiss"
DOCUMENTS_PATH = INDEX_DIR / "documents.pkl"

# embedding model 
MODEL_NAME = "all-MiniLM-L6-v2"   

