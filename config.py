# config.py

from pathlib import Path

# Project Root
BASE_DIR = Path(__file__).resolve().parent

# Data folders
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

# Dataset name from ir_datasets
DATASET_NAME = "wikir/en1k/test"

# Number of documents to save (None = all)
MAX_DOCS = 500000

# Output files
DOCUMENTS_FILE = RAW_DATA_DIR / "documents.json"
QUERIES_FILE = RAW_DATA_DIR / "queries.json"
QRELS_FILE = RAW_DATA_DIR / "qrels.json"

WIKIR_DIR = BASE_DIR / "wikIR1k"

DOCUMENTS_CSV = WIKIR_DIR / "documents.csv"

TRAIN_QUERIES = (
    WIKIR_DIR / "training" / "queries.csv"
)

TRAIN_QRELS = (
    WIKIR_DIR / "training" / "qrels"
)

# Processed Data

PROCESSED_DATA_DIR = DATA_DIR / "processed"

PROCESSED_DOCUMENTS_FILE = (
    PROCESSED_DATA_DIR / "processed_documents.json"
)

PROCESSED_QUERIES_FILE = (
    PROCESSED_DATA_DIR / "processed_queries.json"
)


INDEXES_DIR = DATA_DIR / "indexes"

INVERTED_INDEX_FILE = (
    INDEXES_DIR / "inverted_index.json"
)
# Evaluation

TOP_K = 10

# Embeddings

EMBEDDINGS_DIR = DATA_DIR / "embeddings"

DOCUMENT_EMBEDDINGS_FILE = (
    EMBEDDINGS_DIR / "document_embeddings.npy"
)

DOCUMENT_IDS_FILE = (
    EMBEDDINGS_DIR / "document_ids.json"
)

BERT_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

MAX_DOCS_FOR_BERT = None
BM25_K1 = 1.5
BM25_B = 0.75

MYSQL_HOST = "localhost"
MYSQL_PORT = 3306

MYSQL_DATABASE = "ir_project"

MYSQL_USER = "root"
MYSQL_PASSWORD = ""