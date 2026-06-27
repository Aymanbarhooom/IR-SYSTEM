# -*- coding: utf-8 -*-

# models/embedding_faiss.py
import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config import (
    PROCESSED_DOCUMENTS_FILE,
    EMBEDDINGS_DIR,
    BERT_MODEL_NAME,
    MAX_DOCS_FOR_BERT
)

class FAISSEmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer(BERT_MODEL_NAME)
        self.index = None
        self.doc_ids = []
        self.dimension = 384 
        
        self.index_path = EMBEDDINGS_DIR / "faiss_index.bin"
        self.ids_path = EMBEDDINGS_DIR / "faiss_doc_ids.json"
        
        os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
        
        if os.path.exists(self.index_path) and os.path.exists(self.ids_path):
            self.load_index()
        else:
            print("⚠️ FAISS index not found. Will build when build_index() is called.")

    def build_index(self):
        print("\n🔨 Building FAISS Embeddings Index...")
        
        with open(PROCESSED_DOCUMENTS_FILE, "r", encoding="utf-8") as f:
            documents = json.load(f)

        texts = []
        self.doc_ids = []

        limit = MAX_DOCS_FOR_BERT or len(documents)
        
        for i, (doc_id, doc) in enumerate(documents.items()):
            if i >= limit:
                break
            self.doc_ids.append(doc_id)
            texts.append(doc.get("processed_text", ""))

        print(f"Encoding {len(texts)} documents using {BERT_MODEL_NAME}...")

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=64,         
            convert_to_numpy=True
        )

        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(self.dimension) 
        self.index.add(embeddings)

        faiss.write_index(self.index, str(self.index_path))
        
        with open(self.ids_path, "w", encoding="utf-8") as f:
            json.dump(self.doc_ids, f)

        print(f"✅ FAISS Index built successfully with {len(self.doc_ids)} documents!")

    def load_index(self):
        print("📂 Loading FAISS index...")
        self.index = faiss.read_index(str(self.index_path))
        with open(self.ids_path, "r", encoding="utf-8") as f:
            self.doc_ids = json.load(f)
        print(f"✅ Loaded FAISS index with {len(self.doc_ids)} documents.")

    def search(self, query, top_k=10):
        if self.index is None:
            print("⚠️ Index not built. Building now...")
            self.build_index()

        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.doc_ids):
                results.append((self.doc_ids[idx], float(score)))

        return results