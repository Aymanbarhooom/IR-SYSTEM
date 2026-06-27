# models/embedding_model.py
import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    PROCESSED_DOCUMENTS_FILE,
    DOCUMENT_EMBEDDINGS_FILE,
    DOCUMENT_IDS_FILE,
    BERT_MODEL_NAME,
    MAX_DOCS_FOR_BERT
)


class EmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer(BERT_MODEL_NAME)
        self.doc_ids = []
        self.embeddings = None

        if os.path.exists(DOCUMENT_EMBEDDINGS_FILE) and os.path.exists(DOCUMENT_IDS_FILE):
            self.load_index()
        else:
            print("⚠️ Embeddings files not found. Will build when needed.")

    def build_index(self, force_rebuild=False):
        """بناء أو تحميل الـ embeddings"""
        if not force_rebuild and os.path.exists(DOCUMENT_EMBEDDINGS_FILE) and os.path.exists(DOCUMENT_IDS_FILE):
            self.load_index()
            return

        print("\n🔨 Building Embeddings Index...")
        print(f"Model: {BERT_MODEL_NAME}")
        print(f"Max documents: {MAX_DOCS_FOR_BERT or 'All'}")

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

        print(f"Encoding {len(texts)} documents... (this may take time)")

        self.embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=64   
        )

        os.makedirs(os.path.dirname(DOCUMENT_EMBEDDINGS_FILE), exist_ok=True)
        np.save(DOCUMENT_EMBEDDINGS_FILE, self.embeddings)

        with open(DOCUMENT_IDS_FILE, "w", encoding="utf-8") as f:
            json.dump(self.doc_ids, f, ensure_ascii=False, indent=2)

        print(f"✅ Embeddings built and saved successfully! ({len(self.doc_ids)} documents)")

    def load_index(self):
        print("\n📂 Loading existing embeddings...")
        self.embeddings = np.load(DOCUMENT_EMBEDDINGS_FILE)
        with open(DOCUMENT_IDS_FILE, "r", encoding="utf-8") as f:
            self.doc_ids = json.load(f)
        print(f"✅ Loaded {len(self.doc_ids)} document embeddings.")

    def search(self, query, top_k=10):
        if self.embeddings is None:
            print("⚠️ Embeddings not loaded. Building now...")
            self.build_index()

        if not query or not query.strip():
            return []

        query_embedding = self.model.encode([query], convert_to_numpy=True)

        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        top_indices = similarities.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((self.doc_ids[idx], float(similarities[idx])))

        return results