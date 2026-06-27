# models/tfidf_model.py

import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import PROCESSED_DOCUMENTS_FILE


class TFIDFModel:

    def __init__(self):

        self.vectorizer = TfidfVectorizer()
        self.doc_ids = []
        self.documents = []

        self.document_matrix = None

    def load_documents(self):

        with open(
            PROCESSED_DOCUMENTS_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            docs = json.load(f)

        for doc_id, doc_data in docs.items():

            self.doc_ids.append(doc_id)

            self.documents.append(
                doc_data["processed_text"]
            )

    def build_index(self):
        print("Loading processed documents...")

        self.load_documents()

        print("Building TF-IDF matrix...")

        self.document_matrix = (
            self.vectorizer.fit_transform(
                self.documents
            )
        )

        print(
            f"Indexed {len(self.documents)} documents"
        )

    def search(self, query, top_k=10):

        query_vector = (
            self.vectorizer.transform([query])
        )

        similarities = cosine_similarity(
            query_vector,
            self.document_matrix
        ).flatten()

        ranked = sorted(
            zip(self.doc_ids, similarities),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked[:top_k]