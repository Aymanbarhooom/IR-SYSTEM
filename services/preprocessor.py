# services/preprocessor.py

import json
import string

import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer  

from config import (
    DOCUMENTS_FILE,
    QUERIES_FILE,
    PROCESSED_DOCUMENTS_FILE,
    PROCESSED_QUERIES_FILE
)


class Preprocessor:

    def __init__(self):  # Changed from 'init' to standard Python '__init__'
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()  # Changed to WordNetLemmatizer

    def preprocess_text(self, text):

        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [
            token
            for token in tokens
            if token not in string.punctuation
        ]

        tokens = [
            token
            for token in tokens
            if token.isalpha()
        ]

        tokens = [
            token
            for token in tokens
            if token not in self.stop_words
        ]

        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
        ]

        return " ".join(tokens)

    def load_json(self, filepath):

        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_json(self, data, filepath):

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=4
            )

    def process_documents(self):

        documents = self.load_json(DOCUMENTS_FILE)

        processed_docs = {}

        total = len(documents)

        for i, (doc_id, doc_data) in enumerate(documents.items()):

            original_text = doc_data["text"]

            processed_text = self.preprocess_text(
                original_text
            )

            processed_docs[doc_id] = {
                "processed_text": processed_text
            }

            if (i + 1) % 500 == 0:
                print(f"{i+1}/{total} documents processed")

        self.save_json(
            processed_docs,
            PROCESSED_DOCUMENTS_FILE
        )

        print("Processed documents saved.")

    def process_queries(self):

        queries = self.load_json(QUERIES_FILE)

        processed_queries = {}

        total = len(queries)

        for i, (query_id, query_text) in enumerate(
            queries.items()
        ):

            processed_queries[query_id] = {
                "original_query": query_text,
                "processed_query": self.preprocess_text(
                    query_text
                )
            }

            if (i + 1) % 100 == 0:
                print(f"{i+1}/{total} queries processed")

        self.save_json(
            processed_queries,
            PROCESSED_QUERIES_FILE
        )

        print("Processed queries saved.")

    def run(self):

        print("Processing documents...")
        self.process_documents()

        print("Processing queries...")
        self.process_queries()

        print("Preprocessing completed.")
