import csv
import json

from config import (
    DOCUMENTS_CSV,
    TRAIN_QUERIES,
    TRAIN_QRELS,
    DOCUMENTS_FILE,
    QUERIES_FILE,
    QRELS_FILE
)


class LocalDataLoader:

    def load_documents(self):

        documents = {}

        with open(
            DOCUMENTS_CSV,
            "r",
            encoding="utf-8"
        ) as f:

            reader = csv.DictReader(f)

            for row in reader:

                documents[row["id_right"]] = {
                    "text": row["text_right"]
                }

        return documents

    def load_queries(self):

        queries = {}

        with open(
            TRAIN_QUERIES,
            "r",
            encoding="utf-8"
        ) as f:

            reader = csv.DictReader(f)

            for row in reader:

                queries[row["id_left"]] = (
                    row["text_left"]
                )

        return queries

    def load_qrels(self):

        qrels = {}

        with open(
            TRAIN_QRELS,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                parts = line.strip().split()

                query_id = parts[0]
                doc_id = parts[2]
                relevance = int(parts[3])

                if query_id not in qrels:
                    qrels[query_id] = {}

                qrels[query_id][doc_id] = (
                    relevance
                )

        return qrels

    def save_json(self, data, path):

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f,
                indent=4,
                ensure_ascii=False
            )

    def run(self):

        docs = self.load_documents()
        queries = self.load_queries()
        qrels = self.load_qrels()

        self.save_json(
            docs,
            DOCUMENTS_FILE
        )

        self.save_json(
            queries,
            QUERIES_FILE
        )

        self.save_json(
            qrels,
            QRELS_FILE
        )

        print(
            f"Documents: {len(docs)}"
        )

        print(
            f"Queries: {len(queries)}"
        )

        print(
            f"Qrels: {len(qrels)}"
        )