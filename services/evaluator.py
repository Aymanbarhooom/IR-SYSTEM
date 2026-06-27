# services/evaluator.py

import json
import time

from config import (
    PROCESSED_QUERIES_FILE,
    QRELS_FILE,
    TOP_K
)

from services.searcher import SearchService

from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    average_precision_at_k,
    ndcg_at_k
)


class Evaluator:

    def __init__(self, model_type):

        self.model_type = model_type

        self.search_service = SearchService(
            model_type=model_type
        )

    def load_queries(self):

        with open(
            PROCESSED_QUERIES_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    def load_qrels(self):

        with open(
            QRELS_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    def evaluate(
        self,
        max_queries=None
    ):

        queries = self.load_queries()
        qrels = self.load_qrels()

        precision_scores = []
        recall_scores = []
        map_scores = []
        ndcg_scores = []

        evaluated_queries = 0
        total = len(queries)

        start = time.time()

        for i, (query_id, query_data) in enumerate(queries.items(), start=1):

            if max_queries is not None and evaluated_queries >= max_queries:
                break

            if query_id not in qrels:
                continue

            query_text = query_data["original_query"]

            results = self.search_service.search(
                query_text,
                top_k=TOP_K
            )

            # ---------- Hybrid ----------
            if self.model_type == "hybrid":

                results = results["parallel"]

            retrieved_docs = [
                doc_id
                for doc_id, score in results
            ]

            relevant_dict = qrels[query_id]
            relevant_docs = list(relevant_dict.keys())

            precision_scores.append(
                precision_at_k(
                    retrieved_docs,
                    relevant_docs,
                    TOP_K
                )
            )

            recall_scores.append(
                recall_at_k(
                    retrieved_docs,
                    relevant_docs,
                    TOP_K
                )
            )

            map_scores.append(
                average_precision_at_k(
                    retrieved_docs,
                    relevant_docs,
                    TOP_K
                )
            )

            ndcg_scores.append(
                ndcg_at_k(
                    retrieved_docs,
                    relevant_dict,
                    TOP_K
                )
            )

            evaluated_queries += 1

            if evaluated_queries % 100 == 0:

                elapsed = time.time() - start

                print(
                    f"[{evaluated_queries}/{total}] "
                    f"{elapsed:.1f} sec"
                )

        elapsed = time.time() - start

        return {

            "model": self.model_type,

            "queries": evaluated_queries,

            f"precision@{TOP_K}":
                sum(precision_scores) /
                len(precision_scores),

            f"recall@{TOP_K}":
                sum(recall_scores) /
                len(recall_scores),

            f"MAP@{TOP_K}":
                sum(map_scores) /
                len(map_scores),

            f"nDCG@{TOP_K}":
                sum(ndcg_scores) /
                len(ndcg_scores),

            "time (sec)":
                round(elapsed, 2),

            "queries/sec":
                round(
                    evaluated_queries / elapsed,
                    2
                )
        }