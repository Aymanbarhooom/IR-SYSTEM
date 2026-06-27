# evaluation/evaluator.py
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from services.searcher import SearchService
from evaluation.metrics import (
    precision_at_k, recall_at_k, 
    average_precision_at_k, ndcg_at_k
)

class IREvaluator:
    def __init__(self, model_type, use_refinement=False):
        self.model_type = model_type
        self.use_refinement = use_refinement
        self.search_service = None

    def _get_search_service(self):
        if self.search_service is None:
            self.search_service = SearchService(
                model_type=self.model_type, 
                use_refinement=self.use_refinement
            )
        return self.search_service

    def evaluate_single_query(self, args):
        query_id, query_text, qrels = args
        try:
            service = self._get_search_service()
            results = service.search(query_text, top_k=10)
            
            retrieved = [str(doc_id) for doc_id, _ in results]
            relevant = list(qrels.get(query_id, {}).keys())

            return {
                "query_id": query_id,
                "precision": precision_at_k(retrieved, relevant),
                "recall": recall_at_k(retrieved, relevant),
                "ap": average_precision_at_k(retrieved, relevant),
                "ndcg": ndcg_at_k(retrieved, qrels.get(query_id, {}))
            }
        except:
            return None

    def evaluate(self, num_queries=None, use_multiprocessing=True):
        with open("data/processed/processed_queries.json", "r", encoding="utf-8") as f:
            queries = json.load(f)
        
        with open("data/raw/qrels.json", "r", encoding="utf-8") as f:
            qrels = json.load(f)

        query_list = list(queries.items())[:num_queries] if num_queries else list(queries.items())

        print(f"Evaluating {len(query_list)} queries with {self.model_type}...")

        start = time.time()

        if use_multiprocessing and len(query_list) > 50:
            with ProcessPoolExecutor(max_workers=4) as executor:  # غير حسب جهازك
                futures = [
                    executor.submit(self.evaluate_single_query, (qid, data["original_query"], qrels))
                    for qid, data in query_list
                ]
                results = [f.result() for f in as_completed(futures)]
        else:
            results = [self.evaluate_single_query((qid, data["original_query"], qrels)) 
                      for qid, data in query_list]

        valid_results = [r for r in results if r]
        
        summary = {
            "model": self.model_type,
            "queries": len(valid_results),
            "precision@10": sum(r["precision"] for r in valid_results) / len(valid_results),
            "recall@10": sum(r["recall"] for r in valid_results) / len(valid_results),
            "map@10": sum(r["ap"] for r in valid_results) / len(valid_results),
            "ndcg@10": sum(r["ndcg"] for r in valid_results) / len(valid_results),
            "evaluation_time": round(time.time() - start, 2)
        }

        return summary, valid_results