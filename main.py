# main.py

from services.searcher import SearchService
from services.evaluator import Evaluator
import json
import time
from models.hybrid_model import HybridModel
from config import TOP_K

TEST_QUERY = "northern premier league"


def safe_score_str(score):
    try:
        return f"{float(score):.4f}"
    except Exception:
        return str(score)


def run_search():
    print("\nChoose model:")
    print("1 - TF-IDF")
    print("2 - BM25")
    print("3 - Embedding (BERT/FAISS)")
    print("4 - Hybrid")

    choice = input("\nYour choice: ").strip()

    models = {
        "1": "tfidf",
        "2": "bm25",
        "3": "embedding",
        "4": "hybrid"
    }

    model_type = models.get(choice)

    if not model_type:
        print("Invalid choice")
        return

    search_service = SearchService(model_type=model_type)

    print(f"\nQuery: {TEST_QUERY}")
    print(f"Model: {model_type}")

    results = search_service.search(TEST_QUERY, top_k=10)

    # If hybrid returns both modes as a dict, print both; otherwise print single list
    if isinstance(results, dict):
        # Parallel (RRF)
        print("\nParallel (RRF) Results:\n")
        for rank, (doc_id, score) in enumerate(results.get("parallel", []), start=1):
            print(f"{rank}. {doc_id} | {safe_score_str(score)}")

        # Serial (BM25 -> FAISS)
        print("\nSerial (BM25 -> FAISS) Results:\n")
        for rank, (doc_id, score) in enumerate(results.get("serial", []), start=1):
            print(f"{rank}. {doc_id} | {safe_score_str(score)}")
    else:
        # Single-list results (tfidf / bm25 / embedding)
        print("\nResults:\n")
        for rank, (doc_id, score) in enumerate(results, start=1):
            print(f"{rank}. {doc_id} | {safe_score_str(score)}")


def run_evaluation():
    models = [
        "embedding"
    ]

    for model in models:
        print("\n" + "=" * 60)
        print(f"Evaluating {model}")
        print("=" * 60)

        evaluator = Evaluator(model_type=model)
        results = evaluator.evaluate()

        print()
        for metric, value in results.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
                
def evaluate_hybrid_mode(mode="parallel", num_queries=50):
    """
   Hybrid Evaluating (parallel أو serial)
    """
    print(f"\n{'='*70}")
    print(f"📊  Hybrid Model - Mode: {mode.upper()}")
    print(f"{'='*70}")

    # إنشاء Hybrid Model مباشرة
    hybrid = HybridModel()
    hybrid.build_index()   # بناء أو تحميل الفهرس

    # تحميل الـ queries والـ qrels
    with open("data/processed/processed_queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)
    
    with open("data/raw/qrels.json", "r", encoding="utf-8") as f:   # تأكد من المسار
        qrels = json.load(f)

    precision_scores = []
    recall_scores = []
    ap_scores = []
    ndcg_scores = []
    evaluated = 0

    start_time = time.time()

    for i, (query_id, query_data) in enumerate(queries.items()):
        if evaluated >= num_queries:
            break
            
        if query_id not in qrels:
            continue

        query_text = query_data["original_query"]

        # البحث حسب الوضع
        if mode == "parallel":
            results_dict = hybrid.search(query_text, top_k=TOP_K)
            results = results_dict["parallel"]
        else:  # serial
            results_dict = hybrid.search(query_text, top_k=TOP_K)
            results = results_dict["serial"]

        retrieved_docs = [str(doc_id) for doc_id, _ in results]

        relevant_docs = list(qrels[query_id].keys())

        # حساب المقاييس
        from evaluation.metrics import (
            precision_at_k, recall_at_k, 
            average_precision_at_k, ndcg_at_k
        )

        precision = precision_at_k(retrieved_docs, relevant_docs, TOP_K)
        recall = recall_at_k(retrieved_docs, relevant_docs, TOP_K)
        ap = average_precision_at_k(retrieved_docs, relevant_docs, TOP_K)
        ndcg = ndcg_at_k(retrieved_docs, qrels[query_id], TOP_K)

        precision_scores.append(precision)
        recall_scores.append(recall)
        ap_scores.append(ap)
        ndcg_scores.append(ndcg)
        
        evaluated += 1

        if evaluated % 10 == 0:
            print(f"   Processed {evaluated}/{num_queries} queries...")

    end_time = time.time()

    # النتائج النهائية
    print(f"\n✅ Finished {mode.upper()} Hybrid:")
    print(f"   Queries Evaluated : {evaluated}")
    print(f"   Precision@{TOP_K}   : {sum(precision_scores)/len(precision_scores):.4f}")
    print(f"   Recall@{TOP_K}      : {sum(recall_scores)/len(recall_scores):.4f}")
    print(f"   MAP@{TOP_K}         : {sum(ap_scores)/len(ap_scores):.4f}")
    print(f"   nDCG@{TOP_K}        : {sum(ndcg_scores)/len(ndcg_scores):.4f}")
    print(f"   Time Taken         : {end_time - start_time:.2f} seconds")


def main():
    print("\nIR Project 2026")
    print("----------------------")

    print("\n1 - Search")
    print("2 - Evaluate All Models")
    print("3 - Evaluate Hybrid")

    choice = input("\nChoose option: ").strip()

    if choice == "1":
        run_search()
    elif choice == "2":
        run_evaluation()
    elif choice == "3":
        evaluate_hybrid_mode(mode="parallel", num_queries=50)
        evaluate_hybrid_mode(mode="serial", num_queries=50)
    else:
        print("Invalid option")


if __name__ == "__main__":
    main()
