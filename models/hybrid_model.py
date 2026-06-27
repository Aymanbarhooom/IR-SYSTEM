# models/hybrid_model.py

from config import TOP_K
from models.bm25_model import BM25Model
from models.embedding_faiss import FAISSEmbeddingModel


class HybridModel:

    def __init__(self, mode="parallel"):

        self.mode = mode.lower()

        self.bm25 = BM25Model()
        self.embedding = FAISSEmbeddingModel()

        print(
            f"🔀 Hybrid Model initialized ({self.mode.upper()})"
        )

    def build_index(self):

        print("🔨 Building Hybrid Index...")

        self.bm25.build_index()

        if getattr(self.embedding, "index", None) is None:
            self.embedding.build_index()

        print(
            f"✅ Hybrid Index ready ({len(self.bm25.doc_ids):,} documents)"
        )

    def search(
        self,
        query,
        top_k=TOP_K
    ):

        if self.mode == "parallel":

            return self.parallel_search(
                query,
                top_k
            )

        return self.serial_search(
            query,
            top_k
        )

    # --------------------------------------------------------
    # Parallel Hybrid (Reciprocal Rank Fusion)
    # --------------------------------------------------------

    def parallel_search(
        self,
        query,
        top_k
    ):

        bm25_results = self.bm25.search(
            query,
            top_k * 8
        )

        embedding_results = self.embedding.search(
            query,
            top_k * 8
        )

        scores = {}

        RRF_K = 60

        for rank, (doc_id, _) in enumerate(
            bm25_results,
            start=1
        ):

            scores[str(doc_id)] = (
                scores.get(str(doc_id), 0)
                + 1 / (rank + RRF_K)
            )

        for rank, (doc_id, _) in enumerate(
            embedding_results,
            start=1
        ):

            scores[str(doc_id)] = (
                scores.get(str(doc_id), 0)
                + 1 / (rank + RRF_K)
            )

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked[:top_k]

    # --------------------------------------------------------
    # Serial Hybrid
    # BM25 --> Candidate Generation
    # Embedding --> Re-ranking
    # --------------------------------------------------------

    def serial_search(
        self,
        query,
        top_k
    ):

        candidates = self.bm25.search(
            query,
            300
        )

        if not candidates:

            return []

        candidate_ids = {
            str(doc_id)
            for doc_id, _ in candidates
        }

        embedding_results = self.embedding.search(
            query,
            500
        )

        reranked = []

        for doc_id, score in embedding_results:

            if str(doc_id) in candidate_ids:

                reranked.append(
                    (
                        str(doc_id),
                        float(score)
                    )
                )

        return reranked[:top_k]