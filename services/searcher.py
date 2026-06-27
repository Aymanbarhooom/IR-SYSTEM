# services/searcher.py

from services.preprocessor import Preprocessor
from models.embedding_model import (
    EmbeddingModel
)
from services.query_refinement import QueryRefinement
from models.hybrid_model import (
    HybridModel
)
from models.tfidf_model import TFIDFModel
from models.bm25_model import BM25Model


class SearchService:

    def __init__(self, model_type, use_refinement=False):

        self.preprocessor = Preprocessor()
        self.use_refinement = use_refinement
        if use_refinement:
            self.refiner = QueryRefinement()
        else:
            self.refiner = None
        self.model_type = model_type

        if model_type == "tfidf":

            self.model = TFIDFModel()

        elif model_type == "bm25":

            self.model = BM25Model()

        elif model_type == "embedding":

            self.model = EmbeddingModel()

        elif model_type == "hybrid_parallel":

            self.model = HybridModel(
                mode="parallel"
        )

        elif model_type == "hybrid_serial":

            self.model = HybridModel(
                mode="serial"
       )

        else:

            raise ValueError(
                f"Unknown model: {model_type}"
            )
        print(f"🔄 Building index for {model_type}...")
        self.model.build_index()

    def search(
            self,
            query,
            top_k=10
        ):
        
        if self.use_refinement:
            query = self.refiner.refine_query(query, method="both")
            print(f"✅ Refined query: '{query}'")

        if self.model_type in [
                "embedding",
                "hybrid_parallel",
                "hybrid_serial"
            ]:

            return self.model.search(
                query,
                top_k
            )
        if self.model_type == "hybrid":
            return self.model.search(query, top_k)

        processed_query = (
            self.preprocessor.preprocess_text(
                query
            )
        )

        return self.model.search(
            processed_query,
            top_k
        )