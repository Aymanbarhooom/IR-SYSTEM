# services/query_refinement.py

import json

from spellchecker import SpellChecker

from nltk.corpus import wordnet

from sentence_transformers import (
    SentenceTransformer
)

from sklearn.metrics.pairwise import (
    cosine_similarity
)

from config import (
    PROCESSED_QUERIES_FILE
)


class QueryRefinement:

    def __init__(self):

        self.spell = SpellChecker()

        print("Loading query suggestions model...")

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        with open(
            PROCESSED_QUERIES_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            self.queries = json.load(f)

        self.query_texts = [
            q["original_query"]
            for q in self.queries.values()
        ]

        print("Encoding queries...")

        self.query_embeddings = (
            self.model.encode(
                self.query_texts,
                show_progress_bar=True
            )
        )

    # ----------------------------------
    # 1 Spell Correction
    # ----------------------------------

    def correct_query(
        self,
        query
    ):

        words = query.split()

        corrected = []

        for word in words:

            corrected.append(
                self.spell.correction(word)
            )

        return " ".join(corrected)

    # ----------------------------------
    # 2 WordNet Expansion
    # ----------------------------------

    def expand_query_wordnet(
        self,
        query,
        max_synonyms=2
    ):

        expanded = []

        for word in query.split():

            expanded.append(word)

            synonyms = set()

            for syn in wordnet.synsets(word):

                for lemma in syn.lemma_names():

                    lemma = lemma.replace(
                        "_",
                        " "
                    )

                    if lemma.lower() != word.lower():

                        synonyms.add(
                            lemma.lower()
                        )

            synonyms = list(
                synonyms
            )[:max_synonyms]

            expanded.extend(
                synonyms
            )

        return " ".join(expanded)

    # ----------------------------------
    # 3 Semantic Suggestions
    # ----------------------------------

    def suggest_queries(
        self,
        query,
        top_k=5
    ):

        query_embedding = (
            self.model.encode(
                [query]
            )
        )

        similarities = (
            cosine_similarity(
                query_embedding,
                self.query_embeddings
            )[0]
        )

        ranked = sorted(
            zip(
                self.query_texts,
                similarities
            ),
            key=lambda x: x[1],
            reverse=True
        )

        suggestions = []

        for q, score in ranked:

            if q.lower() != query.lower():

                suggestions.append(
                    (q, round(score, 4))
                )

            if len(suggestions) >= top_k:

                break

        return suggestions

    # ----------------------------------
    # Complete Pipeline
    # ----------------------------------

    def refine_query(
        self,
        query
    ):

        corrected = (
            self.correct_query(
                query
            )
        )

        expanded = (
            self.expand_query_wordnet(
                corrected
            )
        )

        suggestions = (
            self.suggest_queries(
                corrected
            )
        )

        return {

            "original": query,

            "corrected": corrected,

            "expanded": expanded,

            "suggestions": suggestions
        }