import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import streamlit as st

from services.searcher import SearchService

from services.document_store import (
    DocumentStore
)

from services.query_refinement import (
    QueryRefinement
)

st.set_page_config(
    page_title="IR System",
    layout="wide"
)

st.title(
    "Information Retrieval System"
)

query = st.text_input(
    "Enter Query"
)

model = st.selectbox(
    "Choose Model",
    [
        "tfidf",
        "bm25",
        "embedding",
        "hybrid_parallel",
        "hybrid_serial"
    ]
)

use_refinement = st.checkbox(
    "Query Refinement",
    value=True
)

if st.button("Search"):

    if not query:

        st.warning(
            "Please enter a query."
        )

    else:

        @st.cache_resource
        def load_search_service(model):

            return SearchService(
                model_type=model,
                use_refinement=False
        )
        search_service = load_search_service(model)

        store = DocumentStore()

        corrected_query = query

        if use_refinement:

            refiner = QueryRefinement()

            refinement = (
                refiner.refine_query(
                    query
                )
            )

            corrected_query = (
                refinement["corrected"]
            )

            st.subheader(
                "Corrected Query"
            )

            st.write(
                corrected_query
            )

            st.subheader(
                "Suggestions"
            )

            for suggestion, score in (
                refinement["suggestions"]
            ):

                st.write(
                    f"• {suggestion}"
                )

        results = (
            search_service.search(
                corrected_query,
                top_k=10
            )
        )

        st.subheader(
            "Top 10 Results"
        )

        for rank, (
            doc_id,
            score
        ) in enumerate(
            results,
            start=1
        ):

            doc = store.get_document(
                doc_id
            )

            st.markdown(
                f"### {rank}"
            )

            st.write(
                f"Document ID: {doc_id}"
            )

            st.write(
                f"Score: {score:.4f}"
            )

            if doc:

                st.write(
                    doc[0][:1000]
                )

            st.divider()