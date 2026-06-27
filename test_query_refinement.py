# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 13:53:10 2026

@author: Ayman
"""

# test_query_refinement.py

from services.query_refinement import (
    QueryRefinement
)
def main():
    refiner = QueryRefinement()

    query = "local governmint in the"

    result = refiner.refine_query(
        query
    )

    print("\nOriginal:")
    print(
        result["original"]
    )

    print("\nCorrected:")
    print(
        result["corrected"]
    )

    print("\nExpanded:")
    print(
        result["expanded"]
    )

    print("\nSuggestions:")
    for q, score in result["suggestions"]:

        print(
            f"{q} ({score})"
        )
if __name__ == "__main__":
    main()
