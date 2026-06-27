# evaluation/metrics.py
import math

def precision_at_k(retrieved_docs, relevant_docs, k=10):

    retrieved_k = retrieved_docs[:k]

    relevant_retrieved = len(
        set(retrieved_k) & set(relevant_docs)
    )

    return relevant_retrieved / k

def recall_at_k(retrieved, relevant_docs, k):
    if not relevant_docs:
        return 0.0
    retrieved = retrieved[:k]
    relevant = set(retrieved) & set(relevant_docs)
    return len(relevant) / len(relevant_docs)



def average_precision_at_k(
    retrieved_docs,
    relevant_docs,
    k=10
):

    retrieved_k = retrieved_docs[:k]

    num_hits = 0
    precision_sum = 0

    for i, doc_id in enumerate(retrieved_k):

        if doc_id in relevant_docs:

            num_hits += 1

            precision_i = (
                num_hits / (i + 1)
            )

            precision_sum += precision_i

    if len(relevant_docs) == 0:
        return 0

    return (
        precision_sum
        /
        min(
            len(relevant_docs),
            k
        )
    )
def dcg_at_k(
    retrieved_docs,
    relevance_dict,
    k=10
):

    dcg = 0

    for i, doc_id in enumerate(
        retrieved_docs[:k]
    ):

        relevance = (
            relevance_dict.get(
                doc_id,
                0
            )
        )

        dcg += (
            relevance
            /
            math.log2(i + 2)
        )

    return dcg

def ndcg_at_k(
    retrieved_docs,
    relevance_dict,
    k=10
):

    dcg = dcg_at_k(
        retrieved_docs,
        relevance_dict,
        k
    )

    ideal_relevances = sorted(
        relevance_dict.values(),
        reverse=True
    )

    idcg = 0

    for i, rel in enumerate(
        ideal_relevances[:k]
    ):

        idcg += (
            rel
            /
            math.log2(i + 2)
        )

    if idcg == 0:
        return 0

    return dcg / idcg