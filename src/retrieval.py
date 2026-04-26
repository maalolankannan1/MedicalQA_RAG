import numpy as np
from collections import defaultdict
from langchain_core.documents import Document

import sys
sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import config


def get_cosine_retriever(vector_store, k=None):
    k = k or config.TOP_K
    return vector_store.as_retriever(search_kwargs={"k": k})


def get_mmr_retriever(vector_store, k=None, fetch_k=None, lambda_mult=None):
    k = k or config.TOP_K
    fetch_k = fetch_k or config.MMR_FETCH_K
    lambda_mult = lambda_mult or config.MMR_LAMBDA_MULT
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
    )


class BM25Retriever:
    def __init__(self, documents, k=None):
        from rank_bm25 import BM25Okapi

        self.k = k or config.TOP_K
        self.documents = documents
        corpus = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(corpus)

    def invoke(self, query):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][: self.k]
        return [self.documents[i] for i in top_indices]


class HybridRetriever:
    def __init__(self, dense_retriever, sparse_retriever, alpha=None, k=None):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.alpha = alpha or config.HYBRID_ALPHA
        self.k = k or config.TOP_K

    def invoke(self, query):
        dense_results = self.dense_retriever.invoke(query)
        sparse_results = self.sparse_retriever.invoke(query)
        return self._reciprocal_rank_fusion(dense_results, sparse_results)

    def _reciprocal_rank_fusion(self, dense_results, sparse_results):
        scores = defaultdict(float)
        doc_map = {}

        for rank, doc in enumerate(dense_results):
            key = doc.page_content
            scores[key] += self.alpha * (1 / (rank + 1))
            doc_map[key] = doc

        for rank, doc in enumerate(sparse_results):
            key = doc.page_content
            scores[key] += (1 - self.alpha) * (1 / (rank + 1))
            doc_map[key] = doc

        sorted_keys = sorted(scores, key=lambda x: -scores[x])
        return [doc_map[k] for k in sorted_keys[: self.k]]
