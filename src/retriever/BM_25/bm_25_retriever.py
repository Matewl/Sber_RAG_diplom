from rank_bm25 import BM25Okapi
import numpy as np 
from src.retriever.BM_25.utils import clean_text


class BM_25: 
    def get_documents_with_scores(self, query, documents):

        clean_documents = [clean_text(doc.page_content) for doc in documents]
        BMdb = BM25Okapi(clean_documents)

        clean_query = clean_text(query)
        bm25_scores = BMdb.get_scores(clean_query)

        if np.max(bm25_scores) - np.min(bm25_scores) == 0:
            bm25_scores = [0] * len(bm25_scores)
        else:
            bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))

        return bm25_scores