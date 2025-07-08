import sys
import os
import json
import numpy as np

from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from src.retriever.rankers.BGE_ranker import BGE_ranker
from src.retriever.rankers.CE_ranker import CE_ranker
from src.retriever.rankers.llm_ranker import LLM_Ranker
from src.retriever.BM_25.bm_25_retriever import BM_25


class Retriever:
    def __init__(self, 
                 db_path, 
                 embedder,
                 strategy='mmr',
                 ranker_name=None, 
                 ranker_model=None,
                 fusion_alpha=1, 
                 k = 10, 
                 bm25_k = 20,
                 rerank_k = None, 
                 has_answer_th = 0.,
                 
                 path_to_big_part_base=""):

        self.database = FAISS.load_local(db_path,
                             embedder,
                             allow_dangerous_deserialization=True)
        
        self.reranker = self.init_ranker(ranker_name, ranker_model)
        self.bm25 = BM_25()
        self.strategy = strategy
        
        self.fusion_alpha = fusion_alpha
        self.k = k
        self.bm25_k = bm25_k
        self.rerank_k = k if rerank_k is None else rerank_k
        self.has_answer_th = has_answer_th


        self.search_big_part_base = self._load_from_json(path_to_big_part_base)

    def init_ranker(self, ranker_name, ranker_model):
        if ranker_name == 'bge':
            return BGE_ranker(ranker_model)
        if ranker_name == 'ce':
            return CE_ranker(ranker_model)
        if ranker_name == 'llm':
            return LLM_Ranker(ranker_model)
        return None
    
    def rerank_docs(self, query, docs):
        if self.reranker is None:
            return np.zeros(len(docs))
        return self.reranker(query, docs)

    def retrieve(self, query):
        # query = self.get_advance_query(query)
        if self.strategy == 'mmr':  # Отделяем случай, тк в FAISS нет mmr_with_scores
            docs = self.database.max_marginal_relevance_search(query, self.rerank_k)
            scores = self.rerank_docs(query, docs)

        else:
            if self.fusion_alpha == 1:
                docs_with_scores = self.database.similarity_search_with_relevance_scores(query, self.rerank_k)

            # docs, fusion_scores = fusion_retrieval_block(self.database, query, self.strategy, self.fusion_alpha, self.rerank_k)
            # reranker_scores = self.rerank_docs(query, docs)
            docs = [doc for doc, _ in docs_with_scores]
            doc_scores =  [score for _, score in docs_with_scores]
            scores = self.rerank_docs(query, docs)
            # scores = reranker_scores + fusion_scores

        score_doc = list(zip(scores, docs))
        score_doc = sorted(score_doc, key=lambda x: x[0], reverse = True)
        score_doc = score_doc[:self.k]

        if score_doc[0][0] < self.has_answer_th:
            return None, None
        relevant_docs = [elem[1] for elem in score_doc]
        scores = [elem[0] for elem in score_doc]
        return self._get_big_parts(relevant_docs), doc_scores
    
    def _get_big_parts(self, relevant_docs):
        big_parts = []

        for doc in relevant_docs:

            metadata = doc.metadata

            filename = metadata.get('filename')
            part_id = metadata.get('part_id')

            big_parts.append(Document(page_content=self._get_big_part_content(filename, part_id), metadata=metadata))


        return big_parts


    def _get_big_part_content(self, file_name, part_id):
        # Пример: 'part_id': 3, 'chunk_index': '3.1'
        return self.get_big_part_text(file_name, part_id)


    def get_big_part_text(self, file_name, part_number):

        file_parts = self.search_big_part_base.get(file_name, {})

        return file_parts.get(str(part_number))
    

    def _load_from_json(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
