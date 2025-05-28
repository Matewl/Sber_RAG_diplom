from src.retriever.rankers.ranker import Ranker
class CE_ranker(Ranker):
    def __init__(self, reranker_model):
        self.reranker_model = reranker_model

    def __call__(self, query, contexts):
        contexts = [context.page_content for context in contexts]
        rank_result = self.reranker_model.rank(query, contexts)
        scores = [res['score'] for res in rank_result]
        return scores
