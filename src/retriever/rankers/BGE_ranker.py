from src.retriever.rankers.ranker import Ranker

class BGE_ranker(Ranker):
    def __init__(self, reranker_model):
        self.reranker_model = reranker_model

    def __call__(self, query, contexts):
        contexts = [context.page_content for context in contexts]
        scores = [self.reranker_model.compute_score([query, passage], normalize=True) for passage in contexts]
        return scores
