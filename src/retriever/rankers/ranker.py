from typing import List

class Ranker:
    def __call__(self, query: str, contexts: List[str]) -> List[float]:
        return [0.] * len(contexts)