from abc import ABC, abstractmethod

class QueryEnrichmentBlock(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass
    
    @abstractmethod
    def enrich_query(self, query, **kwargs):
        pass
