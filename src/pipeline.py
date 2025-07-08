from omegaconf import OmegaConf

from src.llm.giga_chat import GigaChatLLM
from src.query_enrichment.query_enrichment import QueryEnrichment
from src.retriever.retriever import Retriever
# from src.embedder.embedder_wrappers import Embedder_wrapper_e5_giga
from src.embedder.embedder import Embedder
from src.rag.vanila_rag import RAG
from src.rag.adaptive_rag.adaptive_rag import AdaptiveRAG

from src.summarization.stuff import StuffSummarizator
from src.summarization.map_reduce import MapReduceSummarizator
from src.filtration.map_reduce import MapReduceFiltration
from src.filtration.map_reduce_soft import MapReduceSoftFiltration


class Searcher:
    def __init__(self, cfg_path = 'configs/deployment_config.py'):
        self.cfg = OmegaConf.load(cfg_path)
        self.llm = GigaChatLLM(self.cfg['llm'])
        self.embedder = Embedder(self.cfg['embedder'])
        self.query_enricher = QueryEnrichment(self.llm, [name for name in self.cfg['qe'] if self.cfg['qe'][name]])

        self.retriever = Retriever(self.cfg['retriever']['db_path'],
                                        embedder=self.embedder,
                                        strategy=self.cfg['retriver']['strategy'],
                                        ranker_name=self.cfg['retriver'].get("ranker_name", None),
                                        ranker_model=self.cfg['retriver'].get("ranker_model", None),
                                        fusion_alpha=self.cfg['retriver'].get("fusion_alpha", 1),
                                        k=self.cfg['retriver'].get("k", 5),
                                        bm25_k=self.cfg['retriver'].get("bm25_k", 10),
                                        rerank_k=self.cfg['retriver'].get("rerank_k", 20),
                                        has_answer_th=self.cfg['retriver'].get("has_answer_th", 0)
                                   )
        
        summarizators = {
            'stuff': StuffSummarizator(self.llm),
            'map_reduce': MapReduceSummarizator(self.llm)
        }

        filters = {
            'map_reduce': MapReduceFiltration(self.llm),
            'map_reduce_soft': MapReduceSoftFiltration(self.llm)
        }

        self.summarizator = summarizators[self.cfg['summarization']['type']] if self.cfg['summarization']['enable'] else None
        self.filter = filters[self.cfg['filtering']['type']] if self.cfg['filtering ']['enable'] else None

        RAGs = {
            'vanilla': RAG(self.llm, 
                           self.retriever,
                           self.summarizator,
                           self.filter),
            'adaptive': AdaptiveRAG(self.llm,
                                    self.retriever,
                                    self.summarizator,
                                    self.filter,
                                    self.cfg['rag']['adaptive']['n_loops'],
                                    self.cfg['rag']['adaptive']['n_retries'],
                                    self.cfg['rag']['adaptive']['use_rephrase'])
            }
        
        self.rag = RAGs[self.cfg['rag']['type']]
    
    def search(self, question):
        enriched_query = self.query_enricher.enrich(question) if self.query_enricher else ''
        question = question + '\n' + enriched_query
        return self.rag.get_answer(question)[0]


searcher = Searcher()


