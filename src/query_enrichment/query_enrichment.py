from typing import Dict

from src.query_enrichment.enrichment_blocks import (HyDE, 
                                                    RephraseEnricher, 
                                                    StepBackEnricher, 
                                                    AnswerEnricher)

from src.query_enrichment.qe_utils import EnrichedQuery, type_to_process_map
from src.query_enrichment.qe_chains import create_chains
from src.query_enrichment.enrichment_blocks.query_enricher import QueryEnrichmentBlock
from src.question_answering import QA


class QueryEnrichment:
    def __init__(self, llm, block_names):
        
        chains = create_chains(llm)


        self.enrichers: Dict[str, QueryEnrichmentBlock] = {
                                                            'hyde': HyDE(QA(chains['hyde'], type_to_process_map['hyde'])),
                                                            'rephrase': RephraseEnricher(QA(chains['rephrase'], type_to_process_map['rephrase'])), 
                                                            'step_back': StepBackEnricher(QA(chains['step_back'], type_to_process_map['step_back'])),
                                                            'answer': AnswerEnricher(QA(chains['answer'], type_to_process_map['answer']))
                                                            }
        
        self.block_names = block_names
        
    def enrich(self, query):
        enriched = [query]
        for enrichment_block_name in self.block_names:
            enriched_query = self.enrichers[enrichment_block_name].enrich_query(query)
            enriched.append(enriched_query)
        
        return '\n'.join(enriched)