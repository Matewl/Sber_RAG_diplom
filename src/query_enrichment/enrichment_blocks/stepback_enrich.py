from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.query_enrichment.prompts import STEPBACK_SYSTEM_PROMPT, STEPBACK_USER_TEMPLATE
from src.query_enrichment.enrichment_blocks.query_enricher import QueryEnrichmentBlock
from src.query_enrichment.qe_utils import EnrichedQuery
from src.question_answering import QA

class StepBackEnricher(QueryEnrichmentBlock): 
    def __init__(self, qa: QA):
        self.qa: QA = qa

    def enrich_query(self, query) :
        return self.qa.retrieve_answer(question=query)
    


