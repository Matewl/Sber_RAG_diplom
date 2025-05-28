from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.query_enrichment.prompts import MULTIPLE_QUERIES_SYSTEM_PROMPT, MULTIPLE_QUERIES_USER_PROMPT
from src.query_enrichment.enrichment_blocks.query_enricher import QueryEnrichmentBlock
from src.query_enrichment.qe_utils import EnrichedQuery
from src.query_enrichment.templates import RephrasedQuestionsList
from src.question_answering import QA

class RephraseEnricher(QueryEnrichmentBlock):  
    def __init__(self, qa: QA):
        self.qa: QA = qa 

    def enrich_query(self, query):
        rephrased_queries = self.qa.retrieve_answer(question=query)
        return '\n'.join([q.question for q in rephrased_queries.questions])
     

