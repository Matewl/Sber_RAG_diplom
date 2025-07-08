from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.filtration.prompts import (FILTRATION_MAP_REDUCE_SYSTEM,
                                    FILTRATION_MAP_REDUCE_USER)
from src.filtration.templates import FilteredContext

class MapReduceFiltration:
    def __init__(self, llm):
        self.map_reduce_chain = ChatPromptTemplate.from_messages([('system', FILTRATION_MAP_REDUCE_SYSTEM),
                                                       ('user', FILTRATION_MAP_REDUCE_USER)]) | llm.with_structured_output(FilteredContext)
    def filter(self, question, contexts):
        filter_verdicts = [self.map_reduce_chain.invoke({
            "question": question, 
            "context": context
        }) for context in contexts]

        return [context for context, verdict in zip(contexts, filter_verdicts) if verdict.verdict == 1]