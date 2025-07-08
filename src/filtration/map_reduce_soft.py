from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.filtration.prompts import (FILTRATION_MAP_REDUCE_SOFT_SYSTEM,
                                    FILTRATION_MAP_REDUCE_SOFT_USER)

class MapReduceSoftFiltration:
    def __init__(self, llm):
        self.map_reduce_chain = ChatPromptTemplate.from_messages([('system', FILTRATION_MAP_REDUCE_SOFT_SYSTEM),
                                                       ('user', FILTRATION_MAP_REDUCE_SOFT_USER)]) | llm | StrOutputParser()
    def filter(self, question, contexts):
        filter_contects = [self.map_reduce_chain.invoke({
            "question": question, 
            "context": context
        }) for context in contexts]

        return filter_contects