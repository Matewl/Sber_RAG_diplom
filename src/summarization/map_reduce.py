from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.summarization.prompts import (SUMMARIZATION_STUFF_SYSTEM,
                                       SUMMARIZATION_STUFF_USER,
                                       SUMMARIZATION_MAP_REDUCE_SYSTEM,
                                       SUMMARIZATION_MAP_REDUCE_USER)

class MapReduceSummarizator:
    def __init__(self, llm):
        self.chain = ChatPromptTemplate.from_messages([('system', SUMMARIZATION_STUFF_SYSTEM),
                                                       ('user', SUMMARIZATION_STUFF_USER)]) | llm | StrOutputParser()
        self.map_reduce_chain = ChatPromptTemplate.from_messages([('system', SUMMARIZATION_MAP_REDUCE_SYSTEM),
                                                       ('user', SUMMARIZATION_MAP_REDUCE_USER)]) | llm | StrOutputParser()
    def summarize(self, question, contexts):
        context_summarizes = [self.map_reduce_chain.invoke({
            "question": question, 
            "context": context
        }) for context in contexts]

        return self.chain.invoke({
            "question": question,
            "context": "\n".join(context_summarizes)
        })