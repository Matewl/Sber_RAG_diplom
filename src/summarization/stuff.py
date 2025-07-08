from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.summarization.prompts import SUMMARIZATION_STUFF_SYSTEM, SUMMARIZATION_STUFF_USER

class StuffSummarizator:
    def __init__(self, llm):
        self.chain = ChatPromptTemplate.from_messages([('system', SUMMARIZATION_STUFF_SYSTEM),
                                                       ('user', SUMMARIZATION_STUFF_USER)]) | llm | StrOutputParser()
    
    def summarize(self, question, contexts):
        return self.chain.invoke({
            "question": question,
            "context": "\n".join(contexts)
        })