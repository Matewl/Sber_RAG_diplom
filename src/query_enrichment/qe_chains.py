from typing import Dict

from langchain_gigachat import GigaChat
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, Runnable
from langchain_core.prompts import ChatPromptTemplate


from src.query_enrichment.prompts import (MULTIPLE_DOCS_SYSTEM_PROMPT, MULTIPLE_DOCS_USER_PROMPT,
                                                 HYDE_SYSTEM_PROMPT, HYDE_USER_PROMPT,
                                                 MULTIPLE_QUERIES_SYSTEM_PROMPT, MULTIPLE_QUERIES_USER_PROMPT,
                                                 STEPBACK_SYSTEM_PROMPT, STEPBACK_USER_TEMPLATE, 
                                                 EXTRACT_ABBS_FROM_Q_SYSTEM, EXTRACT_ABBS_FROM_Q_USER_TEMPLATE, 
                                                 CHECK_ABB_TERM_RELEVANCE_SYSTEM, GRADE_USER_TEMPLATE, 
                                                 REPHRASE_W_ABBS_SYSTEM, REPHRASE_W_ABBS_USER_TEMPLATE)

from src.query_enrichment.templates import (AnswerQuestionsList,
                                                   RephrasedQuestionsList)


def create_chains(llm: GigaChat) -> Dict:
    return {
        "answer": ChatPromptTemplate.from_messages([('system', MULTIPLE_DOCS_SYSTEM_PROMPT),  
                                                   ('user', MULTIPLE_DOCS_USER_PROMPT)]) 
                                                   | llm.with_structured_output(AnswerQuestionsList),

        "hyde": ChatPromptTemplate.from_messages([('system', HYDE_SYSTEM_PROMPT),  
                                                   ('user', HYDE_USER_PROMPT)]) 
                                                   | llm
                                                   | StrOutputParser(),

        "rephrase": ChatPromptTemplate.from_messages([('system', MULTIPLE_QUERIES_SYSTEM_PROMPT),  
                                                   ('user', MULTIPLE_QUERIES_USER_PROMPT)]) 
                                                   | llm.with_structured_output(RephrasedQuestionsList),

        "step_back": ChatPromptTemplate.from_messages([('system', STEPBACK_SYSTEM_PROMPT),  
                                                   ('user', STEPBACK_USER_TEMPLATE)]) 
                                                   | llm
                                                   | StrOutputParser(),

        "extract_abbs_from_q": ChatPromptTemplate.from_messages([('system', EXTRACT_ABBS_FROM_Q_SYSTEM),
                                                                 ('user', EXTRACT_ABBS_FROM_Q_USER_TEMPLATE)]) 
                                                                 | llm 
                                                                 | StrOutputParser(),

        "check_abb_term_relevance":  ChatPromptTemplate.from_messages([('system', CHECK_ABB_TERM_RELEVANCE_SYSTEM),
                                                                       ('user', GRADE_USER_TEMPLATE)]) 
                                                                       | llm 
                                                                       | StrOutputParser(),

        "rephrase_w_abbs":  ChatPromptTemplate.from_messages([('system', REPHRASE_W_ABBS_SYSTEM),
                                                                       ('user', REPHRASE_W_ABBS_USER_TEMPLATE)]) 
                                                                       | llm 
                                                                       | StrOutputParser(),

    }


