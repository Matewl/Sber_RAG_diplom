from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

from src.rag.vanila_rag import RAG
from src.rag.prompts import (REPHRASE_PROMPT, 
                         IS_COMPLETE_PROMPT)
from src.rag.structured_outputs import AdaptiveTp
from src.tokenizer import EmbedderTokenizer

class AdaptiveRAG(RAG):
    def __init__(self, llm, retriever, summarizer=None, filter=None, n_loops=5, n_retries=10, use_rephrase=False):
        super().__init__(llm, retriever, summarizer, filter)
        
        self.is_comlete_llm = llm.with_structured_output(AdaptiveTp)
        self.rephrase_llm = llm
        self.n_loops = n_loops
        self.n_retries = n_retries
        self.use_rephrase = use_rephrase

    def get_answer(self, query):
        is_complete_template = ChatPromptTemplate.from_messages([
            ('system', IS_COMPLETE_PROMPT),
            ('user', 'Вопрос:\n{question}\nОтрывки из документов:\n{context}')
        ])

        is_complete_chain = (
            is_complete_template | 
            self.is_comlete_llm         
        )

        rephrase_template = ChatPromptTemplate.from_messages([
            ('system', REPHRASE_PROMPT),
            ('user', 'Вопрос:\n{question}\nОтвет эксперта:\n{answer}')
        ])

        rephrase_chain = (
            rephrase_template | 
            self.rephrase_llm ,
            StrOutputParser())

        documents_history = []
        answers_history = []
        queries_history = [query]
        for _ in range(self.n_loops):
            for _ in range(self.n_retries):
                # try:
                    current_context = [document for document in documents_history] + answers_history
                    if len(answers_history) == 0:
                        retrieved_documents, _ = self.retriever.retrieve('\n'.join(queries_history + answers_history))
                    else:
                        new_context = []
                        for q, a in zip(queries_history[::-1] + answers_history[::-1]):
                            new_context.append(q)
                            new_context.append(a)
                        
                        new_context = '\n'.join(new_context)
                        new_context = EmbedderTokenizer.cut_string_by_max_tokens(new_context)
                        retrieved_documents, _ = self.retriever.retrieve(new_context)
                    
                    retrieved_documents = [document.page_content for document in retrieved_documents if document not in current_context]

                    llm_answer = self.generate_answer(query, '\n'.join(retrieved_documents + current_context))
                    answers_history.append(llm_answer)

                    is_complete = is_complete_chain.invoke({'context': retrieved_documents + current_context, 'question': query}).answer
                    if is_complete:
                        return llm_answer, retrieved_documents + current_context
                    
                    documents_history.extend(retrieved_documents)
                    answers_history.append(llm_answer)
                    if self.use_rephrase:
                        new_query = rephrase_chain.invoke({'question': query, 'answer': llm_answer})
                        queries_history.append(new_query)
                    break
                
                # except Exception as e:
                #     print(e)
                #     continue
        try:
            return llm_answer, retrieved_documents + current_context
        except:
            return None
