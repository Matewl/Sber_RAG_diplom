from langchain.prompts import ChatPromptTemplate

from src.prompts import (STUFF_SYSTEM)
from structured_outputs import RAGAnswerTp
from src.tokenizer.embedder_tokenizer import EmbedderTokenizer

class RAG:
    def __init__(self, llm, retriever, enricher=None):
        self.llm = llm.with_structured_output(RAGAnswerTp)
        self.retriever = retriever
        self.enricher = enricher
    
    def get_context(self, query):
        if self.enricher:
            query = self.enricher.enrich(query)
        query = EmbedderTokenizer().cut_string_by_max_tokens(query)
        try:
            docs, scores = self.retriever.retrieve(query)
        except:
            print(query)
        return [doc.page_content for doc in docs], scores
    
    def generate_answer(self, query, context):
        chat_template = ChatPromptTemplate.from_messages([
            ('system', STUFF_SYSTEM),
            ('user', 'Вопрос:\n{question}\nОтрывки из документов:\n{context}')
        ])

        chain = (
            chat_template | 
            self.llm 
                    )
        answer = chain.invoke({'context': context, 'question': query})
        return answer.answer
    
    def get_answer(self, query):
        context, scores = self.get_context(query)
        return self.generate_answer(query, '\n'.join(context)), context, scores
    
    def __call__(self, query):
        return self.get_answer(query)
