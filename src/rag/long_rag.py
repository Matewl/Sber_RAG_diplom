from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate

class RAG:
    def __init__(self, llm, retriever):
        self.llm = llm.with_structured_output(RAGAnswerTp)
        self.retriever = retriever
    
    def get_context(self, query):
        docs, scores = self.retriever.retrieve(query)
        return '\n'.join([doc.page_content for doc in docs]), scores
    
    def generate_answer(self, query, context):
        chat_template = ChatPromptTemplate.from_messages([
            ('system', STUFF_SYSTEM),
            ('user', 'Отрывки из документов:\n{context}\nВопрос:\n{question}')
        ])

        chain = (
            chat_template | 
            self.llm 
                    )
        answer = chain.invoke({'context': context, 'question': query})
        return answer.answer
    
    def get_answer(self, query):
        context, scores = self.get_context(query)
        return self.generate_answer(query, context), context, scores
    
    def __call__(self, query):
        return self.get_answer(query)
