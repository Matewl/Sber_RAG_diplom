from pydantic import BaseModel, Field, confloat
from langchain.prompts import ChatPromptTemplate
from src.retriever.rankers.ranker import Ranker

class DocumentRankerTp(BaseModel):
    """Оценка полученных документов"""
    judgement: str = Field(description="Рассуждения о том, насколько хорошо можно ответить на вопрос на основе полученного контекста.")
    final_score: int = Field(description="Оценка контекста. От 1 до 100.")


class LLM_Ranker(Ranker):
    def __init__(self, llm):
        self.llm = llm.with_structured_output(DocumentRankerTp)

    def __call__(self, query, contexts):
        RANK_PROMPT = f"""Ты - эксперт по методологии кредитного процесса Сбербанка.
Тебе будет дан вопрос по документам в банковской сфере. Также тебе будет дан контекст, в котором может содержаться информация для ответа на вопрос.
Оцени по шкале от 1 до 100, насколько хорошо данный контекст соответствует вопросу.
Если в контексте мало информации, которая может помочь ответить на вопрос, оценка должна быть ближе к 1.
Если в контексте есть вся информация для ответа на вопрос, оценка должна быть ближе к 100.
"""

        rank_template = ChatPromptTemplate.from_messages([
            ('system', RANK_PROMPT),
            ('user', 'Вопрос:\n{question}\Контекст:\n{context}')
        ])

        rank_chain = (
            rank_template | 
            self.llm 
        )


        scores = []
        for context in contexts:
            model_answer = rank_chain.invoke({'question': query, 'context': context}).final_score
            try:
                scores.append(float(model_answer) / 100)
            except:
                scores.append(0.5)
        return scores
