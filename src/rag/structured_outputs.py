from pydantic import BaseModel, Field


class RAGAnswerTp(BaseModel):
    """Ответ рага"""
    judgement: str = Field(description="Рассуждения о том, как из полученных отрывков получить ответ на вопрос.")
    answer: str = Field(description="Ответ на вопрос")

class AdaptiveTp(BaseModel):
    """Оценка полученных документов"""
    judgement: str = Field(description="Рассуждения о том, можно ли  из полученных отрывков получить ответ на вопрос.")
    answer: bool = Field(description="True, если информации в отрывках хватает для ответа на вопрос, False - иначе ")

class FilterTP(BaseModel):
    """Оценка релевантности контекста"""
    judgement: str = Field(description="Рассуждения о том, содержится ли в отрывке.")
    answer: bool = Field(description="True, если информации в отрывках хватает для ответа на вопрос, False - иначе ")

