from pydantic import BaseModel, Field


class RephrasedQuestion(BaseModel):
    """Преобразованный вопрос."""
    question: str = Field(description="Вопрос, преобразованный к виду запроса к базе данных.")

class RephrasedQuestionsList(BaseModel):
    """Список из 3 преобразованных вопросов."""
    questions: list[RephrasedQuestion] = Field(description="3 версии вопроса, преобразованного к виду запроса к базе данных.")


class AnswerQuestion(BaseModel):
    """Ответ на вопрос."""
    answer: str = Field(description="Гипотетический ответ на вопрос.")

class AnswerQuestionsList(BaseModel):
    """Список из 3 ответов на вопрос."""
    answers: list[AnswerQuestion] = Field(description="3 разнообразных ответа на вопрос.")


