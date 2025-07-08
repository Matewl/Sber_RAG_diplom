from pydantic import BaseModel, Field


class RephrasedQuestion(BaseModel):
    """NDA"""
    question: str = Field(description="""NDA""")

class RephrasedQuestionsList(BaseModel):
    """NDA"""
    questions: list[RephrasedQuestion] = Field(description="""NDA""")

class AnswerQuestion(BaseModel):
    """NDA"""
    answer: str = Field(description="""NDA""")

class AnswerQuestionsList(BaseModel):
    """NDA"""
    answers: list[AnswerQuestion] = Field(description="""NDA""")


