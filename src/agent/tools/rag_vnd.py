from typing import List
from langchain_gigachat.tools.giga_tool import giga_tool
from pydantic import BaseModel, Field

from src.pipeline import searcher

few_shot_examples = [
    {
        "NDA": "NDA"
    }
]

@giga_tool(few_shot_examples=few_shot_examples)
def vnd_tool(question: str = Field(description="NDA")) -> str:
    """NDA"""
    return searcher.search(question)


