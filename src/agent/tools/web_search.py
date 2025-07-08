from typing import List
from langchain_gigachat.tools.giga_tool import giga_tool
from pydantic import BaseModel, Field

class WebSearchResult(BaseModel):
    pages: List[str] = Field(description="NDA")

few_shot_examples = [
    {
        "NDA": "NDA"
    }
]

@giga_tool(few_shot_examples=few_shot_examples)
def web_search_tool(question: str = Field(description="NDA")) -> WebSearchResult:
    """NDA"""
    return WebSearchResult(pages=[''])


