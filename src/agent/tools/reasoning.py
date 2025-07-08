from typing import List
from langchain_gigachat.tools.giga_tool import giga_tool
from pydantic import BaseModel, Field

@giga_tool()
def reasoning_tool(thoughts: str = Field(description="NDA")) -> str:
    """NDA"""
    return thoughts


