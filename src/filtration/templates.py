from pydantic import BaseModel, Field


class FilteredContext(BaseModel):
    """NDA"""
    throughts: str = Field(description="""NDA""") 
    verdict: int = Field(description="""NDA""") 

