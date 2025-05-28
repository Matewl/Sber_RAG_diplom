from typing import Dict, Callable
from langchain_core.runnables import Runnable


class QA:
    def __init__(self, chain: Runnable, process_func: Callable):
        self.chain: Runnable = chain
        self.process_func = process_func
    
    def retrieve_answer(self, **kwargs):
        input =self.process_func(**kwargs)
        return self.chain.invoke(input)

