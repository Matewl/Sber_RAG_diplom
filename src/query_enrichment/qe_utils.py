from dataclasses import dataclass, field
from typing import Union, Dict, List, Optional

@dataclass
class EnrichedQuery:
    question: str
    hypothetical_doc: str = ''
    rephrased_queries: list = field(default_factory=list)
    hypo_answers: list = field(default_factory=list)
    stepback: str = ''


def _default_user_message(question: str):
    return {"question": question}

type_to_process_map = {
    'hyde': _default_user_message,
    'answer': _default_user_message,
    'rephrase': _default_user_message,
    'step_back': _default_user_message,
}