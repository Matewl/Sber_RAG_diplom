from typing import Dict, Union
from tokenizers import Tokenizer

from pathlib import Path
current_dir = Path(__file__).parent


class EmbedderTokenizer:
    def __init__(self):
        self.max_tokens = 512
        self.tokenizer = Tokenizer.from_file(str(current_dir / 'e5-large' / "tokenizer.json"))
    
    def cut_string_by_max_tokens(self, string):
        token_ids = self.tokenizer.encode(string).ids
        return self.tokenizer.decode(token_ids[:self.max_tokens - 2])
    