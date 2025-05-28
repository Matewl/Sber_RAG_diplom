
from nltk import WordPunctTokenizer
from nltk.stem.snowball import SnowballStemmer

with open('src\\retriever\\BM_25\\russian.txt') as f:
    stop_words = f.readlines()

stop_words = set([word[:-1] for word in stop_words])

stemmer = SnowballStemmer('russian')
tokenizer = WordPunctTokenizer()


def clean_text(text):
    text = stemmer.stem(text)
    text = tokenizer.tokenize(text)
    text = [token.lower() for token in text if token.isalpha() and token not in stop_words]

    return text