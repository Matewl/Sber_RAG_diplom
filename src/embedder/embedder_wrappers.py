from src.embedder.utils import prepare_emb, prepare_embs

class Embedder_wrapper:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        embs = self.model.encode(texts)
        return prepare_embs(embs)

    def embed_docs_pc(self, docs):
        texts = [doc.page_content for doc in docs]
        return self.embed_documents(texts)

    def embed_query(self, query):
        return prepare_emb(self.model.encode(query))


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


class Embedder_wrapper_e5_instruct:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        embs = self.model.encode(texts)
        return prepare_embs(embs)

    def embed_docs_pc(self, docs):
        texts = [doc.page_content for doc in docs]
        return self.embed_documents(texts)

    def embed_query(self, query):
        task = 'Given a web search query, retrieve relevant passages that answer the query'
        query = get_detailed_instruct(task, query)

        return prepare_emb(self.model.encode(query))

    def __call__(self, query):
        return self.embed_query(query)

class Embedder_wrapper_e5_giga:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        embs = self.model.embed_documents(texts)
        return prepare_embs(embs)

    def embed_docs_pc(self, docs):
        texts = [doc.page_content for doc in docs]
        return self.embed_documents(texts)

    def embed_query(self, query):
        task = 'Given a search query, retrieve relevant passages that answer the query'
        query = get_detailed_instruct(task, query)

        return prepare_emb(self.model.embed_query(query))

    def __call__(self, query):
        return self.embed_query(query)

# class Embedder_wrapper_nomic:
#     def __init__(self, model):
#         self.model = model

#     def embed_documents(self, texts):
#         prompt = 'search_document: '
#         return [self.model.encode(prompt + text) for text in texts]

#     def embed_docs_pc(self, docs):
#         prompt = 'search_document: '
#         return [self.model.encode(prompt + doc.page_content) for doc in docs]

#     def embed_query(self, query):
#         prompt = 'search_query: '
#         return self.model.encode(prompt + query)