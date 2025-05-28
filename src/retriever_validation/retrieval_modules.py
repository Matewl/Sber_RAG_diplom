import numpy as np
from rank_bm25 import BM25Okapi
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from nltk import WordPunctTokenizer
from nltk.stem.snowball import SnowballStemmer
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_gigachat import GigaChat

from pydantic import BaseModel, Field, confloat
from retriever.retriever import Retriever






with open('russian.txt') as f:
    stop_words = f.readlines()

stop_words = set([word[:-1] for word in stop_words])

stemmer = SnowballStemmer('russian')
tokenizer = WordPunctTokenizer()


class Embedder_wrapper:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return [self.model.encode(text) for text in texts]

    def embed_docs_pc(self, docs):
        return [self.model.encode(doc.page_content) for doc in docs]

    def embed_query(self, query):
        return self.model.encode(query)


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


class Embedder_wrapper_e5_instruct:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return [self.model.encode(text) for text in texts]

    def embed_docs_pc(self, docs):
        return [self.model.encode(doc.page_content) for doc in docs]

    def embed_query(self, query):
        task = 'Given a web search query, retrieve relevant passages that answer the query'
        query = get_detailed_instruct(task, query)

        return self.model.encode(query)

    def __call__(self, query):
        return self.embed_query(query)


class Embedder_wrapper_nomic:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        prompt = 'search_document: '
        return [self.model.encode(prompt + text) for text in texts]

    def embed_docs_pc(self, docs):
        prompt = 'search_document: '
        return [self.model.encode(prompt + doc.page_content) for doc in docs]

    def embed_query(self, query):
        prompt = 'search_query: '
        return self.model.encode(prompt + query)


# def fusion_retrieval_block(db, query, strategy='ss', alpha=0.9, top_k=10):

#     # обычный поиск в базе данных
#     if strategy == 'ss':
#         db_retrieved_scores = db.similarity_search_with_relevance_scores(query, db.index.ntotal)
#     if strategy == 'mmr':
#         db_retrieved_scores = db.max_marginal_search_with_relevance_scores(query, db.index.ntotal)

#     vector_scores = np.array([score for _, score in db_retrieved_scores])
#     vector_scores = (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores))

#     # поиск с помощью БМ-25
#     clean_documents = [clean_text(doc[0].page_content) for doc in db_retrieved_scores]
#     BMdb = BM25Okapi(clean_documents)
#     bm25_scores = BMdb.get_scores(clean_text(query))

#     if np.max(bm25_scores) - np.min(bm25_scores) == 0:
#         bm25_scores = 0
#     else:
#         bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))

#     combined_scores = alpha * vector_scores + (1 - alpha) * bm25_scores 
#     sorted_indices = np.argsort(combined_scores)[::-1]

#     return [db_retrieved_scores[i][0] for i in sorted_indices[:top_k]], \
#         [db_retrieved_scores[i][1] for i in sorted_indices[:top_k]]








class EnrichAsAnswerRetriever(Retriever):
    def __init__(self, db, reranker=None, strategy='mmr', fusion_alpha=1, k = 10, rerank_k = None, has_answer_th = 0., llm=None):
        super().__init__(db, reranker, strategy, fusion_alpha, k, rerank_k, has_answer_th)
        self.llm = llm

    def get_advance_query(self, query):

        PROMPT = \
"""Ты - эксперт по методологии кредитного процесса Сбербанка. Ответ должен содержать 2-3 предложения. 
Тебе будет дан вопрос, связанный с банковской сферой.
Приведи пример ответа на данный вопрос. Ответ должен содержаться во внутренних нормативных документах Банка России

<few_shot_questions>
Вопрос: Расскажи про учет изменений в иностранном законодательстве при управлении оп риском в зарубежных дочерних кредитных организациях?\n\
Ответ: Кредитная организация должна учитывать требования национального законодательства иностранного государства при управлении операционным риском в дочерних организациях, включая порог регистрации событий. При этом показатели операционного риска приводятся в соответствие с требованиями национального законодательства, если оно противоречит требованиям Положения.\n\
Вопрос: Как кредитная организация должна учитывать потери от реализации событий операционного риска при расчете капитала, и какие требования предъявляются к ведению базы событий в этом контексте?\n\
Ответ: Кредитная организация должна ежемесячно определять величину валовых потерь от реализации событий операционного риска и использовать эти данные при выборе подхода к расчету объема капитала на покрытие таких потерь, выбирая между регуляторным и продвинутым подходами.\n\
</few_shot_questons>
"""

        template = ChatPromptTemplate.from_messages([
            ('system', PROMPT),
            ('user', 'Вопрос:\n{question}')
        ])

        chain = (
            template | 
            self.llm | 
            StrOutputParser()
        )

        model_answer = chain.invoke({'question': query})
        return query + '\n' + model_answer


class EnrichAsQueryRetriever(Retriever):
    def __init__(self, db, reranker=None, strategy='mmr', fusion_alpha=1, k = 10, rerank_k = None, has_answer_th = 0., llm=None, query_count=3):
        super().__init__(db, reranker, strategy, fusion_alpha, k, rerank_k, has_answer_th)
        self.llm = llm
        self.query_count = query_count

    def get_advance_query(self, query):
        PROMPT = \
"""Ты - эксперт по методологии кредитного процесса Сбербанка.
Тебе будет дан вопрос, связанный с банковской сферой.
Переформулируй вопрос {n_query} способами таким образом, чтобы ответ на них можно будет найти во внутренних нормативных документах Банка России.
Ответ должен содержать ровно {n_query} вопросов. Старайся, чтобы вопросы немного отличались друг от друга."""


        template = ChatPromptTemplate.from_messages([
            ('system', PROMPT),
            ('user', 'Вопрос:\n{question}')
        ])

        chain = (
            template | 
            self.llm | 
            StrOutputParser()
        )

        model_answer = chain.invoke({'question': query})
        return query + '\n' + model_answer


class EnrichAsCorrectionRetriever(Retriever):
    def __init__(self, db, reranker=None, strategy='mmr', fusion_alpha=1, k = 10, rerank_k = None, has_answer_th = 0., llm=None, query_count=3):
        super().__init__(db, reranker, strategy, fusion_alpha, k, rerank_k, has_answer_th)
        self.llm = llm
        self.query_count = query_count

    def get_advance_query(self, query):
        PROMPT = \
"""Ты - эксперт по методологии кредитного процесса Сбербанка.
Тебе будет дан вопрос, связанный с банковской сферой.
Переформулируй вопрос, таким образом, чтобы он стал более детальным и конкретным. Вопрос должен стать понятнее для человека.
Все сущности вопроса должны сохраниться. Все аббревиатуры в вопросе должны сохраниться."""


        template = ChatPromptTemplate.from_messages([
            ('system', PROMPT),
            ('user', 'Вопрос:\n{question}')
        ])

        chain = (
            template | 
            self.llm | 
            StrOutputParser()
        )

        model_answer = chain.invoke({'question': query})
        return query + '\n' + model_answer

class RAGAnswerTp(BaseModel):
    """Ответ рага"""
    judgement: str = Field(description="Рассуждения о том, как из полученных отрывков получить ответ на вопрос.")
    answer: str = Field(description="Ответ на вопрос")

class RAG:
    def __init__(self, llm, retriever):
        self.llm = llm.with_structured_output(RAGAnswerTp)
        self.retriever = retriever
    
    def get_context(self, query):
        return '\n'.join([doc.page_content for doc in self.retriever.retrieve(query)])
    
    def generate_answer(self, query, context):
        STUFF_SYSTEM = """Ты - эксперт по методологии кредитного процесса Сбербанка. Тебе дан вопрос и релевантные отрывки текста из разных документов.
Создай информативный (не более 200 слов) и максимально точный ответ на заданный вопрос,
основываясь исключительно на приведенные отрывки. Ты должен использовать только информацию из приведенных отрывков.
Используй непредвзятый и журналистский тон. Не повторяй текст.
Не пытайся придумать ответ.
Отвечай только на русском языке за исключением специальных терминов.
Если информации из отрывков недостаточно для точного ответа на вопрос, отвечай: 'Пожалуйста, обратитесь на горячую линию или сформулируйте вопрос иначе'.
"""
        chat_template = ChatPromptTemplate.from_messages([
            ('system', STUFF_SYSTEM),
            ('user', 'Отрывки из документов:\n{context}\nВопрос:\n{question}')
        ])

        chain = (
            chat_template | 
            self.llm 
                    )
        answer = chain.invoke({'context': context, 'question': query})
        return answer.answer
    
    def get_answer(self, query):
        context = self.get_context(query)
        return self.generate_answer(query, context)
    
    def __call__(self, query):
        return self.get_answer(query)
    
class AdaptiveTp(BaseModel):
    """Оценка полученных документов"""
    judgement: str = Field(description="Рассуждения о том, можно ли  из полученных отрывков получить ответ на вопрос.")
    answer: bool = Field(description="True, если информации в отрывках хватает для ответа на вопрос, False - иначе ")

class AdaptiveRAG(RAG):
    def __init__(self, llm, retriever, n_loops=5, n_retries=10, use_rephrase=False):
        self.llm = llm.with_structured_output(RAGAnswerTp) 
        self.is_comlete_llm = llm.with_structured_output(AdaptiveTp)
        self.rephrase_llm = llm
        self.retriever = retriever
        self.n_loops = n_loops
        self.n_retries = n_retries
        self.use_rephrase = use_rephrase

    def get_answer(self, query):
        IS_COMPLETE_PROMPT = """Ты - эксперт по методологии кредитного процесса Сбербанка. Тебе дан вопрос и релевантные отрывки текста из нескольких документов.
Определи, достаточно ли информации в отрывках для ответа на поставленный вопрос.
Верни True только в том случае, если информации достаточно для полного ответа на вопрос.
Если информации достаточно только для частичного ответа на вопрос, верни False". 
"""
        is_complete_template = ChatPromptTemplate.from_messages([
            ('system', IS_COMPLETE_PROMPT),
            ('user', 'Отрывки из документов:\n{context}\nВопрос:\n{question}')
        ])

        is_complete_chain = (
            is_complete_template | 
            self.is_comlete_llm         
        )

        REPHRASE_PROMPT = """Ты - эксперт по методологии кредитного процесса Сбербанка. 
Тебе дан вопрос и ответ эксперта на него. Известно, что данный ответ является неполным. 
Прежде, чем дать ответ, эксперт на основе вопроса ищет отрывки документов в базе данных, по которым можно дать ответ.
Перефразируй или уточни вопрос таким образом, чтобы по нему легче было найти релевантные отрывки и дать правильный ответ. 
Смысл измененного вопроса должен совпадать со смыслом исходного вопроса.
"""
        rephrase_template = ChatPromptTemplate.from_messages([
            ('system', REPHRASE_PROMPT),
            ('user', 'Вопрос:\n{question}\nОтвет эксперта:\n{answer}')
        ])

        rephrase_chain = (
            rephrase_template | 
            self.rephrase_llm ,
            StrOutputParser()        )

        documents_history = []
        answers_history = []
        queries_history = [query]
        for loop in range(self.n_loops):
            for retry in range(self.n_retries):
                # try:
                    current_context = [document for document in documents_history] + answers_history
                    retrieved_documents = self.retriever.retrieve('\n'.join(queries_history + answers_history))

                    retrieved_documents = [document.page_content for document in retrieved_documents if document not in current_context]

                    llm_answer = self.generate_answer(query, '\n'.join(retrieved_documents + current_context))
                    answers_history.append(llm_answer)

                    is_complete = is_complete_chain.invoke({'context': retrieved_documents + current_context, 'question': query}).answer
                    if is_complete:
                    # if is_complete == 'Да' or is_complete == 'да':
                        return llm_answer
                    
                    documents_history.extend(retrieved_documents)
                    answers_history.append(llm_answer)
                    if self.use_rephrase:
                        new_query = rephrase_chain.invoke({'question': query, 'answer': llm_answer})
                        queries_history.append(new_query)
                    break
                
                # except:
                    continue
        try:
            return llm_answer
        except:
            return None
        
class InformationExtractor:
    def __init__(self, llm):
        self.llm = llm
    
    def summarize(self, query, paragraphs):
        SUMMARIZE_PROMPT = """Ты - эксперт по методологии кредитного процесса Сбербанка. Тебе дан вопрос и релевантные отрывки текста из нескольких документов.
Тебе нужно суммаризовать информацию из приведенных отрывков по следующим правилам:
1. Суммаризация должна опираться только на данные отрывки.
2. Суммаризация не должны содержать информацию, которая не представлена в отрывках.
3. В суммаризацию должна входить только та информация, которая может быть полезна при ответе на заданный вопрос.
4. В суммаризации должны получиться только целостные и точные факты из приведенных отрывков.
"""
        summarize_template = ChatPromptTemplate.from_messages([
            ('system', SUMMARIZE_PROMPT),
            ('user', 'nВопрос:\n{question}\nОтрывки из документов:\n{context}')
        ])

        summarize_chain = (
            summarize_template | 
            self.llm | 
            StrOutputParser()
        )
        summarization = summarize_chain.invoke({'context': paragraphs, 'question': query})

        return summarization

class QueryEnrichment:
    def __init__(self, llm):
        self.llm = llm
    def enrich_query(self, query):
        return ''

class HyDE(QueryEnrichment):
    def enrich_query(self, query):
        HYDE_PROMPT = """Ты - эксперт по методологии кредитного процесса Сбербанка.
Тебе дан вопрос. Сгененируй отрывок документа по следующим правилам:
1. В отрывке документа хранится вся нужная информация для ответа на вопрос.
2. Текст в отрывке должен быть точным, информативным и хорошо структурированным.
3. Избегай общих фраз, делай текст конкретным.
4. Опимальный объем: не более 10 предложений.
"""
        hyde_template = ChatPromptTemplate.from_messages([
            ('system', HYDE_PROMPT),
            ('user', 'nВопрос:\n{question}\)')
        ])

        hyde_chain = (
            hyde_template | 
            self.llm | 
            StrOutputParser()
        )
        hypothetical_doc = hyde_chain.invoke({'question': query})

        return hypothetical_doc

class SQuARE:
    def __init__(self, llm, n_questions=3):
        self.llm = llm
        self.n_questions = n_questions
    def generate_qa_pairs(self, query, contexts):
        SQUARE_PROMPT = f"""Ты - эксперт по методологии кредитного процесса Сбербанка.
Тебе дан вопрос и релевантные отрывки текста из нескольких документов.
Сгенерируй {self.n_questions} вопросов по заданным отрывкам и ответы к ним. Ответы на вопросы должны присутствовать в приведенных отрывках.
После генерации ответов дай ответ на исходный вопрос, опираясь на исходный контекст и информацию, которую ты узнал, отвечая на вопросы.
"""
        square_template = ChatPromptTemplate.from_messages([
            ('system', SQUARE_PROMPT),
            ('user', 'Вопрос:\n{question}\nОтрывки из документов:\n{context}')
        ])

        square_chain = (
            square_template | 
            self.llm | 
            StrOutputParser()
        )
        square = square_chain.invoke({'question': query, 'context': '\n'.join(contexts)})

        return square

class CoTFilter:
    def __init__(self, llm):
        self.llm = llm

    def generate_qa_pairs(self, query, context):
        FILTER_PROMPT = f"""Ты - эксперт по методологии кредитного процесса Сбербанка.
Тебе дан вопрос и релевантный отрывок документа.
Твоя задача определить, есть ли в этом отрывке информация, которая может быть полезна для ответа на вопрос.
Отвечай одним словом, только \"Да\" или \"Нет\".
Ответь \"Да\", если в отрывке есть информация, которая может помочь ответить на вопрос. Ответ \"Нет\" иначе.
"""
        filter_template = ChatPromptTemplate.from_messages([
            ('system', FILTER_PROMPT),
            ('user', 'Вопрос:\n{question}\nОтрывки из документов:\n{context}')
        ])

        filter_chain = (
            filter_template | 
            self.llm | 
            StrOutputParser()
        )
        filter_verdict = filter_chain.invoke({'question': query, 'context': '\n'.join(context)})

        return filter_verdict

class ExtractFilter:
    def __init__(self, llm):
        self.llm = llm

    def generate_qa_pairs(self, query, context):
        FILTER_PROMPT = f"""Ты - эксперт по методологии кредитного процесса Сбербанка.
Тебе дан вопрос и релевантный отрывок документа.
Твоя задача определить, какие части отрывка могут содержать информацию, полезную для ответа на вопрос.
Ответ должен содержать только эти части. Цитируй исходный отрывок дословно. Если для ответа на вопрос нужен целый отрывок, процитируй его польностью. 

"""
        filter_template = ChatPromptTemplate.from_messages([
            ('system', FILTER_PROMPT),
            ('user', 'Вопрос:\n{question}\nОтрывки из документов:\n{context}')
        ])

        filter_chain = (
            filter_template | 
            self.llm | 
            StrOutputParser()
        )
        filter_verdict = filter_chain.invoke({'question': query, 'context': '\n'.join(context)})

        return filter_verdict
