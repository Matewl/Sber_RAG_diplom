import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_gigachat import  GigaChatEmbeddings, GigaChat
from langchain.schema import Document
from datasets import Dataset
import pickle
import ragas
from ragas import evaluate
from ragas.metrics import (faithfulness, 
                           context_precision, 
                           context_recall, 
                           answer_correctness, 
                           answer_relevancy, 
                           noise_sensitivity,
                           faithfulness_ru, 
                           answer_correctness_ru, 
                           answer_correctness_devide_ru, 
                           answer_relevancy_ru)
from ragas.run_config import RunConfig

cert_file='C:\\Users/22589910/Desktop/work/certs/our_certs/chain_ift_pem.pem'  
key_file='C:\\Users/22589910/Desktop/work/certs/our_certs/CI02851867-IFT-RISKASSISTANT2.key'  

embedder = GigaChatEmbeddings(
            base_url='https://gigachat-ift.sberdevices.delta.sbrf.ru/v1',
            model= 'Embeddings',
            cert_file=cert_file,
            key_file=key_file,
            verify_ssl_certs=False,

)
llm = GigaChat(
            base_url='https://gigachat-ift.sberdevices.delta.sbrf.ru/v1',
            model = 'GigaChat-2-MAX',
            cert_file=cert_file,
            key_file=key_file, 
            verify_ssl_certs=False,
            temperature=0.0,
            max_tokens=32000,
            profanity_check=False,
)



bench_df = pd.read_excel('after_adding_doc_name_.xlsx')
bench_df = bench_df[bench_df['docs_bucket'] == 'group standards']
queries = list(bench_df['Вопрос'])
answers = list(bench_df['Ответ'])

def create_dataset(res):
    user_input = []
    retrieved_contexts = []
    response = []
    reference = []
    for u, r, ref in zip(queries, res, answers):
        if r:
            user_input.append(u)
            retrieved_contexts.append(r[1])
            response.append(r[0])
            reference.append(ref)
    df = pd.DataFrame({
        'user_input': user_input,
        'retrieved_contexts':retrieved_contexts ,
        'response':response,
        'reference': reference
    })
    return Dataset.from_pandas(df)



config = RunConfig(max_workers=8, 
                   max_retries=1, 
                   max_wait=300, 
                   timeout=400)

def evaluate_all_metrics(dataset, llm, config):
    metrics = evaluate(dataset=dataset, 
         metrics=[
                # faithfulness,
                # context_precision,
                # context_recall,
                answer_relevancy_ru,
                # answer_correctness,
                # answer_correctness_ru,
                # faithfulness_ru,
                answer_correctness_devide_ru,
                # noise_sensitivity,
                ],
        llm=llm, 
        embeddings=embedder, 
        run_config=config
    )
    return metrics.to_pandas()

def evaluate_RAG(res_path):
    res = pickle.load(open(res_path, 'rb'))
    dataset = create_dataset(res)
    metrics = evaluate_all_metrics(dataset, llm, config)
    metrics.to_csv(res_path.split('.')[0] + '.csv')




