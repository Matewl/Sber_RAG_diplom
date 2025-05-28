from scipy.spatial.distance import cosine
import numpy as np

def metric_docs_docs(retrieved_docs, true_docs, embedder):
    retrieved_embs = embedder.embed_documents(retrieved_docs)
    true_embs = embedder.embed_documents(true_docs)

    res_1 = 0
    for true_emb in true_embs:
        min_dist = np.inf
        for doc_emb in retrieved_embs:
            min_dist = min(min_dist, cosine(true_emb, doc_emb))
        res_1 += min_dist

    res_2 = 0
    for doc_emb in retrieved_embs:
        max_dist = -np.inf
        for true_emb in true_embs:
            max_dist = max(max_dist, cosine(true_emb, doc_emb))
        res_2 += max_dist


    return 1 - res_1 / len(retrieved_docs), 1 - res_2 / len(retrieved_docs)


def metric_docs_answer(retrieved_docs, answer, embedder):
    res = 0
    retrieved_embs = embedder.embed_documents(retrieved_docs)
    answer_emb = embedder.embed_query(answer)
    min_dist = np.inf
    max_dist = -np.inf

    for doc_emb in retrieved_embs:
        dist = cosine(answer_emb, doc_emb)
        min_dist = min(min_dist, dist)
        max_dist = max(max_dist, dist)
        res += dist

    return 1 - res / len(retrieved_docs), 1-min_dist, 1-max_dist


def NotIsNaN(obj):
    return obj == obj


def count_metrics(retriever, queries, embedder):
    """
    Считаем метрики на основе полученных документов (которые отдал ретривер) и запроса. 
    
        doc_doc_metric - насколько полученные документы похожи на документы из gold ответа
        doc_answer_metric - насколько полученные документы похожи на gold ответ
        confident - насколько ретривер уверен в том, что полученные документы отвечают на вопрос
    """


    doc_doc_metrics_prec = []
    doc_doc_metrics_rec = []

    doc_answer_metrics_mean = []
    doc_answer_metrics_max = []
    doc_answer_metrics_min = []

    confident_metrics = []

    retrieved_docs_all = []
    for row in queries.iterrows():
        row = row[1].loc()
        query = row['query']
        answer = row['gold_answer']
        answer_docs = []

        if NotIsNaN(row['gold_passage1']) and row['gold_passage1'] != 0: answer_docs.append(row['gold_passage1'])
        if NotIsNaN(row['gold_passage2']) and row['gold_passage2'] != 0: answer_docs.append(row['gold_passage2'])
        if NotIsNaN(row['gold_passage3']) and row['gold_passage3'] != 0: answer_docs.append(row['gold_passage3'])

        retrieved_docs = retriever.retrieve(query)
        retrieved_docs = ['\n'.join(doc.page_content.split('\nГлава: ')[1:]) for doc in retrieved_docs]
        retrieved_docs_all.append(retrieved_docs)
        doc_doc_res = metric_docs_docs(retrieved_docs, answer_docs, embedder)

        doc_doc_metrics_prec.append(doc_doc_res[0])
        doc_doc_metrics_rec.append(doc_doc_res[1])

        doc_ans_res = metric_docs_answer(retrieved_docs, answer, embedder)

        doc_answer_metrics_mean.append(doc_ans_res[0])
        doc_answer_metrics_min.append(doc_ans_res[1])
        doc_answer_metrics_max.append(doc_ans_res[2])

        confident_metrics.append(1 - cosine(embedder.embed_query(retrieved_docs[0]), embedder.embed_query(query)))

    res = {
        'doc_doc_metric_prec': doc_doc_metrics_prec,
        'doc_doc_metric_rec': doc_doc_metrics_rec,

        'doc_answer_metric_mean': doc_answer_metrics_mean,
        'doc_answer_metric_min': doc_answer_metrics_min,
        'doc_answer_metric_max': doc_answer_metrics_max,

        'confident_metric': confident_metrics,
    }

    return res, retrieved_docs_all

def get_norm_results(results):
    """
    Нормализуем метрики
    """
    result_items = list(results.items())
    all_results = {key : [] for key in result_items[0][1]}
    norm_results = list({key : results[key].copy() for key in results}.items())
    for result in result_items:
        for key in result[1]:
            all_results[key].append(result[1][key])

    for result in norm_results:
        for key in result[1]:
            result[1][key] -= np.mean(all_results[key])
            result[1][key] /= np.std(all_results[key])

    return norm_results

def get_best_result(results):
    """
    Выбираем лучший результат:
        Первый критерий - средняя метрика
        При равенстве первого критерия сравниваются отдельные метрики в таком порядке:
            doc_doc_metric_rec
            doc_answer_metric_mean
            confident_metric
            doc_answer_metric_min
            doc_answer_metric_max
            doc_doc_metric_prec
    """
    norm_results = get_norm_results(results)
    best_value = -np.inf
    best_key = None

    for result in norm_results:
        sum_result = sum([val for val in result[1].values()])
        if best_value < sum_result:
            best_value = sum_result
            best_key = result[0]
        elif best_value == sum_result: # tie break
            best_res = norm_results[best_key]
            crits = ['doc_doc_metric_rec',
                     'doc_answer_metric_mean',
                     'confident_metric',
                     'doc_answer_metric_min',
                     'doc_answer_metric_max',
                     'doc_doc_metric_prec']
            for crit in crits:
                if best_res[crit] < result[1][crit]:
                    best_value = sum_result
                    best_key = result[0]
                    break
                if best_res[crit] > result[1][crit]:
                    break
    return {best_key : results[best_key]}