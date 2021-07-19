import numpy as np
import pandas as pd
import nltk.data
import pickle
from rank_bm25 import BM25L

def rank(query):
    PM_Articles = pd.read_csv("data/PM classified data/Final_PM_Reduced.csv", encoding='latin1')


    corpus = []

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    Title = PM_Articles['Title']
    #Abstract = PM_Articles['Abstract']
    NCT_ID = PM_Articles['NCT ID']

    gene = "BRAF (V600E)"


    for article in PM_Articles.Abstract.apply(lambda row: row.lower()):
        corpus.extend(tokenizer.tokenize(article))



    embedding_file = "data/models/PM_Articles_DistilledBert_reduced.emb"
    with open(embedding_file, mode='rb') as emb_f:
        corpus_embeddings = pickle.load(emb_f)

    bm25 = BM25L(corpus)
    tokenized_gene = gene.split(" ")
    BM25_Score = bm25.get_scores(tokenized_gene) * 2
    query_embeddings = bm25.get_scores(query)
    query_embeddings=np.array([elemnts*query_embeddings for elemnts in np.ones(768, dtype = int)]).T
    topk=10
    score_corpus = np.sum(query_embeddings * corpus_embeddings, axis=1) / np.linalg.norm(corpus_embeddings, axis=1)
    results=[]
    score_list=[]
    Title1=[]
    topk_idx = np.argsort(score_corpus)[::-1][:topk]

    i = 0
    for idx in topk_idx:
        i = i + 1
        score = score_corpus[idx] + BM25_Score[idx]
        results.append('https://clinicaltrials.gov/ct2/show/' + NCT_ID[idx] + '?term=' + NCT_ID[idx] + '&draw=2&rank=1 ')
        score_list.append(score)
        Title1.append(Title[idx])
    return results, score_list, query, Title1
