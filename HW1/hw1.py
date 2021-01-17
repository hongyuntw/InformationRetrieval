
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np


docs_id = []
queries = []
queries_id = []


doc_tfs = []
tf_idfs = []
query_vocab = set()
vocab = set()

def load_file():
    base_dir = './ntust-ir-2020/'
    doc_path = base_dir + 'docs/'
    query_path = base_dir + 'queries/'

    for file in os.listdir(query_path):
        file_path = query_path + file
        fp = open(file_path, 'r')
        query = fp.readline()
        words = query.split()
        for word in words:
            query_vocab.add(word)
        queries.append(words)
        queries_id.append(file.replace('.txt',''))
        fp.close()

    for file in os.listdir(doc_path):
        file_path = doc_path + file
        fp = open(file_path, 'r')
        doc = fp.readline()
        words = doc.split()
        doc_tf = {}
        # for tf
        for word in words:
            if word in query_vocab:
                vocab.add(word)
            if word in vocab:    
                if word in doc_tf:
                    doc_tf[word] += 1
                else:
                    doc_tf[word] = 1
        
        doc_tfs.append(doc_tf)
        docs_id.append(file.replace('.txt',''))
        fp.close()


    print(len(doc_tfs))
    print(len(queries))
    print(len(query_vocab))
    print(len(vocab))

def query2array(query):
    query_arr = []
    for q in query:
        q_arr = []
        for word in query_vocab:
            if word == q:
                q_arr.append(1)
            else:
                q_arr.append(0)
        query_arr.append(q_arr)
    query_arr = np.array(query_arr)
    return query_arr

def getRelevance(query, doc_tfidf):
    relevance = np.zeros(4191)
    for q in query:
        idx = 0
        for word in query_vocab:
            if word == q:
                relevance = np.add(relevance,doc_tfidf[:, idx])
            idx += 1
    # relevance /= len(query)
    return relevance

def tfidf():
    
    # calcuate idfs
    doc_idfs = {}
    for word in query_vocab:
        # avoid division by zero
        dfi = 1
        for doc_tf in doc_idfs:
            if word in doc_tf:
                dfi += 1
        idf = np.log10(1 + (4191 / dfi))
        doc_idfs[word] = idf

    # compute tfidf
    doc_tfidf = []
    for doc_tf in doc_tfs:
        tfidf = []
        for word in query_vocab:
            # calculate tf
            tf = 0
            if word in doc_tf:
                tf = doc_tf[word]
            if tf > 0 :
                tf = 1 + np.log10(tf)
            idf = doc_idfs[word]
            # tfidf.append(tf * idf)
            tfidf.append(tf)
        # normalize
        if max(tfidf) - min(tfidf) != 0:
            tfidf = tfidf / np.linalg.norm(tfidf)
        doc_tfidf.append(tfidf)

    doc_tfidf = np.array(doc_tfidf)
    # normalize
    # doc_tfidf = doc_tfidf / np.linalg.norm(doc_tfidf)
    print(doc_tfidf.shape)


    fp = open('result_norm.txt', 'w')
    print('Query,RetrievedDocuments', file=fp)
    for query_id, query in zip(queries_id, queries):
        print(query_id + ',', file=fp, end='')
        query_array = query2array(query)
        relevance = cosine_similarity(query_array, doc_tfidf)
        relevance = relevance.sum(axis=0)
        relevance = relevance / len(query)

        # relevance = getRelevance(query,doc_tfidf)
        doc_dict = dict(zip(docs_id, relevance))
        sorted_docs = sorted(doc_dict.items(), key=lambda x: x[1], reverse=True)
        for _doc in sorted_docs:
            doc_id, value = _doc
            print(doc_id , file=fp , end=' ')
        print("",file=fp)
    fp.close()
    return

if __name__ == "__main__":
    load_file()
    tfidf()

    