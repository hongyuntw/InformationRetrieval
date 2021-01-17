import os
import numpy as np
import math
docs_id = []
queries = []
queries_id = []

doc_tfs = []
tf_idfs = []
doc_lens = []
query_vocab = set()
vocab = set()




def load_file():
    base_dir = './HW2/ntust-ir-2020/'
    doc_path = base_dir + 'docs/'
    query_path = base_dir + 'queries/'

    total_len = 0
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
        total_len += len(words)
        doc_lens.append(len(words))
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

    avg_doc_len = total_len / 4091
    print(len(doc_tfs))
    print(len(queries))
    print(len(query_vocab))
    print(len(vocab))
    print(avg_doc_len)
    return avg_doc_len


def BM(avg_doc_len):
    # Discriminative power
    doc_idfs = {}
    for word in query_vocab:
        # avoid division by zero
        ni = 0
        for doc_tf in doc_tfs:
            if word in doc_tf:
                ni += 1
        idf = math.log((4091 - ni + 0.5) / (ni + 0.5))
        doc_idfs[word] = idf


    # compute tfidf
    doc_tfidf = []
    idx = 0
    for doc_tf in doc_tfs:
        tfidf = []
        doc_len_normalize = doc_lens[idx] / avg_doc_len
        idx += 1
        for word in query_vocab:
            # calculate tf
            tf = 0
            if word in doc_tf:
                tf = doc_tf[word]
            if tf > 0:
                tf = tf / (1 - b + b * doc_len_normalize)
                tf = ((K1 + 1) * (tf + delta)) / (K1 + tf + delta)
                
            idf = doc_idfs[word]
            tfidf.append(tf * idf * (idf))
        doc_tfidf.append(tfidf)
    doc_tfidf = np.array(doc_tfidf)
    print(doc_tfidf.shape)

    fp = open('./HW2/result.txt', 'w')
    print('Query,RetrievedDocuments', file=fp)
    for query_id, query in zip(queries_id, queries):
        print(query_id + ',', file=fp, end='')
        query_tf = [0] * 123

        for q in query:
            idx = 0
            for word in query_vocab:
                if word == q:
                    query_tf[idx] += 1
                    break
                idx += 1
        for i in range(123):
            query_tf[i] = ((K3+1) * query_tf[i]) / (K3+query_tf[i])
        query_tf = np.array(query_tf)
        relevance = np.matmul(doc_tfidf, query_tf)
        doc_dict = dict(zip(docs_id, relevance))
        sorted_docs = sorted(doc_dict.items(), key=lambda x: x[1], reverse=True)
        for _doc in sorted_docs:
            doc_id, value = _doc
            print(doc_id , file=fp , end=' ')
        print("",file=fp)
    fp.close()
    return



    
# hyper params
K1 = 3
b = 0.75
K3 = 1000
delta  = 0.2

if __name__ == "__main__":
    avg_doc_len = load_file()
    BM(avg_doc_len)

    