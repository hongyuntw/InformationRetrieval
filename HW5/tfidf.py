
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np
import pickle
from numba import jit
import pickle
from sklearn.preprocessing import normalize



docs_id = []
queries = []
queries_id = []
docs = []

doc_tfs = []
tf_idfs = []
word2id = {}

def normal(vec):
    return vec / np.linalg.norm(vec)


def load_file():
    base_dir = './HW5/ntust-ir-2020_hw5_new/'
    doc_path = base_dir + 'docs/'
    query_path = base_dir + 'queries/'

    with open('./HW5/vocab34898', 'rb') as handle:
        vocab = pickle.load(handle)
    vocab = set(vocab)

    for file in os.listdir(query_path):
        file_path = query_path + file
        fp = open(file_path, 'r')
        query = fp.readline()
        words = query.split()
        vocab.update(words)
        queries.append(words)
        queries_id.append(file.replace('.txt',''))
        fp.close()

    docs_len  = []
    for file in os.listdir(doc_path):
        file_path = doc_path + file
        fp = open(file_path, 'r')
        doc = fp.readline()
        words = doc.split()
        doc_tf = {}
        # for tf
        for word in words:
            if word in vocab:    
                if word in doc_tf:
                    doc_tf[word] += 1
                else:
                    doc_tf[word] = 1
        docs.append(words)
        doc_tfs.append(doc_tf)
        docs_len.append(len(words))
        docs_id.append(file.replace('.txt', ''))
        fp.close()

    vocab = list(vocab)
    print('len of vocab',  len(vocab))
    num_of_vocab = len(vocab)
    for i in range(num_of_vocab):
        word2id[vocab[i]] = i

    
    num_of_doc = 30000
    doc_word_matrix = np.zeros([num_of_doc, num_of_vocab], dtype=np.int)
    print('counting matrix')
    for doc_idx, doc in enumerate(docs):
        for word in doc:
            try:
                word_idx = word2id[word]
                doc_word_matrix[doc_idx][word_idx] += 1
            except:
                pass
    np.save('./HW5/doc_word_matrix{}'.format(num_of_vocab),doc_word_matrix)
    print('done counting matrix')
    print(doc_word_matrix.shape)




    return vocab , doc_word_matrix  , word2id , docs_len  



def load_file_from_local():
    print('loading')
    base_dir = './HW5/ntust-ir-2020_hw5_new/'
    doc_path = base_dir + 'docs/'
    query_path = base_dir + 'queries/'


    for file in os.listdir(query_path):
        file_path = query_path + file
        fp = open(file_path, 'r')
        query = fp.readline()
        words = query.split()
        queries.append(words)
        queries_id.append(file.replace('.txt',''))
        fp.close()


    docs_len = []
    idx = 0
    for file in os.listdir(doc_path):
        file_path = doc_path + file
        fp = open(file_path, 'r')
        doc = fp.readline()
        words = doc.split()
        docs.append(words)
        docs_len.append(len(words))
        docs_id.append(file.replace('.txt', ''))
        doc_id2idx[file.replace('.txt', '')] = idx
        idx += 1
        fp.close()

    with open("./HW5/vocab11417", "rb") as f:
        vocab = pickle.load(f)
    with open("./HW5/word2id11417", "rb") as f:
        word2id = pickle.load(f)

    doc_word_matrix = np.load('./HW5/doc_word_matrix' + str(len(vocab)) + '.npy')

    with open('./HW5/docs_vocab' + str(len(vocab)), 'rb') as f:
        docs_vocab = pickle.load(f)
    with open('./HW5/query_vocab', 'rb') as f:
        query_vocab = pickle.load(f)

    num_of_vocab = len(vocab)
    num_of_doc = len(docs)
    print('vocab size:',num_of_vocab)

    print(doc_word_matrix.shape)
    print('loading done')

    return vocab , doc_word_matrix  , word2id , docs_len , docs_vocab , query_vocab


def tfidf():
    num_of_doc = 30000
    # calcuate idfs
    word_idf = {}
    for word in vocab:
        idf = 0
        for doc in docs:
            if word in doc:
                idf += 1
        idf = np.log10((num_of_doc+1)/(idf+1))
        word_idf[word] = idf

    # # compute tfidf
    doc_tfidf = []
    for doc_idx in range(num_of_doc):
        tfidf = []
        for word in vocab:
            # calculate tf
            word_idx = word2id[word]
            tf = doc_word_matrix[doc_idx][word_idx]
            if tf > 0:
                tf = 1 + np.log10(tf)
            idf = word_idf[word]
            tfidf.append(tf * idf)
        tfidf = normal(tfidf)
        doc_tfidf.append(tfidf)

    doc_tfidf = np.array(doc_tfidf)
    np.save('./HW5/doc_tfidf',doc_tfidf)
    print(doc_tfidf.shape)


    query_tfidf = []
    for query in queries:
        query_vec = [0.0] * len(vocab)
        for q in query:
            word_idx = word2id[q]
            query_vec[word_idx] = word_idf[q]
        query_vec = normal(query_vec)
        query_tfidf.append(query_vec)

    query_tfidf = np.array(query_tfidf)
    print(query_tfidf.shape)
    np.save('./HW5/query_tfidf',query_tfidf)

    return doc_tfidf , query_tfidf


def Rocchio(doc_tfidf, query_tfidf):
    a = 1
    b = 0.75
    topk_relevance = 4
    topk_non_relevance = 4
    r = 0.15
    max_iter = 4

    for iteration in range(max_iter):
        print("Iteration #" + str(iteration + 1) + "...")

        relevance = cosine_similarity(query_tfidf, doc_tfidf)

        for i in range(len(queries)):
            query_vec = query_tfidf[i]
            doc_relevance = relevance[i]

            topk_doc_idx = np.argsort(doc_relevance)[-topk_relevance:]
            topk_doc_vec = np.array([0.0] * doc_tfidf.shape[1])
            for idx in topk_doc_idx:
                d_v = doc_tfidf[idx]
                topk_doc_vec += d_v
            
            topk_n_doc_idx = idx = np.argsort(doc_relevance)[:topk_non_relevance]
            topk_n_doc_vec = np.array([0.0] * doc_tfidf.shape[1])
            for idx in topk_n_doc_idx:
                d_v = doc_tfidf[idx]
                topk_n_doc_vec += d_v
            
            query_vec = (a * query_vec) + (b * (1 / topk_relevance) * topk_doc_vec) + (r * (1 / topk_non_relevance) * topk_n_doc_vec)
            query_vec = normal(query_vec)
            query_tfidf[i] = query_vec
    
    all_q_relevance = cosine_similarity(query_tfidf, doc_tfidf)
    fp = open('./HW5/result_Rocchio_self.txt', 'w')
    print('Query,RetrievedDocuments', file=fp)
    query_idx = 0
    for query_id, query in zip(queries_id, queries):
        print(query_id + ',', file=fp, end='')
        relevance = all_q_relevance[query_idx]
        doc_dict = dict(zip(docs_id, relevance))
        sorted_docs = sorted(doc_dict.items(), key=lambda x: x[1], reverse=True)
        count = 0

        for _doc in sorted_docs:
            doc_id, value = _doc
            print(doc_id, file=fp, end=' ')
            count += 1
            if count >= 1000:
                break
        query_idx += 1
        print('' , file=fp)
            
    fp.close()


# load_file()
# vocab, doc_word_matrix, word2id, docs_len , docs_vocab , query_vocab = load_file_from_local()
vocab , doc_word_matrix  , word2id , docs_len   = load_file()
doc_tfidf, query_tfidf = tfidf()
# doc_tfidf = np.load('./HW5/doc_tfidf.npy')
# query_tfidf = np.load('./HW5/query_tfidf.npy')
print('----')
print(doc_tfidf.shape)
print(query_tfidf.shape)


Rocchio(doc_tfidf, query_tfidf)
    

    