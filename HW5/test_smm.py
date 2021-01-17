import os
import numpy as np
import re
from numba import jit,njit
from numba.typed import Dict , List
from numba import types
import pickle
import math
np.random.seed(21312)

docs_id = []
queries = []
docs = []
queries_id = []

# doc_id2idx =  Dict.empty(
#     key_type=types.unicode_type,
#     value_type=types.int32,
# )
doc_id2idx = {}


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

    return vocab, doc_word_matrix, word2id, docs_len, docs_vocab, query_vocab
    

def print_result(topk_doc_for_queries):
    fp = open('./HW5/result_kl.txt', 'w')
    print('Query,RetrievedDocuments', file=fp)
    query_idx = 0
    for query_id, query in zip(queries_id, queries):
        print(query_id + ',', file=fp, end='')
        doc_ids_for_query = topk_doc_for_queries[query_idx]
        for doc_id in doc_ids_for_query:
            print(doc_id, file=fp, end=' ')
        print('',file=fp)
                
    fp.close()

vocab, doc_word_matrix, word2id, docs_len , docs_vocab , query_vocab = load_file_from_local()
p_bg = np.sum(doc_word_matrix,axis=0)
print(p_bg.shape)
p_bg = p_bg / np.sum(docs_len)

num_of_vocab = len(vocab)
num_of_doc = len(docs)
num_of_query = len(queries)

print(num_of_doc, num_of_vocab)


p_tsmm_w = np.load('./HW5/p_smm_itermode11417iter_5.npy')
p_smm = np.load('./HW5/p_tsmm_w_itermode11417iter_5.npy')
with open("./HW5/topk_doc_for_queries5", "rb") as f:
    topk_doc_for_queries = pickle.load(f)

print_result(topk_doc_for_queries)