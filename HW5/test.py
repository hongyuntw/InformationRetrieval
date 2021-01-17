import numpy as np
import os
import numpy as np
import re
import math
from numba import jit,njit
from numba.typed import Dict , List
from numba import types
import pickle


docs_id = []
queries = []
docs = []
queries_id = []

doc_id2idx = {}



def vocab_filter(s):
    if len(s) <=2 or bool(re.search(r'\d', s)):
        return False
    return True


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
    for file in os.listdir(doc_path):
        file_path = doc_path + file
        fp = open(file_path, 'r')
        doc = fp.readline()
        words = doc.split()
        docs.append(words)
        docs_len.append(len(words))
        docs_id.append(file.replace('.txt', ''))
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


def load_file():
    base_dir = './HW5/ntust-ir-2020_hw5_new/'
    doc_path = base_dir + 'docs/'
    query_path = base_dir + 'queries/'


    vocab = set()
    word_tf = {}
    query_vocab = set()

    for file in os.listdir(query_path):
        file_path = query_path + file
        fp = open(file_path, 'r')
        query = fp.readline()
        words = query.split()
        vocab.update(words)
        query_vocab.update(words)
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
        # vocab.update(words)
        for word in words:
            if vocab_filter(word):
                if word in word_tf:
                    word_tf[word] += 1
                else:
                    word_tf[word] = 1 
        docs.append(words)
        docs_len.append(len(words))
        docs_id.append(file.replace('.txt', ''))
        doc_id2idx[file.replace('.txt', '')] = idx
        idx += 1
        fp.close()


    for k , v in word_tf.items():
        if v >= 50:
            vocab.add(k)

    vocab = np.array(list(vocab))


    num_of_vocab = len(vocab)
    num_of_doc = len(docs)
    print('vocab size:',num_of_vocab)




    doc_word_matrix = np.zeros([num_of_doc , num_of_vocab], dtype=np.int)


    word2id = {}

    for i in range(num_of_vocab):
        word2id[vocab[i]] = i

    docs_vocab = []
    print('counting matrix')
    for doc_idx, doc in enumerate(docs):
        doc_vocab = set()
        for word in doc:
            try:
                word_idx = word2id[word]
                doc_word_matrix[doc_idx][word_idx] += 1
                doc_vocab.add(word)
            except:
                pass
        docs_vocab.append(doc_vocab)
    print('done counting matrix')
    print(doc_word_matrix.shape)
    with open('./HW5/vocab' + str(len(vocab)), 'wb') as handle:
        pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./HW5/word2id' + str(len(vocab)), 'wb') as handle:
        pickle.dump(word2id, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./HW5/docs_vocab' + str(len(vocab)), 'wb') as handle:
        pickle.dump(docs_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./HW5/query_vocab', 'wb') as handle:
        pickle.dump(query_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)


    np.save('./HW5/doc_word_matrix' + str(len(vocab)) ,doc_word_matrix )




    return vocab , doc_word_matrix  , word2id , docs_len , docs_vocab , query_vocab


# vocab, doc_word_matrix, word2id, docs_len , docs_vocab , query_vocab = load_file()
vocab, doc_word_matrix, word2id, docs_len , docs_vocab , query_vocab = load_file_from_local()


p_bg = np.sum(doc_word_matrix,axis=0)
print(p_bg.shape)
p_bg = p_bg / np.sum(docs_len)

num_of_vocab = len(vocab)
num_of_doc = len(docs)
num_of_query = len(queries)
print('loading done')


p_tsmm_w = np.load('./HW5/p_tsmm_w0.8_K_5V_11417iter_300.npy')
p_smm = np.load('./HW5/p_smm0.8_K_5V_11417iter_300.npy')




#  expand query

expand_number = 20
expand_queries = []

for query_idx, query in enumerate(queries):
    print(query)
    prob_array = p_smm[query_idx]
    for i in prob_array.argsort()[-expand_number:][::-1]:
        # print(vocab[i])
        query.append(vocab[i])

    query = list(set(query))
    print(query)
    expand_queries.append(query)
    print('=============')
    # break
    #
# with open('./HW5/expand_queries_{}'.format(expand_number) + str(len(vocab)), 'wb') as handle:
#     pickle.dump(expand_queries, handle, protocol=pickle.HIGHEST_PROTOCOL)




# a = 0.4
# b = 0.4


# fp = open('./HW5/result_kl.txt', 'w')
# print('Query,RetrievedDocuments', file=fp)
# query_idx = 0
# for query_id, query in zip(queries_id, queries):
#     print(query_id + ',', file=fp, end='')
#     relevance = []
#     for doc_idx in range(num_of_doc):
#         doc_vocab = docs_vocab[doc_idx]
#         doc_vocab = doc_vocab | query_vocab
#         s = 0.0
#         for word in doc_vocab:
#             word_idx = word2id[word]
#         # for word_idx in range(num_of_vocab):
#             if vocab[word_idx] in query:
#                 a_term = a * (1 / len(query))
#             else:
#                 a_term = 0
#             b_term = b * p_smm[query_idx][word_idx]

#             ab_term = (1 - a - b) * p_bg[word_idx]
#             p_w_d = doc_word_matrix[doc_idx][word_idx] / docs_len[doc_idx]
#             p_w_d = p_w_d +  p_bg[word_idx]
#             # if p_w_d > 0:
#             s += (a_term + b_term + ab_term) * math.log(p_w_d)
#         relevance.append(-s)
#     query_idx += 1
#     doc_dict = dict(zip(docs_id, relevance))
#     sorted_docs = sorted(doc_dict.items(), key=lambda x: x[1], reverse=False)
#     count = 0
#     for _doc in sorted_docs:
#         doc_id, value = _doc
#         print(doc_id, file=fp, end=' ')
#         count += 1
#         if count >= 1000:
#             print('',file=fp)
#             break
# fp.close()




    



    