import os
import numpy as np
import re
from numba import jit,njit
from numba.typed import Dict , List
from numba import types
import pickle
from sklearn.preprocessing import normalize
from tqdm import tqdm
import math

docs_id = []
queries = []
docs = []
queries_id = []
query_len = []
doc_id2idx = {}

np.set_printoptions(suppress=True)

def vocab_filter(s):
    if len(s) <= 2 or bool(re.search(r'\d', s)):
        return False
    return True



def load_file_from_local():
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
        vocab.update(words)
        docs.append(words)
        docs_len.append(len(words))
        docs_id.append(file.replace('.txt', ''))
        doc_id2idx[file.replace('.txt', '')] = idx
        idx += 1
        fp.close()

    docs_len = np.array(docs_len)
    vocab = np.array(list(vocab))

    num_of_vocab = len(vocab)
    num_of_doc = len(docs)
    num_of_query = len(queries)
    print('vocab size:',num_of_vocab)

 

    # with open('./HW5/vocab' + str(len(vocab)), 'rb') as handle:
    #     vocab = pickle.load(handle)
    with open('./HW5/word2id' + str(len(vocab)), 'rb') as handle:
        word2id = pickle.load(handle)
    # with open('./HW5/query_vocab', 'rb') as handle:
    #     query_vocab = pickle.load(handle)
    doc_word_matrix = np.load('./HW5/doc_word_matrix' + str(len(vocab)) + '.npy')
    query_word_matrix = np.load('./HW5/query_word_matrix' + str(len(vocab)) + '.npy')
    print('doc_word_matrix' , doc_word_matrix.shape)
    print('query_word_matrix',query_word_matrix.shape)


    return vocab , doc_word_matrix  , word2id , docs_len , query_vocab , query_word_matrix

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
        vocab.update(words)
        docs.append(words)
        docs_len.append(len(words))
        docs_id.append(file.replace('.txt', ''))
        doc_id2idx[file.replace('.txt', '')] = idx
        idx += 1
        fp.close()

    docs_len = np.array(docs_len)
    vocab = np.array(list(vocab))

    num_of_vocab = len(vocab)
    num_of_doc = len(docs)
    num_of_query = len(queries)
    print('vocab size:',num_of_vocab)

    doc_word_matrix = np.zeros([num_of_doc , num_of_vocab], dtype=np.int)
    query_word_matrix = np.zeros([num_of_query , num_of_vocab], dtype=np.int)


    word2id = {}

    for i in range(num_of_vocab):
        word2id[vocab[i]] = i

    print('counting doc matrix')
    for doc_idx, doc in enumerate(docs):
        for word in doc:
            try:
                word_idx = word2id[word]
                doc_word_matrix[doc_idx][word_idx] += 1
            except:
                pass
    print('done counting doc matrix')
    print(doc_word_matrix.shape)

    print('counting query matrix')
    for query_idx, query in enumerate(queries):
        for word in query:
            try:
                word_idx = word2id[word]
                query_word_matrix[query_idx][word_idx] += 1
            except:
                pass
    print('done counting query matrix')
    print(query_word_matrix.shape)


    with open('./HW5/vocab' + str(len(vocab)), 'wb') as handle:
        pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./HW5/word2id' + str(len(vocab)), 'wb') as handle:
        pickle.dump(word2id, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open('./HW5/docs_vocab' + str(len(vocab)), 'wb') as handle:
    #     pickle.dump(docs_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./HW5/query_vocab', 'wb') as handle:
        pickle.dump(query_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
    np.save('./HW5/doc_word_matrix' + str(len(vocab)), doc_word_matrix)
    np.save('./HW5/query_word_matrix' + str(len(vocab)) ,query_word_matrix )


    return vocab , doc_word_matrix  , word2id , docs_len , query_vocab , query_word_matrix


def load_file_small_vocab_from_local():
    print('loading')
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
        queries.append(words)
        vocab.update(words)
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

    with open("./HW5/vocab" + str(len(vocab)), "rb") as f:
        vocab = pickle.load(f)
    with open("./HW5/word2id" + str(len(vocab)), "rb") as f:
        word2id = pickle.load(f)

    doc_word_matrix = np.load('./HW5/doc_word_matrix' + str(len(vocab)) + '.npy')
    query_word_matrix = np.load('./HW5/query_word_matrix' + str(len(vocab)) + '.npy')

    with open('./HW5/docs_vocab' + str(len(vocab)), 'rb') as f:
        docs_vocab = pickle.load(f)
    with open('./HW5/query_vocab', 'rb') as f:
        query_vocab = pickle.load(f)

    num_of_vocab = len(vocab)
    num_of_doc = len(docs)
    print('vocab size:',num_of_vocab)

    print('doc_word_matrix', doc_word_matrix.shape)
    print('query_word_matrix',query_word_matrix.shape)
    print('loading done')

    return vocab , doc_word_matrix  , word2id , docs_len , docs_vocab , query_vocab , query_word_matrix


def load_file_small_vocab():
    base_dir = './HW5/ntust-ir-2020_hw5_new/'
    doc_path = base_dir + 'docs/'
    query_path = base_dir + 'queries/'


    # vocab = set()
    with open('./HW5/vocab34898', 'rb') as handle:
        vocab = pickle.load(handle)
    vocab = set(vocab)

    query_vocab = set()

    with open(base_dir + 'query_list.txt', 'r') as query_file_names:
        lines = query_file_names.readlines()
        for line in lines:
            file_path = query_path + line.replace('\n', '.txt')
            fp = open(file_path, 'r')
            query = fp.readline()
            words = query.split()
            query_len.append(len(words))
            vocab.update(words)
            query_vocab.update(words)
            queries.append(words)
            queries_id.append(line.replace('\n', ''))
            fp.close()


    docs_len = []
    word_tf = {}

    with open(base_dir + 'doc_list.txt', 'r') as doc_file_names:
        lines = doc_file_names.readlines()
        for line in lines:
            file_path = doc_path + line.replace('\n', '.txt')
            fp = open(file_path, 'r')
            doc = fp.readline()
            words = doc.split()
            # for word in words:
            #     if word in word_tf:
            #         word_tf[word] += 1
            #     else:
            #         word_tf[word] = 1
            docs.append(words)
            docs_len.append(len(words))
            docs_id.append(line.replace('\n', ''))
            fp.close()


    # for k , v in word_tf.items():
    #     if 10000 > v > 5:
    #         vocab.add(k)


    vocab = np.array(list(vocab))


    num_of_vocab = len(vocab)
    num_of_doc = len(docs)
    num_of_query = len(queries)
    print('vocab size:',num_of_vocab)


    doc_word_matrix = np.zeros([num_of_doc , num_of_vocab], dtype=np.int)
    query_word_matrix = np.zeros([num_of_query, num_of_vocab], dtype=np.int)


    word2id = {}
    for i in range(num_of_vocab):
        word2id[vocab[i]] = i

    for query_idx, query in enumerate(queries):
        for word in query:
            word_idx = word2id[word]
            query_word_matrix[query_idx][word_idx] += 1
    print(query_word_matrix.shape)
    print('done query matrix')


    docs_vocab = []
    print('counting matrix')
    for doc_idx, doc in enumerate(docs):
        count = 0
        for word in doc:
            try:
                word_idx = word2id[word]
                doc_word_matrix[doc_idx][word_idx] += 1
                count += 1
            except Exception as e:
                pass
        # docs_vocab.append(doc_vocab)
    print('done counting matrix')
    print(doc_word_matrix.shape)
    # with open('./HW5/vocab' + str(len(vocab)), 'wb') as handle:
    #     pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open('./HW5/word2id' + str(len(vocab)), 'wb') as handle:
    #     pickle.dump(word2id, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open('./HW5/docs_vocab' + str(len(vocab)), 'wb') as handle:
    #     pickle.dump(docs_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # np.save('./HW5/query_word_matrix' + str(len(vocab)), query_word_matrix)
    # np.save('./HW5/doc_word_matrix' + str(len(vocab)) ,doc_word_matrix )

    return vocab , doc_word_matrix  , word2id , docs_len , docs_vocab , query_vocab , query_word_matrix


@njit()
def normal(vec):
    if np.sum(vec) > 0:
        return vec / np.sum(vec)
    else:
        return vec

# @jit(nopython=True)
# def matmul(mat1, mat2):
#     mat = np.empty(shape=(mat1.shape[0], mat2.shape[1]), dtype=np.float32)
#     for i in range(mat1.shape[0]):
#         for j in range(mat2.shape[1]):
#             s = 0
#             for k in range(mat2.shape[0]):
#                 s += mat1[i,k] * mat2[k,j]
#             mat[i,j] = s
#     return mat



def main():
    # Q * W
    # error
    global query_len
    p_tsmm_w = np.random.uniform(1.0, 10.0, size=(num_of_query, num_of_vocab))
    p_tsmm_w = normalize(p_tsmm_w, axis=1, norm='l1')
    # p_tsmm_w = np.load('./HW5/p_tsmm_w57.npy')
    # p_smm = np.load('./HW5/p_smm57.npy')

    p_smm = np.random.uniform(1.0, 10.0,size=(num_of_query, num_of_vocab))
    p_smm = normalize(p_smm, axis=1, norm='l1')

    print('p_tsmm_w shape', p_tsmm_w.shape)
    print('p_smm shape' , p_smm.shape)
    query_len = np.array(query_len)

    print('query len', query_len.shape)
    p_ulm = query_word_matrix / query_len.reshape(query_word_matrix.shape[0], -1)
    p_w_d = doc_word_matrix / docs_len.reshape(doc_word_matrix.shape[0], -1)
    # D * W
    kl = np.zeros([num_of_query, num_of_doc])


    max_iter = main_iter
    for iteration in range(max_iter):
        print("Main Iteration #" + str(iteration + 1) + "...")
        p_smm , p_tsmm_w = EM(p_smm , p_tsmm_w)
        print('-----KL------')
        kl = KL(query_word_matrix, p_smm, p_bg, doc_word_matrix, kl_a, kl_b , p_w_d , kl)
        print('kl',kl.shape)
        print('-----KL done-----')
        print('-----update Rq_matrix')
        update_Rq_matrix(kl, update_topk, iteration)

    np.save('./HW5/kl_result{}_{}_{}_{}_{}_{}_{}_{}'.format(main_iter,em_iter,kl_a,kl_b,e_a,r,init_topk,update_topk), kl)
    # np.save('./HW5/p_smm', p_smm)
    # np.save('./HW5/p_tsmm_w', p_tsmm_w)
    
    print_result(kl)


def update_Rq_matrix(kl,topk,iter):
    global Rq_matrix
    global init_Rq_matrix
    # topk = topk + iter * topk
    # print('iter{} _ topk{}'.format(iter,topk))
    # if iter > 2:
    #     Rq_matrix = np.zeros([num_of_query , num_of_doc], dtype=np.float32)
    # Rq_matrix = init_Rq_matrix
    for query_idx, ranking in enumerate(kl):
        # kl means distance  , get topk smallest elememt in ranking
        topk_smallest_idx = np.argpartition(ranking, topk)
        for i in range(topk):
            Rq_matrix[query_idx][topk_smallest_idx[i]] = 1


def EM(p_smm,p_tsmm):
    max_iter = em_iter
    for iteration in range(max_iter):
        print("SMM Iteration #" + str(iteration + 1) + "...")
        print('E step')
        p_tsmm_w = SMM_e_step(e_a , p_smm)
        p_tsmm_w = normalize(p_tsmm_w, axis=1, norm='l1')
        print('M step')
        p_smm = SMM_m_step(Rq_matrix, p_tsmm_w , doc_word_matrix)
        p_smm = normalize(p_smm, axis=1, norm='l1')
    check_psmm(p_smm)
    return p_smm , p_tsmm_w

def check_psmm(p_smm):
    for query_idx in range(num_of_query):
        print(queries[query_idx])
        for i in p_smm[query_idx].argsort()[-10:][::-1]:
            print(vocab[i])
        break


@njit()
def SMM_e_step(a , p_smm):
    # Q * W
    up = p_smm * (1 - a)
    # Q * W
    repeat_p_bg = p_bg.repeat(num_of_query).reshape((-1, num_of_query)).transpose()
    down = (p_smm * (1 - a)) + (repeat_p_bg * a)    
    # point wise divide
    p_tsmm_w = np.divide(up, down)
 
    return p_tsmm_w

@njit()
def SMM_m_step(Rq_matrix , p_tsmm_w , doc_word_matrix):
    # Rq_matrix shape is Q * D and  only contain zero or one
    # Q * W
    up = np.dot(Rq_matrix, doc_word_matrix)
    up = up * p_tsmm_w
    # down shape should be Q * 1
    down = np.sum(up, axis=1)
    # each row divide by q in Q
    up = up / down.reshape(up.shape[0], -1)
    return up

@njit()
def KL(p_ulm, p_smm , p_bg , doc_word_matrix , a , b , p_w_d , kl):
    # p_ulm = query_word_matrix / (sum for each row)
    # ULM shape Q * W
    p_ulm = p_ulm * a
    p_smm = p_smm * b
    scale_p_bg = p_bg * (1 - a - b)
    repeat_p_bg = scale_p_bg.repeat(num_of_query).reshape((-1,num_of_query)).transpose()
    # x shape  Q * W
    x = p_ulm + p_smm + repeat_p_bg
    for i in range(x.shape[0]):
        x[i] = normal(x[i])

    # y shape D * W
    repeat_p_bg_by_d = p_bg.repeat(num_of_doc).reshape((-1, num_of_doc)).transpose()
    y = (r * p_w_d) + ((1 - r) * repeat_p_bg_by_d)
    for i in range(y.shape[0]):
        y[i] = normal(y[i])
    y = np.log(y)
    # kl shape Q * D = kl(q||d)
    kl = -1 * np.dot(x, y.transpose())


    return kl

    
def print_result(kl , reverse=False):
    fp = open('./HW5/result_kl.txt', 'w')
    print('Query,RetrievedDocuments', file=fp)
    query_idx = 0
    for query_id, query in zip(queries_id, queries):
        print(query_id + ',', file=fp, end='')
        relevance = kl[query_idx]
        for doc_idx in np.argsort(relevance)[:5000]:
            doc_id = docs_id[doc_idx]
            print(doc_id, file=fp, end=' ')        
        print('',file=fp)
        query_idx += 1
        # doc_dict = dict(zip(docs_id, relevance))
        # sorted_docs = sorted(doc_dict.items(), key=lambda x: x[1], reverse=reverse)
        # count = 0
        # for _doc in sorted_docs:
        #     doc_id, value = _doc
        #     print(doc_id, file=fp, end=' ')
        #     count += 1
        #     if count >= 5000:
        #         print('',file=fp)
        #         break
        # query_idx += 1       
    fp.close()


vocab, doc_word_matrix, word2id, docs_len, docs_vocab, query_vocab, query_word_matrix = load_file_small_vocab()


# vocab, doc_word_matrix, word2id, docs_len, docs_vocab, query_vocab, query_word_matrix = load_file_small_vocab_from_local()

# vocab , doc_word_matrix  , word2id , docs_len  , query_vocab , query_word_matrix  = load_file()
# vocab , doc_word_matrix  , word2id , docs_len  , query_vocab , query_word_matrix  = load_file_from_local()


# hparams
main_iter = 3
em_iter = 10
kl_a = 0.1
kl_b = 0.8
e_a = 0.3
r = 0.3
# p_w_d = r * p_w_d + ((1-r)) * repeat_p_bg_by_d)
init_topk = 5
update_topk = 35

docs_len = np.array(docs_len)
print('total documents len:',np.sum(docs_len))
p_bg = np.sum(doc_word_matrix, axis=0)
p_bg = p_bg / np.sum(docs_len)
p_bg = normal(p_bg)

print('p_bg shape ' , p_bg.shape)

num_of_vocab = len(vocab)
num_of_doc = len(docs)
num_of_query = len(queries)
print('doc , vocab , query')
print(num_of_doc, num_of_vocab, num_of_query)

    



# Q * D
rankings = np.load('./HW5/rankings_rocchio.npy')
Rq_matrix = np.zeros([num_of_query , num_of_doc], dtype=np.float32)
for query_idx, ranking in enumerate(rankings):
    for i in range(init_topk):
        Rq_matrix[query_idx][ranking[i]] = 1
print('Rq_matrix' , Rq_matrix.shape)

# rochhio2 = np.load('./HW5/rocchio_relevant.npy')
# for query_idx, ranking in enumerate(rochhio2):
#     for doc_idx in ranking:
#         Rq_matrix[query_idx][int(doc_idx)] = 1
        
bm25 = np.load('./HW5/bm25.npy')
print('bm25' , bm25.shape)
Rq_matrix = np.zeros([num_of_query, num_of_doc], dtype=np.float32)
for query_idx, ranking in enumerate(bm25):
    # rel_doc_ids = np.nonzero(ranking)
    rel_doc_ids = ranking.argsort()[-init_topk:][::-1]
    for doc_idx in rel_doc_ids:
        Rq_matrix[query_idx][doc_idx] = 1

Rq_matrix = np.zeros([num_of_query, num_of_doc], dtype=np.float32)
rankings = np.load('./HW5/kl_result.npy')
for query_idx, ranking in enumerate(rankings):
    for doc_idx in np.argsort(ranking)[:init_topk]:
        Rq_matrix[query_idx][doc_idx] = 1



init_Rq_matrix = Rq_matrix
doc_word_matrix = doc_word_matrix.astype(Rq_matrix.dtype)

main()
# 11352166