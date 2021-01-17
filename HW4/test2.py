import os
import numpy as np
import math
import re
import pickle


docs_id = []
queries = []
docs = []
queries_id = []

word_appear_freq = {}


def vocab_filter(s):
    if len(s) <=2 or bool(re.search(r'\d', s)):
        return False
    return True

def load_file():
    base_dir = './HW4/ntust-ir-2020_hw4_v2/'
    doc_path = base_dir + 'docs/'
    query_path = base_dir + 'queries/'


    vocab = set()

    for file in os.listdir(query_path):
        file_path = query_path + file
        fp = open(file_path, 'r')
        query = fp.readline()
        words = query.split()
        vocab.update(words)
        queries.append(words)
        queries_id.append(file.replace('.txt',''))
        fp.close()


    for file in os.listdir(doc_path):
        file_path = doc_path + file
        fp = open(file_path, 'r')
        doc = fp.readline()
        words = doc.split()
        for word in words:
            if vocab_filter(word):
                vocab.add(word) 
        # vocab.update(words)
        docs.append(words)
        docs_id.append(file.replace('.txt',''))
        fp.close()

    vocab = list(vocab)


    num_of_vocab = len(vocab)
    print('num of vocab : ',num_of_vocab)
    num_of_doc = 14955

    doc_word_matrix = np.zeros([num_of_doc , num_of_vocab], dtype=np.int)
    p_bg = np.zeros(num_of_vocab,dtype='float32')

    word2id = {}
    for i in range(num_of_vocab):
        word2id[vocab[i]] = i

    docs_len = []
    print('counting matrix')
    for doc_idx, doc in enumerate(docs):
        word_count_for_document = np.zeros(num_of_vocab, dtype=np.int)
        for word in doc:
            if word in vocab:
                word_idx = word2id[word]
                word_count_for_document[word_idx] += 1
                p_bg[word_idx] += 1
        doc_word_matrix[doc_idx] = word_count_for_document
    print('done counting matrix')
    print(doc_word_matrix.shape)


    return vocab , doc_word_matrix , p_bg , word2id , docs_len


def normal(vec):
    if np.sum(vec) > 0:
        return vec / np.sum(vec)
    else:
        return vec

    


def plsa(num_of_topic, vocab , doc_word_matrix , p_bg , word2id , docs_len):


    num_of_vocab = len(vocab)
    num_of_doc = len(docs)


    num_of_topic = 16
    a = 0.5
    b = 0.4

    print(num_of_vocab,num_of_doc,num_of_topic)
    
    doc_topic_prob = np.load('./Doc_Topic_10000_1_final.npy') # P(t | d)
    topic_word_prob = np.load('./Topic_Word_10000_1_final.npy') # P(w | t)

    print(doc_topic_prob.shape)
    print(topic_word_prob.shape)
    



    # inference
    print('inference....')

    p_bg = np.sum(doc_word_matrix,axis=0)
    print(p_bg.shape)
    p_bg = p_bg / np.sum(docs_len)

    fp = open('./HW4/result_m.txt', 'w')
    print('Query,RetrievedDocuments', file=fp)
    for query_id, query in zip(queries_id, queries):
        print(query_id + ',', file=fp, end='')
        relevance = []
        for doc_idx in range(num_of_doc):
            q_sum = 1.0
            for word in query:
                word_idx = word2id[word]
                if docs_len[doc_idx] > 0:
                    a_term = a * doc_word_matrix[doc_idx][word_idx] / docs_len[doc_idx]
                s = 0.0
                for topic_idx in range(num_of_topic):
                    s += topic_word_prob[topic_idx][word_idx] * doc_topic_prob[doc_idx][topic_idx]
                b_term = b * s
                ab_term = (1 - a - b) * p_bg[word_idx]
                q_sum = q_sum * (a_term + b_term + ab_term)
            relevance.append(q_sum)
        doc_dict = dict(zip(docs_id, relevance))
        sorted_docs = sorted(doc_dict.items(), key=lambda x: x[1], reverse=True)
        count = 0
        for _doc in sorted_docs:
            doc_id, value = _doc
            print(doc_id, file=fp, end=' ')
            count += 1
            if count >= 1000:
                print('',file=fp)
                break
    fp.close()

            
#  scp nlp@140.118.127.72:~/ir/HW4/result.txt ./





# params
K = 16

    
if __name__ == "__main__":
    vocab , doc_word_matrix , p_bg , word2id , docs_len = load_file()
    print('loading done')
    plsa(K,vocab , doc_word_matrix , p_bg , word2id , docs_len)


    