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


    doc_idx = 0
    docs_len = []
    for file in os.listdir(doc_path):
        file_path = doc_path + file
        fp = open(file_path, 'r')
        doc = fp.readline()
        words = doc.split()
        vocab.update(words)
        docs.append(words)
        docs_id.append(file.replace('.txt',''))
        fp.close()

    vocab = list(vocab)


    num_of_vocab = len(vocab)
    num_of_doc = 14955

    doc_word_matrix = np.zeros([num_of_doc , num_of_vocab], dtype=np.int)

    word2id = {}
    for i in range(num_of_vocab):
        word2id[vocab[i]] = i

    docs_len = []
    print('counting matrix')
    for doc_idx, doc in enumerate(docs):
        word_count_for_document = np.zeros(num_of_vocab, dtype=np.int)
        for word in doc:
            word_idx = word2id[word]
            word_count_for_document[word_idx] += 1
        docs_len.append(len(doc))
        doc_word_matrix[doc_idx] = word_count_for_document
    print('done counting matrix')
    print(doc_word_matrix.shape)


    return vocab , doc_word_matrix  , word2id , docs_len


def normal(vec):
    if np.sum(vec) > 0:
        return vec / np.sum(vec)
    else:
        return vec

    


def plsa(num_of_topic, vocab , doc_word_matrix  , word2id , docs_len):


    num_of_vocab = len(vocab)
    num_of_doc = len(docs)
    print(num_of_vocab,num_of_doc,num_of_topic)

    
    doc_topic_prob = np.empty([num_of_doc, num_of_topic], dtype='float32')  # P(t | d)
    topic_word_prob = np.empty([num_of_topic, num_of_vocab], dtype='float32') # P(w | t)
    topic_prob = np.zeros([num_of_doc, num_of_vocab , num_of_topic ], dtype='float32') # P(t | d, w)
    
    
    #  initialize
    print('initialize')
    doc_topic_prob = np.random.random(size=doc_topic_prob.shape)
    for doc_idx in range(num_of_doc):
        doc_topic_prob[doc_idx] = normal(doc_topic_prob[doc_idx])
    
    topic_word_prob = np.random.random(size=topic_word_prob.shape)
    for topic_idx in range(K):
        topic_word_prob[topic_idx] = normal(topic_word_prob[topic_idx])
    

    loss_pre = -99999.0
    loss = 0.0

    max_iter = 100
    # Run the EM algorithm
    for iteration in range(max_iter):
        print("Iteration #" + str(iteration + 1) + "...")
        print("E step:")
        # e step find p(t|w,d)
        for doc_idx in range(num_of_doc):
            words = list(set(docs[doc_idx]))
            for word in words:
                word_idx = word2id[word]
                s = 0.0
                for topic_idx in range(num_of_topic):
                    topic_prob[doc_idx][word_idx][topic_idx] = topic_word_prob[topic_idx][word_idx] * doc_topic_prob[doc_idx][topic_idx]
                    s += topic_prob[doc_idx][word_idx][topic_idx]
                if s > 0:
                    topic_prob[doc_idx][word_idx] /= s
                
                
                    
        print("M step:")
        print('update P(w | t)')
        # update P(w | t)

        for topic_idx in range(num_of_topic):
            denominater = 0.0
            topic_word_prob[topic_idx]  = 0.0
            for doc_idx in range(num_of_doc):
                words = list(set(docs[doc_idx]))
                for word in words:
                    word_idx = word2id[word]
                    count = doc_word_matrix[doc_idx][word_idx]
                    denominater += count * topic_prob[doc_idx][word_idx][topic_idx]
                    topic_word_prob[topic_idx][word_idx] += count * topic_prob[doc_idx][word_idx][topic_idx]
            if denominater > 0:
                topic_word_prob[topic_idx] /= denominater
            topic_word_prob[topic_idx] = normal(topic_word_prob[topic_idx])


        # update P(t | d)
        print('update P(t | d)')
        for topic_idx in range(num_of_topic):
            for doc_idx in range(num_of_doc):
                words = list(set(docs[doc_idx]))
                s = 0.0
                for word in words:
                    word_idx = word2id[word]
                    count = doc_word_matrix[doc_idx][word_idx]
                    s += count * topic_prob[doc_idx][word_idx][topic_idx]
                doc_topic_prob[doc_idx][topic_idx] = s
                if len(docs[doc_idx]) > 0:
                    doc_topic_prob[doc_idx] /= len(docs[doc_idx])
                doc_topic_prob[doc_idx] = normal(doc_topic_prob[doc_idx])



        loss_pre = loss

        loss = 0.0
        # calculate loss
        for doc_idx in range(num_of_doc):
            words  = list(set(docs[doc_idx]))
            for word in words:
                word_idx = word2id[word]
                count = doc_word_matrix[doc_idx][word_idx]
                s = 0.0
                for topic_idx in range(num_of_topic):
                    s += doc_topic_prob[doc_idx][topic_idx] * topic_word_prob[topic_idx][word_idx]
                loss += count * np.log10(s)
        
        print('loss:' + str(-loss))

        if np.abs(loss-loss_pre) <= 10:
            print('Saving....')
            name = '_K_' + str(num_of_topic) + 'V_' + str(num_of_vocab) + 'iter_' + str(iteration + 1)
            np.save('topic_word_prob'+ name,topic_word_prob)
            np.save('doc_topic_prob'+ name,doc_topic_prob)
            break


        if (iteration+1) % 10 == 0:
            print('Saving....')
            name = '_K_' + str(num_of_topic) + 'V_' + str(num_of_vocab) + 'iter_' + str(iteration + 1)
            np.save('topic_word_prob'+ name,topic_word_prob)
            np.save('doc_topic_prob'+ name,doc_topic_prob)



    # inference
    print('inference....')

    a = 0.7
    b = 0.2

    p_bg = np.sum(doc_word_matrix,axis=0)
    print(p_bg.shape)
    p_bg = p_bg / np.sum(docs_len)
    
    fp = open('./HW4/result.txt', 'w')
    print('Query,RetrievedDocuments', file=fp)
    for query_id, query in zip(queries_id, queries):
        print(query_id + ',', file=fp, end='')
        relevance = []
        for doc_idx in range(num_of_doc):
            q_sum = 1.0
            for word in query:
                word_idx = word2id[word]
                if docs_len[doc_idx]>0:
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
K = 32

    
if __name__ == "__main__":
    vocab , doc_word_matrix  , word2id , docs_len = load_file()
    print('loading done')
    plsa(K,vocab , doc_word_matrix  , word2id , docs_len)


    