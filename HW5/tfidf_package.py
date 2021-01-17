import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

import pickle
import os
# Config
root_path = './HW5/ntust-ir-2020_hw5_new/'  # Relative path of homework data

# TF-IDF
max_df = 0.95        # Ignore words with high df. (Similar effect to stopword filtering)
min_df = 1           # Ignore words with low df.
smooth_idf = True    # Smooth idf weights by adding 1 to df.
sublinear_tf = True  # Replace tf with 1 + log(tf).

# Rocchio (Below is a param set called Ide Dec-Hi)
alpha = 1
beta = 0.8
gamma = 0.15
rel_count = 5 
nrel_count = 3
iters = 2
bm25topk = 5


documents, queries = [], []

base_dir = './HW5/ntust-ir-2020_hw5_new/'
doc_path = base_dir + 'docs/'
query_path = base_dir + 'queries/'

query_list = []
with open(base_dir + 'query_list.txt', 'r') as query_file_names:
    lines = query_file_names.readlines()
    for line in lines:
        file_path = query_path + line.replace('\n', '.txt')
        fp = open(file_path, 'r')
        query = ' '.join(fp.readline().split())
        queries.append(query)
        query_list.append(line.replace('\n',''))
        fp.close()

doc_list = []
with open(base_dir + 'doc_list.txt', 'r') as doc_file_names:
    lines = doc_file_names.readlines()
    for line in lines:
        file_path = doc_path + line.replace('\n', '.txt')
        fp = open(file_path, 'r')
        doc = ' '.join(fp.readline().split())
        doc_list.append(line.replace('\n',''))
        documents.append(doc)
        fp.close()



# Build TF-IDF vectors of docs and queries
# vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df,
#                              smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)
# doc_tfidfs = vectorizer.fit_transform(documents).toarray()
# query_vecs = vectorizer.transform(queries).toarray()

# feature_names = vectorizer.get_feature_names()

# print(type(feature_names))
# print(len(feature_names))
# with open('./HW5/vocab' + str(len(feature_names)), 'wb') as handle:
#     pickle.dump(feature_names, handle, protocol=pickle.HIGHEST_PROTOCOL)
# exit(0)


doc_tfidfs = np.load('./HW5/doc_tfidfs.npy')
query_vecs = np.load('./HW5/query_vecs.npy')
# np.save('./HW5/doc_tfidfs', doc_tfidfs)
# np.save('./HW5/query_vecs', query_vecs)

bm25 = np.load('./HW5/bm25.npy')
bm25_topk_index = np.empty((len(queries), bm25topk), dtype=int)
for query_idx, sim in enumerate(bm25):
    rel_doc_ids = sim.argsort()[-bm25topk:][::-1]
    for i in range(bm25topk):
        bm25_topk_index[query_idx][i] = rel_doc_ids[i]

plsa = np.load('./HW5/PLSA_relevant.npy')

# Rank documents based on cosine similarity
cos_sim = cosine_similarity(query_vecs, doc_tfidfs)
rankings = np.flip(cos_sim.argsort(), axis=1)

for _iter in range(iters):
    print('iter {}'.format(_iter + 1))
    
    # Update query vectors with Rocchio algorithm
    rel_vecs = doc_tfidfs[rankings[:, :rel_count]].mean(axis=1)        
    nrel_vecs = doc_tfidfs[rankings[:, -nrel_count:]].mean(axis=1)
    if _iter < 2:
        for query_idx in range(len(queries)):
            for i in range(bm25topk):
                rel_vecs[query_idx] += doc_tfidfs[bm25_topk_index[query_idx][i]]
            rel_vecs[query_idx] = rel_vecs[query_idx] / bm25topk
        for query_idx in range(len(queries)):
            plsa_rel = plsa[query_idx]
            for doc_idx in plsa_rel:
                rel_vecs[query_idx] += doc_tfidfs[doc_idx]
            rel_vecs[query_idx] = rel_vecs[query_idx] / plsa_rel.shape[0]
            
    # query_vecs = alpha * query_vecs + beta * rel_vecs - gamma * nrel_vecs
    query_vecs = alpha * query_vecs + beta * rel_vecs
    
    # Rerank documents based on cosine similarity
    cos_sim = cosine_similarity(query_vecs, doc_tfidfs)
    rankings = np.flip(cos_sim.argsort(axis=1), axis=1)

print('ranking.....')
print(rankings.shape)
np.save('./HW5/rankings_rocchio', rankings)

print('cos_sim.....')
print(cos_sim.shape)
np.save('./HW5/cos_sim_rocchio',cos_sim)



with open('./HW5/result_Rocchio.txt', mode='w') as file:
    file.write('Query,RetrievedDocuments\n')
    for query_name, ranking in zip(query_list, rankings):
        ranking = ranking[:5000]
        ranked_docs = ' '.join([doc_list[idx] for idx in ranking])
        file.write('%s,%s\n' % (query_name, ranked_docs))