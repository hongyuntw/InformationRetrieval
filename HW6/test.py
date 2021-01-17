# bm25
import os
import numpy as np
import math
import pickle
import re
import pandas as pd
from tqdm import tqdm
import pickle
import torch
import transformers
from transformers import AdamW
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForQuestionAnswering , RobertaForSequenceClassification
from sklearn.preprocessing import normalize
from transformers import BertTokenizer , BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification



def process_text(text):
    text = str(text)
    text = re.sub(r'[^\w\s]','',text)
    text = text.lower()
    text = text.replace('\n',' ').replace('\t' , ' ')
    text = text.strip()
    return text.split()

    
doc_df = pd.read_csv('documents.csv')
docs_id = doc_df['doc_id'].tolist()
docs = doc_df['doc_text'].apply(process_text).tolist()
doc_id2idx = {}
for i in range(len(docs_id)):
    doc_id2idx[docs_id[i]] = i
train_query_df = pd.read_csv('train_queries.csv')
train_queries = train_query_df['query_text'].apply(process_text).tolist()
train_queries_id = train_query_df['query_id'].astype('str').tolist()
train_queries_pos_docs = train_query_df['pos_doc_ids'].tolist()
train_queries_top_bm25 = train_query_df['bm25_top1000'].tolist()
train_queries_top_bm25_scores = train_query_df['bm25_top1000_scores'].tolist()

print(len(train_queries))
print(len(train_queries_id))
print(len(train_queries_pos_docs))
print(len(train_queries_top_bm25))
print(len(train_queries_top_bm25_scores))
test_query_df = pd.read_csv('test_queries.csv')
test_queries = test_query_df['query_text'].apply(process_text).tolist()
test_queries_id = test_query_df['query_id'].astype('str').tolist()
test_queries_top_bm25 = test_query_df['bm25_top1000'].tolist()
test_queries_top_bm25_scores = test_query_df['bm25_top1000_scores'].tolist()

print(len(test_queries))
print(len(test_queries_id))
print(len(test_queries_top_bm25))
print(len(test_queries_top_bm25_scores))

def normal(vec):
    if np.sum(vec) > 0:
        return vec / np.sum(vec)
    else:
        return vec



class TestDataset(Dataset):
    def __init__(self, input_ids, token_type_ids , attention_mask):
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask        
    def __getitem__(self,idx):
        inputid = self.input_ids[idx]
        tokentype = self.token_type_ids[idx]
        attentionmask = self.attention_mask[idx]
        return inputid , tokentype , attentionmask
    def __len__(self):
        return len(self.input_ids)

def softmax(x):
    """Compute the softmax in a numerically stable way."""
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x




def print_result(bm25_score,model_scores,query_bm25_doc_id,query_id , alpha = 2):
    print(query_id + ',', file=fp, end='')

    scores = bm25_score + alpha * model_scores
    doc_dict = dict(zip(query_bm25_doc_id, scores))
    sorted_docs = sorted(doc_dict.items(), key=lambda x: x[1], reverse=True)
    print(sorted_docs[:20])

    for _doc in sorted_docs:
        doc_id, value = _doc
        print(doc_id, file=fp, end=' ')
    print('',file=fp)


pretrained_model = 'bert-base-uncased'
# pretrained_model = 'bert-large-uncased'
# pretrained_model = "amberoad/bert-multilingual-passage-reranking-msmarco"


# tokenizer = RobertaTokenizer.from_pretrained(pretrained_model)
# model = RobertaForSequenceClassification.from_pretrained(pretrained_model)
# pretrained_model = "amberoad/bert-multilingual-passage-reranking-msmarco"

# tokenizer = BertTokenizer.from_pretrained(pretrained_model)
# model = BertForSequenceClassification.from_pretrained(pretrained_model)

tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
model = AutoModelForSequenceClassification.from_pretrained(pretrained_model)

device='cuda'
BATCH_SIZE = 8

size = 120000
epoch = 1

checkpoint = './{}_epoch{}_size{}'.format(pretrained_model+'2',epoch,size)

# checkpoint = './bert-multilingual-passage-reranking-msmarco-nofreeze-bert_1'
checkpoint = 'bert-base-uncased6_epoch1_size120000'
model.load_state_dict(torch.load(checkpoint,map_location='cpu'))

model.eval().to(device)

# fp = open('./result_{}_{}.txt'.format(size,epoch), 'w')
# print('query_id,ranked_doc_ids', file=fp)

query_model_score = []
for query_idx , query in enumerate(test_queries):
    query_id = test_queries_id[query_idx]
    query = ' '.join(query)
    print(query_idx , query_id , query)

    query_bm25_doc_id = test_queries_top_bm25[query_idx].split()
    tmp_bm25_score = test_queries_top_bm25_scores[query_idx].split()
    
#     bm25 score
    bm25_score = [[]]
    for score in tmp_bm25_score:
        bm25_score[0].append(float(score))


    doc_score = {}
    input_ids = []
    token_type_ids = []
    attention_mask = []
    # for doc_id in query_bm25_doc_id:
    #     doc_idx = doc_id2idx[doc_id]
    for doc_id in query_bm25_doc_id:
        doc_idx = doc_id2idx[doc_id]
        document = ' '.join(docs[doc_idx])
        tmp_dict =  tokenizer(query,
                          document,
                          max_length=512,
                          return_tensors='pt',
                          return_token_type_ids = True,
                          pad_to_max_length=True,
                          padding='max_length',
                          truncation=True)
        input_ids.append(tmp_dict['input_ids'][0])
        token_type_ids.append(tmp_dict['token_type_ids'][0])
        attention_mask.append(tmp_dict['attention_mask'][0])
    test_set = TestDataset(input_ids, token_type_ids,attention_mask)
    loader =  DataLoader(test_set, batch_size=BATCH_SIZE,shuffle=False)
    print(len(loader))
    model_scores = np.array([])
    for data in loader:
        tokens_tensors, segments_tensors, masks_tensors = [t.to(device) for t in data]
        outputs = model(input_ids=tokens_tensors, 
                token_type_ids=segments_tensors, 
                attention_mask=masks_tensors)
        batch_scores = outputs[0][:,1].detach().cpu().numpy()
        model_scores = np.append(model_scores,batch_scores)
    
    # print(model_scores[:5])
    print(model_scores.shape)
    query_model_score.append(model_scores)
    print('-------max,min-------')
    print(max(model_scores),min(model_scores))


# fp.close()

with open('{}_query_model_score{}_{}'.format(pretrained_model+'6',size,epoch), 'wb') as handle:
    pickle.dump(query_model_score, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('model_score_bert_train_on100k', 'wb') as handle:
#     pickle.dump(query_model_score, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('bert-multilingual-passage-reranking-msmarco-nofreeze-2-score', 'wb') as handle:
#     pickle.dump(query_model_score, handle, protocol=pickle.HIGHEST_PROTOCOL)





    
    
    


    

    
    
    


    
