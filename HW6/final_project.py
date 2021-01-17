import torch
import transformers
import pickle
import numpy as np
from transformers import AdamW
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForQuestionAnswering , RobertaForSequenceClassification
from transformers import BertTokenizer , BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import re






def process_text(text):
    text = str(text)
    text = re.sub(r'[^\w\s]','',text)
    text = text.lower()
    text = text.replace('\n',' ').replace('\t' , ' ')
    text = text.strip()
    return text.split()

def get_rel_score(query,document):
    pass



def get_chunks(document,window_size,stride):
    chunks = []
    for i in range(0,len(document),stride):
        chunks.append(document[i : i+window_size])
    return chunks

# def load_data():
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

with open('doc_chunks', 'rb') as handle:
    doc_chunks = pickle.load(handle)

print(len(test_queries))
print(len(test_queries_id))
print(len(test_queries_top_bm25))
print(len(test_queries_top_bm25_scores))
print(len(doc_chunks))

pretrained_model = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
model = AutoModelForSequenceClassification.from_pretrained(pretrained_model)
device='cuda'
checkpoint = 'bert-base-uncased_epoch1_size120000'
model.load_state_dict(torch.load(checkpoint,map_location='cpu'))
model.eval().to(device)

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

def get_relqc(query,chunks):
    input_ids = []
    token_type_ids = []
    attention_mask = []
    query = ' '.join(query)
    for chunk in chunks:
        chunk = ' '.join(chunk)
        tmp_dict =  tokenizer(query,
                          chunk,
                          max_length=20,
                          return_tensors='pt',
                          return_token_type_ids = True,
                          pad_to_max_length=True,
                          padding='max_length',
                          truncation=True)
        input_ids.append(tmp_dict['input_ids'][0])
        token_type_ids.append(tmp_dict['token_type_ids'][0])
        attention_mask.append(tmp_dict['attention_mask'][0])
    BATCH_SIZE = 32
    test_set = TestDataset(input_ids, token_type_ids,attention_mask)
    loader =  DataLoader(test_set, batch_size=BATCH_SIZE,shuffle=False)
    model_scores = np.array([])
    for data in loader:
        tokens_tensors, segments_tensors, masks_tensors = [t.to(device) for t in data]
        outputs = model(input_ids=tokens_tensors, 
                token_type_ids=segments_tensors, 
                attention_mask=masks_tensors)
        batch_scores = outputs[0][:,1].detach().cpu().numpy()
        model_scores = np.append(model_scores,batch_scores)

    # topk_indices = model_scores.argsort()[-topk:][::-1]
    # print(topk_indices)
    return model_scores
def get_relcd(chunks,document):
    input_ids = []
    token_type_ids = []
    attention_mask = []
    document = ' '.join(document)
    for chunk in chunks:
        query = ' '.join(chunk)
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
    BATCH_SIZE = 16
    test_set = TestDataset(input_ids, token_type_ids,attention_mask)
    loader =  DataLoader(test_set, batch_size=BATCH_SIZE,shuffle=False)
#     print(len(loader))
    model_scores = np.array([])
    for data in loader:
        tokens_tensors, segments_tensors, masks_tensors = [t.to(device) for t in data]
        outputs = model(input_ids=tokens_tensors, 
                token_type_ids=segments_tensors, 
                attention_mask=masks_tensors)
        batch_scores = outputs[0][:,1].detach().cpu().numpy()
        model_scores = np.append(model_scores,batch_scores)

    # topk_indices = model_scores.argsort()[-topk:][::-1]
    # print(topk_indices)
    return model_scores

with open('doc_chunk_dict', 'rb') as handle:
    doc_chunk_dict = pickle.load(handle)
with open('rel_query_topkdoc_chunks', 'rb') as handle:
    rel_query_topkdoc_chunks = pickle.load(handle)
with open('rel_qds', 'rb') as handle:
    rel_qds = pickle.load(handle)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

topkd = 10
topkc = 10
rel_query_topkdoc_chunks = []
phase3_scores = []
query_rel_cd_scores = []

# phase2_scores = []
with open('phase2_scores', 'rb') as handle:
    phase2_scores = pickle.load(handle)

# with open('phase2_scores', 'rb') as handle:
#     phase2_scores = pickle.load(handle)
for i in range(len(test_queries)):
    query = test_queries[i]
    query_id = test_queries_id[i]
    print(i,query_id , query)
    rel_qd = rel_qds[i]
    bm25_top1000_document_ids = test_queries_top_bm25[i].split()


    topk_first_rerank_docs_indices = rel_qd.argsort()[-topkd:][::-1]
    
#     chunks_in_topk_doc = []
#     scores_in_topk_doc = []
    
    be_selected_chunks = []
    
    for rerank_docs_index in topk_first_rerank_docs_indices:
        doc_id = bm25_top1000_document_ids[rerank_docs_index]
        doc_idx = doc_id2idx[doc_id]
        chunks = doc_chunks[doc_idx]
        be_selected_chunks.extend(chunks)
    
#     phase2_score = get_relqc(query,be_selected_chunks)
#     phase2_scores.append(phase2_score)
    
    phase2_score = phase2_scores[i]
    topk_chunks = []
    rel_qc_score = []
    
    topk_chunks_indices = phase2_score.argsort()[-topkc:][::-1]
    for index in topk_chunks_indices:
        topk_chunks.append(be_selected_chunks[index])
        rel_qc_score.append(phase2_score[index])
    
    softmax_rel_qc_score = softmax(rel_qc_score)

#     phase 3
    rel_qcds = []
    rel_cd_scores = []
    for doc_id in bm25_top1000_document_ids:
        doc_idx = doc_id2idx[doc_id]
        doc = docs[doc_idx]
        rel_cd_score = get_relcd(topk_chunks,doc)
        rel_cd_scores.append(rel_cd_score)
    print(len(rel_cd_scores))
    query_rel_cd_scores.append(rel_cd_scores)
with open('query_rel_cd_scores', 'wb') as handle:
    pickle.dump(query_rel_cd_scores, handle, protocol=pickle.HIGHEST_PROTOCOL) 
        
#     phase3_scores.append(rel_qcds)

# with open('phase2_scores', 'wb') as handle:
#     pickle.dump(phase2_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)      


# #         doc = docs[doc_idx]
# # #         scores = get_relcd(chunks,doc)
# #         chunks , scores  = doc_chunk_dict[doc_id]
# #         chunks_in_topk_doc.extend(chunks)
# #         scores_in_topk_doc.extend(scores)
#     chunks_in_topk_doc = np.array(chunks_in_topk_doc)
#     scores_in_topk_doc = np.array(scores_in_topk_doc)
    

        
#     qc_scores = get_relqc(query,topk_chunks)
#     softmax_qc_scores = softmax(qc_scores)
        
        
# #     phase 3


    

# # with open('doc_chunk_dict', 'wb') as handle:
# #     pickle.dump(doc_chunk_dict, handle, protocol=pickle.HIGHEST_PROTOCOL) 


# with open('phase3_scores', 'wb') as handle:
#     pickle.dump(phase3_scores, handle, protocol=pickle.HIGHEST_PROTOCOL) 

 