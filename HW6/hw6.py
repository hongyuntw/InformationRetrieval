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

# pretrained_model = 'roberta-large'
# pretrained_model = 'bert-large-uncased'
# pretrained_model = "amberoad/bert-multilingual-passage-reranking-msmarco"
pretrained_model = 'bert-base-uncased'

device = 'cuda'


# tokenizer = RobertaTokenizer.from_pretrained(pretrained_model)
# model = RobertaForSequenceClassification.from_pretrained(pretrained_model)

tokenizer = BertTokenizer.from_pretrained(pretrained_model)
model = BertForSequenceClassification.from_pretrained(pretrained_model)


data_size = 120000
pretrained_model = 'bert-base-uncased'

# with open('{}_input_dict{}'.format(pretrained_model,data_size), 'rb') as handle:
#     input_dict = pickle.load(handle)

with open('{}_input_ids{}'.format(pretrained_model,data_size), 'rb') as handle:
    input_ids = pickle.load(handle)

with open('{}_attention_mask{}'.format(pretrained_model,data_size), 'rb') as handle:
    attention_mask = pickle.load(handle)

with open('{}_token_type_ids{}'.format(pretrained_model,data_size), 'rb') as handle:
    token_type_ids = pickle.load(handle)

# with open('{}_train_y{}'.format(pretrained_model,data_size), 'rb') as handle:
#     train_y = pickle.load(handle)

with open('train_y{}'.format(data_size), 'rb') as handle:
    train_y = pickle.load(handle)
    
# print(input_dict['input_ids'].size())
print(len(input_ids))
print(len(attention_mask))
print(len(token_type_ids))
print(len(train_y))
    


# class TrainDataset(Dataset):
#     def __init__(self, input_dict, y):
#         self.input_ids = input_dict['input_ids']
#         self.token_type_ids = input_dict['token_type_ids']
#         self.attention_mask = input_dict['attention_mask']
#         self.y = y
        
#     def __getitem__(self,idx):
#         inputid = self.input_ids[idx]
#         tokentype = self.token_type_ids[idx]
#         attentionmask = self.attention_mask[idx]
#         y = self.y[idx]
#         return inputid , tokentype , attentionmask, y
    
#     def __len__(self):
#         return len(self.input_ids)

class TrainDataset(Dataset):
    def __init__(self, input_ids ,token_type_ids , attention_mask  , y):
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.y = y
        
    def __getitem__(self,idx):
        inputid = self.input_ids[idx]
        tokentype = self.token_type_ids[idx]
        attentionmask = self.attention_mask[idx]
        y = self.y[idx]
        return inputid , tokentype , attentionmask, y
    
    def __len__(self):
        return len(self.input_ids)
    
BATCH_SIZE = 8
# trainSet = TrainDataset(input_dict, train_y)
trainSet = TrainDataset(input_ids ,token_type_ids , attention_mask  , train_y)

trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE,shuffle=True,drop_last=False)
print(len(trainLoader))

lr = 1e-5
# optimizer = AdamW(model.parameters(), lr=lr)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

weight = torch.FloatTensor([4, 100]).to(device)
binary_loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
# binary_loss_fct = torch.nn.CrossEntropyLoss()

model = model.to(device)
model.train() 

# for name, param in model.named_parameters():
# 	if 'classifier' not in name: # classifier layer
# 		param.requires_grad = False

EPOCHS = 2

for epoch in range(EPOCHS):
    running_loss = 0.0
    count = 0
    correct_count = 0
    for data in trainLoader:
        tokens_tensors, segments_tensors, masks_tensors, labels = [t.to(device) for t in data]

        optimizer.zero_grad()
        outputs = model(input_ids=tokens_tensors, 
                      token_type_ids=segments_tensors, 
                      attention_mask=masks_tensors)
        
        pred = outputs[0].argmax(1)
        count += labels.size()[0]
        correct_count += (labels==pred).sum().cpu().item()
        
        loss = binary_loss_fct(outputs[0],labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        print(f'\r Epoch:{epoch}/{EPOCHS} Acc% : {correct_count/count} Data:{count}/{data_size}',end='')
        
    CHECKPOINT_NAME = './{}_epoch{}_size{}'.format(pretrained_model+'7',epoch,data_size) 
    # CHECKPOINT_NAME = './bert-multilingual-passage-reranking-msmarco-nofreeze-bert-100k_{}'.format(epoch)
    # CHECKPOINT_NAME = './bert-multilingual-passage-reranking-msmarco-nofreeze-3-120k_{}'.format(epoch)

    torch.save(model.state_dict(), CHECKPOINT_NAME)
    print('\n epoch {} done'.format (epoch))
