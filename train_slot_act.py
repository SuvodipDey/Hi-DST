"""
Description :
Slot-action prediction model.
Finetunes bert-base-uncased model on slot-action prediction data.

Run Command:
python train_slot_act.py -in=<path of the input data> -path=<output dir> -src_file=<name of the python script>
"""
#--------------------------------------------

import torch
import torchtext.vocab as vocab
import random
import math
import time
import argparse
import os
import shutil
import pandas as pd
import numpy as np
import torch
import transformers
from transformers import BertTokenizer, BertModel, AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef

#-----------------------------------------

SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)      
    device = torch.device("cuda")
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
    
parser = argparse.ArgumentParser()
parser.add_argument('-path','--path', help='path of the working directiory', required=True)
parser.add_argument('-src_file','--src_file', help='path of the source file', required=False, default='')
parser.add_argument('-in','--in', help='path of the input files', required=True)
args = vars(parser.parse_args())
work_dir = args['path']
src_file = args['src_file']
in_dir = args['in']

print("Path of the working directory : {}".format(work_dir))
if(not os.path.isdir(work_dir)):
    print("Directory does not exist.")
    exit(0)
    
print("Path of the input directory : {}".format(in_dir))
if(not os.path.isdir(in_dir)):
    print("Directory does not exist.")
    exit(0)
    
if(src_file):
    try:
        shutil.copy(src_file, work_dir)
    except:
        print("File {} failed to get copied to {}".format(src_file, work_dir))

#-----------------------------------------

domain_list = ['police', 'restaurant', 'hotel', 'taxi', 'attraction', 'train', 'hospital']  

slot_detail = {'Type': 'type', 'Price': 'price', 'Parking': 'parking', 'Stay': 'stay', 'Day': 'day', 
               'People': 'people', 'Post': 'post', 'Addr': 'address', 'Dest': 'destination', 'Arrive': 'arrive', 
               'Depart': 'departure', 'Internet': 'internet', 'Stars': 'stars', 'Phone': 'phone', 'Area': 'area', 
               'Leave': 'leave', 'Time': 'time', 'Ticket': 'ticket', 'Ref': 'reference', 'Food': 'food', 
               'Name': 'name', 'Department': 'department', 'Fee': 'fee', 'Id': 'id', 'Car': 'car'}
  
LABEL_COLUMN = "slot_act"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#-----------------------------------------
    
def get_word(word):
    return glove.vectors[glove.stoi[word]]

def prepare_data(df, max_len, batch_size, slot_dict, domain_dict):
    input_ids = []
    attention_masks = []
    for i in df.index:
        # Encode system and user utterance pair
        encoded_dict = tokenizer.encode_plus(
                                df['sys'][i].lower(),df['usr'][i].lower(),
                                add_special_tokens = True,
                                padding='max_length', 
                                truncation=True,
                                max_length = max_len,
                                return_attention_mask = True,
                                return_tensors = 'pt',
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    
    slot_ids = [slot_dict[s] for s in df.slot]
    slot_ids = torch.tensor(slot_ids)
    
    domain_ids = [domain_dict[s] for s in df.domain]
    domain_ids = torch.tensor(domain_ids)
    
    labels = torch.tensor(df[LABEL_COLUMN])
    
    dataset = TensorDataset(input_ids, attention_masks, slot_ids, domain_ids, labels)
    dataloader = DataLoader(
            dataset,
            sampler = RandomSampler(dataset),
            batch_size = batch_size
        )
    return dataloader

class Model(nn.Module):
    def __init__(self, weights_matrix, domain_matrix, num_labels):
        super(Model, self).__init__()
        self.encode = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights_matrix))
        self.embedding_domain = nn.Embedding.from_pretrained(torch.FloatTensor(domain_matrix))
        self.drop_out = nn.Dropout(0.3)
        self.gelu = nn.GELU()
        self.l1 = nn.Linear(300*2, 768)
        self.l2 = nn.Linear(768*2, num_labels)
        self.smax = nn.Softmax(dim=1)
    
    def forward(self, input_ids, attention_masks, slot_ids, domain_ids):
        outputs = self.encode(input_ids, attention_masks)
        with torch.no_grad(): #Freezing GloVe embeddings
            slot_embeddings = self.embedding(slot_ids)
            domain_embeddings = self.embedding_domain(domain_ids)
            input2 = torch.cat((slot_embeddings, domain_embeddings), 1)

        input1 = outputs[2][-2]
        input2 = self.l1(input2)
        input2 = self.gelu(input2)
        input3=torch.unsqueeze(input2, -1)
        
        a = torch.matmul(input1, input3)/28.0
        a = self.smax(torch.squeeze(a, -1))
        a = torch.unsqueeze(a, -1)
        input1 = input1.permute(0, 2, 1)
        input1 = torch.matmul(input1, a)
        input1 = torch.squeeze(input1,-1)
        
        output = torch.cat((input1, input2), 1)
        output = self.drop_out(output)
        output = self.l2(output)
        return output

def evaluate_metrics(dataloader, model):
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            b_input_ids = batch[0].to(device)
            b_attn_mask = batch[1].to(device)
            b_slot_ids = batch[2].to(device)
            b_domain_ids = batch[3].to(device)
            labels = batch[4].to(device)
             
            outputs = model(b_input_ids, b_attn_mask, b_slot_ids, b_domain_ids)
            loss = criterion(outputs, labels)
            total_loss = total_loss + loss.item()
            
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            y_true.extend(labels.cpu().numpy().tolist()) 
            y_pred.extend(predicted.cpu().numpy().tolist()) 
            
    avg_loss = total_loss/len(dataloader)
    print("MCC : {}".format(matthews_corrcoef(y_true, y_pred)))
    print("Classification Report")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix")
    print(confusion_matrix(y_true, y_pred))
    return avg_loss

#-----------------------------------------

#Load GLOVE embeddings
#glove = vocab.GloVe(name='42B', dim=300, cache='.vector_cache')
glove = vocab.GloVe(name='42B', dim=300)
print('Loaded {} words from Glove'.format(len(glove.itos)))

#Build domain dictionary
domain_dict = {}
for i, k in enumerate(domain_list):
    domain_dict[k] = i
print(domain_dict)
print("domain_dict : {}".format(domain_dict))

#Build slot dictionary
slot_dict = {}
slot_rev_dict = {}
for i, k in enumerate(slot_detail):
    slot_dict[slot_detail[k]] = i
    slot_rev_dict[i] = slot_detail[k]
print("slot_dict : {}".format(slot_dict))
print("slot_rev_dict : {}".format(slot_rev_dict))

#Loading Glove embeddings for slot
matrix_len = len(slot_dict)
weights_matrix = np.zeros((matrix_len, 300))
words_not_found = 0
for i in slot_rev_dict:    
    try: 
        weights_matrix[i] = get_word(slot_rev_dict[i])
    except KeyError:
        words_not_found += 1
        print("{} not found".format(slot_rev_dict[i]))
        weights_matrix[i] = np.random.normal(scale=0.6, size=(300, ))
print("#Words not found : {}".format(words_not_found))

#Loading Glove embeddings for domain
matrix_len = len(domain_list)
domain_matrix = np.zeros((matrix_len, 300))
domain_not_found = 0
for i in range(len(domain_list)):
    try: 
        domain_matrix[i] = get_word(domain_list[i])
    except KeyError:
        domain_not_found += 1
        print("{} not found".format(domain_list[i]))
        domain_matrix[i] = np.random.normal(scale=0.6, size=(300, ))
print("Shape of domain matrix: {}".format(domain_matrix.shape))
print("#Domain not found : {}".format(domain_not_found))

#-----------------------------------------

#Load data
file_path = os.path.join(in_dir, 'train_slot_act.tsv')
train_df = pd.read_csv(file_path, sep='\t')
print("Shape of Training data : {}".format(train_df.shape))

file_path = os.path.join(in_dir, 'test_slot_act.tsv')
test_df = pd.read_csv(file_path, sep='\t')
print("Shape of Test data : {}".format(test_df.shape))

file_path = os.path.join(in_dir, 'dev_slot_act.tsv')
valid_df = pd.read_csv(file_path, sep='\t')
print("Shape of Valid data : {}".format(valid_df.shape))

num_labels = len(train_df[LABEL_COLUMN].unique())
print("Number of labels : {}".format(num_labels))

# Set class weights to handle imbalanced class ratios (if required)
class_weights = torch.ones(num_labels)
print("class weights : {}".format(class_weights))

MAX_LEN = 200
print("Max length final : {}".format(MAX_LEN))

batch_size = 32
print("Batch size : {}".format(batch_size))

print("Loading Train data")
train_dataloader = prepare_data(train_df, MAX_LEN, batch_size, slot_dict, domain_dict)
print("Loading Test data")
test_dataloader = prepare_data(test_df, MAX_LEN, batch_size, slot_dict, domain_dict)
print("Loading Validation data")
valid_dataloader = prepare_data(valid_df, MAX_LEN, batch_size, slot_dict, domain_dict)
print("Data load completed")

print("Size of Train loader : {}".format(len(train_dataloader)))
print("Size of Test loader : {}".format(len(test_dataloader)))
print("Size of Valid loader : {}".format(len(valid_dataloader)))

model = Model(weights_matrix, domain_matrix, num_labels)
model.to(device)

#-----------------------------------------

print('Starting Training ...')
clip = 2.0
num_epoch = 4
best_valid_loss = 9999
best_test_loss = 9999
best_train_loss = 0
best_model = 0
model_copy = type(model)(weights_matrix, domain_matrix, num_labels)

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = AdamW(model.parameters(),
                  lr = 2e-5,
                  eps = 1e-8
               )

total_steps = len(train_dataloader) * num_epoch
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps)

for epoch in range(num_epoch):
    model.train()
    print("Epoch {} --------------------------".format(epoch+1))
    running_loss = 0.0
    for i, batch in enumerate(train_dataloader):
        b_input_ids = batch[0].to(device)
        b_attn_mask = batch[1].to(device)
        b_slot_ids = batch[2].to(device)
        b_domain_ids = batch[3].to(device)
        b_labels = batch[4].to(device)

        optimizer.zero_grad()
        outputs = model(b_input_ids, b_attn_mask, b_slot_ids, b_domain_ids)
        loss = criterion(outputs, b_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        scheduler.step()

    print("Training Accuracy :-")
    train_loss = evaluate_metrics(train_dataloader, model)
    print("Validation Accuracy :-")
    valid_loss = evaluate_metrics(valid_dataloader, model)
    print("Test Accuracy :-")
    test_loss = evaluate_metrics(test_dataloader, model)
    print("Epoch {} : Train loss = {} : Valid loss = {} : Test loss = {}".format(epoch + 1, train_loss, valid_loss, test_loss))
    if(valid_loss < best_valid_loss):
        best_valid_loss = valid_loss
        best_test_loss = test_loss
        best_train_loss = train_loss
        best_model = epoch+1
        model_copy.load_state_dict(model.state_dict())
        print("Model {} copied".format(epoch+1))

print('Finished Training ...')
PATH = os.path.join(work_dir , 'slot_action_model.pt')
torch.save(model_copy.state_dict(), PATH)
model.to('cpu')
model_copy.to(device)
print("---Best model---")
print("Epoch {} : Train loss = {} : Validation Loss = {} : Test loss = {}".format(best_model, best_train_loss, best_valid_loss, best_test_loss))
print("Training Accuracy :-")
train_loss = evaluate_metrics(train_dataloader, model_copy)
print("Validation Accuracy :-")
valid_loss = evaluate_metrics(valid_dataloader, model_copy)
print("Test Accuracy :-")
test_loss = evaluate_metrics(test_dataloader, model_copy)
print("Verifying Epoch {} : Train loss = {} : Validation Loss = {} : Test loss = {}".format(best_model, train_loss, valid_loss, test_loss))
print("done")

#-----------------------------------------