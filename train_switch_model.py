"""
Description :
Domain change prediction model.
Finetunes bert-base-uncased model on domain change prediction data.

Run Command:
python train_switch_model.py -in=<path of the input data> -path=<output dir> -src_file=<name of the python script>
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

print("Path of working directory : {}".format(work_dir))
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

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
LABEL_COLUMN = "label"

#-----------------------------------------

def prepare_data(df, max_len, batch_size):
    input_ids = []
    attention_masks = []

    for i in df.index:
        # Encode system and user utterance pair
        encoded_dict = tokenizer.encode_plus(
                            df['sys'][i].lower(), df['usr'][i].lower(),
                            add_special_tokens = True,
                            max_length = max_len,
                            padding='max_length', 
                            truncation=True,
                            return_attention_mask = True,
                            return_tensors = 'pt',
                       )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(df[LABEL_COLUMN])
    
    dataset = TensorDataset(input_ids, attention_masks, labels)
    dataloader = DataLoader(
            dataset,
            sampler = RandomSampler(dataset),
            batch_size = batch_size
        )
    return dataloader

class Model(nn.Module):
    def __init__(self, num_labels):
        super(Model, self).__init__()
        self.encode = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
        self.drop_out = nn.Dropout(0.3)
        self.l1 = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_masks):
        outputs = self.encode(input_ids, attention_masks)
        input1 = torch.mean(outputs[2][-2], dim=1)
        input1 = self.drop_out(input1)
        output1 = self.l1(input1)
        return output1

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
            labels = batch[2].to(device)
             
            outputs = model(b_input_ids, b_attn_mask)
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

#Load data
file_path = os.path.join(in_dir, 'train_switch.tsv')
train_df = pd.read_csv(file_path, sep='\t')
print("Shape of Training data : {}".format(train_df.shape))

file_path = os.path.join(in_dir, 'dev_switch.tsv')
valid_df = pd.read_csv(file_path, sep='\t')
print("Shape of Validation data : {}".format(valid_df.shape))

file_path = os.path.join(in_dir, 'test_switch.tsv')
test_df = pd.read_csv(file_path, sep='\t')
print("Shape of Test data : {}".format(test_df.shape))

num_labels = len(train_df[LABEL_COLUMN].unique())
print("Number of labels : {}".format(num_labels))

# Set class weights to handle imbalanced class ratios (if required)
class_weights = torch.ones(num_labels)
print("class weights : {}".format(class_weights))

MAX_LEN = 200
print("Max length : {}".format(MAX_LEN))

batch_size = 32
print("Batch size : {}".format(batch_size))

print("Loading Train data")
train_dataloader = prepare_data(train_df, MAX_LEN, batch_size)
print("Loading Test data")
test_dataloader = prepare_data(test_df, MAX_LEN, batch_size)
print("Loading Validation data")
valid_dataloader = prepare_data(valid_df, MAX_LEN, batch_size)

print("Size of Train loader : {}".format(len(train_dataloader)))
print("Size of Valid loader : {}".format(len(valid_dataloader)))
print("Size of Test loader : {}".format(len(test_dataloader)))

model = Model(num_labels)
model.to(device)

#-----------------------------------------

print('Starting Training ...')
clip = 2.0
num_epoch = 5
best_valid_loss = 9999
best_test_loss = 9999
best_train_loss = 0
best_model = 0
model_copy = type(model)(num_labels)

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
        b_labels = batch[2].to(device)

        optimizer.zero_grad()
        outputs = model(b_input_ids, b_attn_mask)
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
PATH = os.path.join(work_dir , 'switch_model.pt')
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