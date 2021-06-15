"""
Description :
Slot-value prediction model.
Finetunes pre-trained SQuAD model on slot-value prediction data.

Run Command:
python train_slot_value.py -in=<path of the input data> -path=<output dir> -src_file=<name of the python script>
"""
#--------------------------------------------

import torch
import random
import math
import time
import datetime
import pandas as pd
import argparse
import os
import shutil
import sys 
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForQuestionAnswering, AdamW
from transformers import get_linear_schedule_with_warmup

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

#-----------------------------------------

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def prepare_data(df, max_len, batch_size):
    input_ids = []
    attention_masks = []
    token_type_ids = []

    for index, row in df.iterrows():
        question = row['question'].lower() # Domain-slot pair as question
        text = row['text'].lower() # Utterance as context
        
        encoded_dict = tokenizer.encode_plus(
                            question, text,                 
                            add_special_tokens = True, 
                            max_length = max_len,     
                            padding='max_length', 
                            truncation=True,
                            return_tensors = 'pt', 
                       )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        token_type_ids.append(encoded_dict['token_type_ids'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    idx1 = torch.tensor(df.idx1)
    idx2 = torch.tensor(df.idx2)
    
    dataset = TensorDataset(input_ids, attention_masks, token_type_ids, idx1, idx2)
    dataloader = DataLoader(
            dataset,
            sampler = RandomSampler(dataset), 
            batch_size = batch_size
        )
    return dataloader

def predict(model, dl, max_len, batch_size):
    count = 0
    match = 0
    s_match = 0
    m=0
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dl:
            b_input_ids = batch[0].to(device)
            b_attn_masks = batch[1].to(device)
            b_token_ids = batch[2].to(device)
            b_idx1 = batch[3].to(device)
            b_idx2 = batch[4].to(device)

            outputs = model(b_input_ids, attention_mask=b_attn_masks, 
                            token_type_ids=b_token_ids, start_positions=b_idx1, end_positions=b_idx2)
            loss, start_scores, end_scores = outputs[:3]
  
            p_idx1 = torch.argmax(start_scores, dim=1)
            p_idx2 = torch.argmax(end_scores, dim=1)

            n = len(batch[0])
            c = 0
            w = 0
            for i in range(n):
                if(b_idx1[i].item()==p_idx1[i].item() and b_idx2[i].item()==p_idx2[i].item()):
                    c = c+1
                if(b_idx1[i].item()>=p_idx1[i].item() and b_idx2[i].item()<=p_idx2[i].item() and p_idx1[i].item()<=p_idx2[i].item()):
                    w = w+1
            match = match + c
            s_match = s_match + w
            count = count+n
            total_loss = total_loss + loss
            m = m+1
    avg_loss = total_loss/m
    print("Data Size : {}".format(count))
    print("Match : {}".format(match))
    print("Accuracy : {}".format(match/count))
    print("Sub Match : {}".format(s_match))
    print("Sub Accuracy : {}".format(s_match/count))
    print("Average Loss : {}".format(total_loss/m))
    return avg_loss
    
def train(model, max_len, batch_size, epochs):
    loss_values = []
    best_loss = 999999
    best_model = 0
    model_copy = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    optimizer = AdamW(model.parameters(),
                  lr = 2e-5,
                  eps = 1e-8
                )
    
    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dl) * epochs
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps)
    
    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()
        total_loss = 0
        model.train()

        for step, batch in enumerate(train_dl):
            b_input_ids = batch[0].to(device)
            b_attn_masks = batch[1].to(device)
            b_token_ids = batch[2].to(device)
            b_idx1 = batch[3].to(device)
            b_idx2 = batch[4].to(device)
            model.zero_grad()        

            outputs = model(b_input_ids, attention_mask=b_attn_masks, 
                            token_type_ids=b_token_ids, start_positions=b_idx1, end_positions=b_idx2)
            loss, start_scores, end_scores = outputs[:3]

            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_dl)            
        loss_values.append(avg_train_loss)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
        
        print("Running Prediction")
        print("Performance on Training set")
        avg_train_loss = predict(model, train_dl, max_len, batch_size)
        print("Performance on Test set")
        avg_test_loss = predict(model, test_dl, max_len, batch_size)
        print("Performance on Validation set")
        avg_valid_loss = predict(model, valid_dl, max_len, batch_size)
        print("Loss:-")
        print("Avg Train loss={} : Avg Valid Loss={} : Avg Test Loss : {}".format(avg_train_loss, avg_valid_loss, avg_test_loss))
        if(avg_valid_loss < best_loss):
            best_loss = avg_valid_loss
            best_model = epoch_i+1
            model_copy.load_state_dict(model.state_dict())
            print("Model {} copied".format(epoch_i+1))
        print("---------------------------------")
        
    print("Best model : {}".format(best_model))
    PATH = os.path.join(work_dir , 'slot_value_model.pt')
    torch.save(model_copy.state_dict(), PATH)
    model.to('cpu')
    model_copy.to(device)
    print("Performance on Training set")
    avg_train_loss = predict(model_copy, train_dl, max_len, batch_size)
    print("Performance on Test set")
    avg_test_loss = predict(model_copy, test_dl, max_len, batch_size)
    print("Performance on Validation set")
    avg_valid_loss = predict(model_copy, valid_dl, max_len, batch_size)
    print("Loss:-")
    print("Avg Train loss={} : Avg Test Loss={} : Avg Valid Loss : {}".format(avg_train_loss, avg_test_loss, avg_valid_loss))
    print("---------------------------------")
    print("done")

#-----------------------------------------

# Load data
file_path = os.path.join(in_dir, 'train_slot_value.tsv')
train_df = pd.read_csv(file_path, sep='\t')

file_path = os.path.join(in_dir, 'test_slot_value.tsv')
test_df = pd.read_csv(file_path, sep='\t')

file_path = os.path.join(in_dir, 'dev_slot_value.tsv')
valid_df = pd.read_csv(file_path, sep='\t')

print("Shape of train data : {}".format(train_df.shape))
print("Shape of test data : {}".format(test_df.shape))
print("Shape of valid data : {}".format(valid_df.shape))

max_len = 100
print("Max length : {}".format(max_len))

batch_size = 32
#batch_size = 16
print("Batch size : {}".format(batch_size))

train_dl = prepare_data(train_df, max_len, batch_size)
test_dl = prepare_data(test_df, max_len, batch_size)
valid_dl = prepare_data(valid_df, max_len, batch_size)

print("Size of Train loader : {}".format(len(train_dl)))
print("Size of Test loader : {}".format(len(test_dl)))
print("Size of Valid loader : {}".format(len(valid_dl)))

# Load pre-trained Squad model
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model.to(device)

epochs = 4
print("Starting training...")
print("---------------------------------")
train(model, max_len, batch_size, epochs)
print("---------------------------------")

#-----------------------------------------