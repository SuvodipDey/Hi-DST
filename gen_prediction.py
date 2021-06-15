"""
Description :
Generate DST prediction using Hi-DST.

Run Command:
python gen_prediction.py -in=<path of multiWOZ data> -out=<output dir> -key=<any unique key to identify result>
"""
#--------------------------------------------
import math
import time
import datetime
import random
import argparse
import os
import six, re
import json
import shutil
import pandas as pd
import numpy as np
import torch
import torchtext.vocab as vocab
import transformers
from transformers import BertTokenizer, BertModel, BertForQuestionAnswering
import torch.nn as nn
from model_class import SwitchModel, DomainModel, SlotActionModel
analyze = True
#--------------------------------------------

default_path = os.path.join('data', 'mwz2.1')

parser = argparse.ArgumentParser()
parser.add_argument('-in','--in', help='Name of the input directory containing the input files.', required=False, default=default_path)
parser.add_argument('-out','--out', help='path of the output directiory', required=True)
parser.add_argument('-key','--key', help='model key', required=True)
parser.add_argument('-switch_path','--switch_path', help='path of domain change prediction model', required=True)
parser.add_argument('-domain_path','--domain_path', help='path of domain prediction model', required=True)
parser.add_argument('-slot_act_path','--slot_act_path', help='path of slot action prediction model', required=True)
parser.add_argument('-slot_val_path','--slot_val_path', help='path of slot value prediction model', required=True)

args = vars(parser.parse_args())
in_dir = args['in']
out_dir = args['out']
model_key = args['key']
switch_path = args['switch_path']
domain_path = args['domain_path']
slot_act_path = args['slot_act_path']
slot_val_path = args['slot_val_path']

print("Path of input directory : {}".format(in_dir))
print("Path of output directory : {}".format(out_dir))
print("Path of domain change model : {}".format(switch_path))
print("Path of domain model : {}".format(domain_path))
print("Path of slot action model : {}".format(slot_act_path))
print("Path of slot value model : {}".format(slot_val_path))
print("Model key : {}".format(model_key))

if(not os.path.isdir(in_dir)):
    print("Input directory {} does not exist.".format(in_dir))
    exit(0)
    
if(not os.path.isdir(out_dir)):
    print("Creating output directiory : {}".format(out_dir))
    os.mkdir(out_dir) 

f_str = "log_test_{}.json".format(model_key)
filename = os.path.join(out_dir, f_str)
print("Output filename : {}".format(filename))

#--------------------------------------------

domain_list = ['police', 'restaurant', 'hotel', 'taxi', 'attraction', 'train', 'hospital']

slot_detail = {'Type': 'type', 'Price': 'price', 'Parking': 'parking', 'Stay': 'stay', 'Day': 'day', 
               'People': 'people', 'Post': 'post', 'Addr': 'address', 'Dest': 'destination', 'Arrive': 'arrive', 
               'Depart': 'departure', 'Internet': 'internet', 'Stars': 'stars', 'Phone': 'phone', 'Area': 'area', 
               'Leave': 'leave', 'Time': 'time', 'Ticket': 'ticket', 'Ref': 'reference', 'Food': 'food', 
               'Name': 'name', 'Department': 'department', 'Fee': 'fee', 'Id': 'id', 'Car': 'car'}

meta = {'attraction': {'name', 'type', 'area'}, 
        'hotel': {'name', 'type', 'parking', 'area', 'day', 'stay', 'internet', 'people', 'stars', 'price'}, 
        'restaurant': {'name', 'food', 'area', 'day', 'time', 'people', 'price'}, 
        'taxi': {'arrive', 'departure', 'leave', 'destination'}, 
        'train': {'arrive', 'day', 'leave', 'destination', 'departure', 'people'}
       }

question_dict = {}
question_dict['type'] = 'What is the type of domain?'
question_dict['price'] = 'What is the price range of the domain?'
question_dict['stay'] = 'How many days to stay in the domain?'
question_dict['day'] = 'What day of the week to book the domain?'
question_dict['people'] = 'A domain booking for how many people?'
question_dict['destination'] = 'What is the destination of the domain?'
question_dict['arrive'] = 'What is the arrival time of the domain?'
question_dict['departure'] = 'What is the departure location of the domain?'
question_dict['stars'] = 'What is the star rating of the domain?'
question_dict['area'] = 'What is the area or location of the domain?'
question_dict['leave'] = 'What is the leaving time of the domain?'
question_dict['food'] = 'What is the food type of the domain?'
question_dict['name'] = 'What is the name of the domain?'
question_dict['time'] = 'What is the booking time of the domain?'

hotel_type = ["hotel", "guesthouse", "guest house", "lodge"]
attraction_type = ['sport', 'entertainment', 'cinema', 'museum', 'theatre', 'church', 'boat', 'architecture', 'college', 'park', 'theater', 'camboats', 'concert', 'park', 'concert', 'hiking', 'historical', 'gallery', 'nightclub', 'special', 'swimming', 'gastropub', 'outdoor', 'pool', 'pub', 'club', 'swim', 'hall', 'movie']

dataset_config = os.path.join('trippy_label_variant', 'multiwoz21.json')
with open(dataset_config, "r", encoding='utf-8') as f:
    raw_config = json.load(f)
class_types = raw_config['class_types']
slot_list = raw_config['slots']
label_maps = raw_config['label_maps']

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#--------------------------------------------

domain_dict = {}
for i, k in enumerate(domain_list):
    domain_dict[k] = i
print("domain_dict : {}".format(domain_dict))

slot_dict = {}
slot_rev_dict = {}
for i, k in enumerate(slot_detail):
    slot_dict[slot_detail[k]] = i
    slot_rev_dict[i] = slot_detail[k]
print("slot_dict : {}".format(slot_dict))
print("slot_rev_dict : {}".format(slot_rev_dict))

#Loading Glove embeddings   
glove = vocab.GloVe(name='42B', dim=300, cache='.vector_cache')
print('Loaded {} words from Glove'.format(len(glove.itos)))

def get_word(word):
    return glove.vectors[glove.stoi[word]]

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

#--------------------------------------------

#Loading domain switch model
switch_model_path = os.path.join(switch_path, 'switch_model.pt')
switch_model = SwitchModel(3)
switch_model.load_state_dict(torch.load(switch_model_path))
switch_model.eval()
print("Switch Model Loaded")

#Loading domain prediction model
domain_model_path = os.path.join(domain_path, 'domain_model.pt')
domain_model = DomainModel(domain_matrix, 2)
domain_model.load_state_dict(torch.load(domain_model_path))
domain_model.eval()
print("Domain Model Loaded")

#Loading slot action model
slot_action_path = os.path.join(slot_act_path, 'slot_action_model.pt')
slot_act_model = SlotActionModel(weights_matrix, domain_matrix, 10)
slot_act_model.load_state_dict(torch.load(slot_action_path))
slot_act_model.eval()
print("Slot Action Model Loaded")

#Loading slot value model
slot_value_model_path = os.path.join(slot_val_path, 'slot_value_model.pt')
slot_value_model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
slot_value_model.load_state_dict(torch.load(slot_value_model_path))
slot_value_model.eval()
print("Slot Value Model Loaded")

#--------------------------------------------

def load_json(data_file):
    if os.path.isfile(data_file):
        with open(data_file, 'r') as read_file:
            data = json.load(read_file)
            return data

def load_list_file(list_file):
    with open(list_file, 'r') as read_file:
        dialog_id_list = read_file.readlines()
        dialog_id_list = [l.strip('\n') for l in dialog_id_list]
        return dialog_id_list
    return

def cleanBeliefState(belief_state):
    bs = {}
    for k,v in belief_state.items():
        if (v!='none'):
            bs[k] = v
    return bs

def cleanDialogAct(dialog_act):
    dst = {}
    for k in dialog_act:
        if(dialog_act[k] == "do n't care" or dialog_act[k]=="do nt care"):
            dst[k] = "dontcare"
        else:
            dst[k] = dialog_act[k]
    return dst

def correctSlotName(slot):
    if(slot=="arriveby"):
        return "arrive"
    elif(slot=="leaveat"):
        return "leave"
    elif(slot=="pricerange"):
         return "price"
    else:
        return slot
    
def getBeliefState(belief_state):
    bs = {}
    for l in range(len(belief_state)):
        for sv in belief_state[l]['slots']:
            b_key = sv[0]
            if("-book" in b_key):
                b_key_l = b_key.split(" ")
                b_key = b_key_l[0].split("-")[0]+"-"+correctSlotName(b_key_l[1])
            else:
                b_key = b_key.split("-")[0]+"-"+correctSlotName(b_key.split("-")[1])
            if (sv[1]!='none'):
                bs[b_key] = sv[1]
                
    return cleanBeliefState(bs)
    
def getTurnLabel(tl):
    turn_label = {}
    for l in range(len(tl)):
        sv = tl[l]
        b_key = sv[0]
        if("-book" in b_key):
            b_key_l = b_key.split(" ")
            b_key = b_key_l[0].split("-")[0]+"-"+correctSlotName(b_key_l[1])
        else:
            b_key = b_key.split("-")[0]+"-"+correctSlotName(b_key.split("-")[1])
        turn_label[b_key] = sv[1]
        
    return cleanBeliefState(turn_label)

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")

def normalize_time(text):
    text = re.sub("(\d{1})(a\.?m\.?|p\.?m\.?)", r"\1 \2", text) # am/pm without space
    text = re.sub("(^| )(\d{1,2}) (a\.?m\.?|p\.?m\.?)", r"\1\2:00 \3", text) # am/pm short to long form
    text = re.sub("(^| )(at|from|by|until|after) ?(\d{1,2}) ?(\d{2})([^0-9]|$)", r"\1\2 \3:\4\5", text) # Missing separator
    text = re.sub("(^| )(\d{2})[;.,](\d{2})", r"\1\2:\3", text) # Wrong separator
    
    #text = re.sub("(^| )(\d{1})[;.,](\d{2})", r" \2:\3", text) # Wrong separator
    text = re.sub("(^| )(\d{2}):(\d{2})/", r"\1\2:\3", text) # Wrong separator
    text = re.sub("(^| )(\d{1}) (\d{2})", r"\1\2:\3", text) # Wrong separator
    text = re.sub("(^| )(\d{2}):!(\d{1})", r"\1\2:1\3", text) # Wrong format
    
    text = re.sub("(^| )(at|from|by|until|after) ?(\d{1,2})([;., ]|$)", r"\1\2 \3:00\4", text) # normalize simple full hour time
    text = re.sub("(^| )(\d{1}:\d{2})", r"\g<1>0\2", text) # Add missing leading 0
    # Map 12 hour times to 24 hour times
    text = re.sub("(\d{2})(:\d{2}) ?p\.?m\.?", lambda x: str(int(x.groups()[0]) + 12 if int(x.groups()[0]) < 12 else int(x.groups()[0])) + x.groups()[1], text)
    text = re.sub("(^| )24:(\d{2})", r"\g<1>00:\2", text) # Correct times that use 24 as hour
    return text


def normalize_text(utt):
    text = convert_to_unicode(utt)
    text = text.lower()
    text = normalize_time(text)
    text = re.sub("n't", " not", text)
    text = re.sub("(^| )zero(-| )star([s.,? ]|$)", r"\g<1>0 star\3", text)
    text = re.sub("(^| )one(-| )star([s.,? ]|$)", r"\g<1>1 star\3", text)
    text = re.sub("(^| )two(-| )star([s.,? ]|$)", r"\g<1>2 star\3", text)
    text = re.sub("(^| )three(-| )star([s.,? ]|$)", r"\g<1>3 star\3", text)
    text = re.sub("(^| )four(-| )star([s.,? ]|$)", r"\g<1>4 star\3", text)
    text = re.sub("(^| )five(-| )star([s.,? ]|$)", r"\g<1>5 star\3", text)
    text = re.sub("(^| )(\d{1})-star([s.,? ]|$)", r"\1\2 star\3", text)
    text = re.sub("archaelogy", "archaeology", text) # Systematic typo
    text = re.sub("mutliple", "multiple", text) # Systematic typo
    text = re.sub("(^| )b ?& ?b([.,? ]|$)", r"\1bed and breakfast\2", text) # Normalization
    text = re.sub("bed & breakfast", "bed and breakfast", text) # Normalization
    return text

def getQuestion(dom, slot, is_ref):
    q = ""
    if(is_ref):
        q = "What is the reference point of {} {}?".format(dom, slot)
    else:
        q = question_dict[slot]
        q = q.replace("domain", dom)
    return q.lower()

def getSpanDict(i, log):
    span_dict = {}
    if(i<0):
        return span_dict
    
    t = log[i]
    span_info_len = 0
    if('span_info' in t.keys()):
        span_info_len = len(t['span_info'])
    for idx in range(span_info_len):
        dom = t['span_info'][idx][0].split("-")[0].lower()
        if t['span_info'][idx][1] in slot_detail:
            sl = slot_detail[t['span_info'][idx][1]]
            span_key = dom+"-"+sl
            if(span_key not in span_dict):
                v = t['span_info'][idx][2].lower()
                span_value = [v, t['span_info'][idx][3], t['span_info'][idx][4]]
                span_dict[span_key] = span_value
            else:
                v = "{}$${}".format(span_dict[span_key][0], t['span_info'][idx][2].lower())
                start_idx = "{}$${}".format(span_dict[span_key][1], t['span_info'][idx][3])
                end_idx = "{}$${}".format(span_dict[span_key][2], t['span_info'][idx][4])
                span_value = [v, start_idx, end_idx]
                span_dict[span_key] = span_value
    return span_dict

#--------------------------------------------

def getProbability(output):
    prob = output[0].detach().numpy()
    prob = np.exp(prob)
    sm = np.sum(prob)
    prob = prob/sm
    p = [round(x,4) for x in prob]
    return p
    
def predictSwitch(sys, usr):
    max_len = 200
    encoding = tokenizer.encode_plus(sys, usr, add_special_tokens = True, 
                                     padding='max_length', 
                                     truncation=True,
                                     max_length = max_len,
                                     return_attention_mask = True)
    
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
    output = switch_model(torch.tensor([input_ids]), torch.tensor([attention_mask]))
    pred = torch.argmax(output).item()
    prob = getProbability(output)
    return pred, prob

def predictDomain(sys, usr, domain_id):
    max_len = 200
    encoding = tokenizer.encode_plus(sys, usr, add_special_tokens = True, 
                                     padding='max_length', 
                                     truncation=True,
                                     max_length = max_len,
                                     return_attention_mask = True)
    
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
    output = domain_model(torch.tensor([input_ids]), torch.tensor([attention_mask]), torch.tensor([domain_id]))
    pred = torch.argmax(output).item()
    prob = getProbability(output)
    return pred, prob[pred]

def predictDomainList(sys, usr):
    pred_dom = []
    dom_list = ['restaurant', 'hotel', 'taxi', 'attraction', 'train']
    for dom in dom_list:
        dom_id = domain_dict[dom]
        pred, prob = predictDomain(sys, usr, dom_id)
        if(pred==1):
            pred_dom.append((dom, prob))
        pred_dom.sort(key=lambda tup: tup[1], reverse=True)
    return pred_dom

def predictSlotAction(sys, usr, slot_id, domain_id):
    max_len = 200
    encoding = tokenizer.encode_plus(sys, usr, add_special_tokens = True, 
                                 padding='max_length', 
                                 truncation=True,
                                 max_length = max_len,
                                 return_attention_mask = True)
        
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
    output = slot_act_model(torch.tensor([input_ids]), torch.tensor([attention_mask]),  torch.tensor([slot_id]), torch.tensor([domain_id]))
    
    pred = torch.argmax(output).item()
    prob = getProbability(output)
    
    return pred, prob

def extractAnswer(question, text):
    max_len = 100
    encoding = tokenizer.encode_plus(
                            question, text,                 
                            add_special_tokens = True, 
                            max_length = max_len,     
                            padding='max_length', 
                            truncation=True)
    
    input_ids, attn_masks, token_type_ids = encoding["input_ids"], encoding["attention_mask"], encoding["token_type_ids"]
    outputs = slot_value_model(torch.tensor([input_ids]),
                                               attention_mask=torch.tensor([attn_masks]), 
                                               token_type_ids=torch.tensor([token_type_ids]))
    
    idx1 = torch.argmax(outputs.start_logits, dim=1).item()
    idx2 = torch.argmax(outputs.end_logits, dim=1).item()
    
    # Check if answer in extracted from the question
    q_len = len(tokenizer.encode_plus(question)["input_ids"])
    if(idx1<q_len or idx2<q_len or idx2<idx1):
        answer = "none"
    else:
        lst = []
        for i in range(idx1, idx2+1):
            lst.append(input_ids[i])
        answer = tokenizer.decode(lst, clean_up_tokenization_spaces=True)
    return answer

def getReference(usr, domain, slot, pred_slots, informed_slots):
    slot_value = "none"
    qs = getQuestion(domain, slot, True)
    ref_dom = extractAnswer(qs, usr)
    ref_slot = slot
    
    sl_name = ["destination", "departure"]
    sl_time = ["arrive", "leave"]
    dom_travel = ["taxi", "train"]
    
    dom_list = []
    for slot_key in pred_slots:
        dom = slot_key.split("-")[0]
        if(dom!=domain and dom not in dom_list):
            dom_list.append(dom)
            
    if (ref_dom=="none"):
        if(len(dom_list)==1):
            ref_dom = dom_list[0]
    
    if (ref_dom!="none"):
        for dom in domain_list:
            if (ref_dom==dom or dom in ref_dom):
                ref_dom = dom
                break
            else:
                if(dom=="hotel"):
                    for v in hotel_type:
                        if (v in ref_dom):
                            ref_dom = dom
                elif(dom=="attraction"):
                    for v in attraction_type:
                        if (v in ref_dom):
                            ref_dom = dom

        if(ref_dom not in dom_travel and slot in sl_name):
            ref_slot = "name"
        if(ref_dom not in dom_travel and slot in sl_time):
            ref_slot = "time"
        slot_key = ref_dom + "-" + ref_slot
        if(log_print):
            print("Ref of {} : {}".format(domain+"-"+slot, slot_key))
        if slot_key in pred_slots:
            slot_value = pred_slots[slot_key]   
        if(slot_value=="none" and slot_key in informed_slots):
            temp_val = informed_slots[slot_key]
            if("$$" in temp_val):
                slot_value = temp_val.split("$$")[0]
            
    return slot_value

def predictSlotValue(sys, usr, domain, slot, slot_action, pred_slots, informed_slots):
    slot_value = 'none'
    if(slot_action==1): #Request
        slot_value = '?' 
    elif(slot_action==2): #Dont care
        slot_value = 'dontcare' 
    elif(slot_action==3): #Yes
        slot_value = 'yes'
    elif(slot_action==4): #No
        slot_value = 'no'
    elif(slot_action==5): #Singular
        if(slot=='people'):
            slot_value = '1'
    elif(slot_action==6): #Type
        if(domain=='hotel' and slot=='type'):
            slot_value = 'hotel'
    elif(slot_action==7): #Extract from user
        qs = getQuestion(domain, slot, False)
        text = usr
        slot_value = extractAnswer(qs, text)
    elif(slot_action==8): #Extract from sys
        slot_key = domain+"-"+ slot
        text = normalize_text(sys)
        if (slot_key in informed_slots and informed_slots[slot_key]!="none"):
            temp_val = informed_slots[slot_key]
            if("$$" not in temp_val):
                slot_value = temp_val
            else:
                slot_value = temp_val.split("$$")[0]
        else:
            qs = getQuestion(domain, slot, False)
            slot_value = extractAnswer(qs, text)
    elif(slot_action==9): #Copy from previous states       
        slot_value = getReference(usr, domain, slot, pred_slots, informed_slots)
    return slot_value

def isUnseen(sl_key, slot_value, pred_bs):
    f = True
    if (sl_key in pred_bs): 
        if(slot_value==pred_bs[sl_key]):
            f=False
        else:
            v = pred_bs[sl_key]
            if v in label_maps:
                for value_label_variant in label_maps[v]:
                    if slot_value == value_label_variant:
                        f = False
                        break
            
            if (f and slot_value in label_maps):
                for value_label_variant in label_maps[slot_value]:
                    if v == value_label_variant:
                        f = False
                        break
    return f

def getStringList(l):
    return [str(x) for x in l]

def updateReferenceTravel(usr, domain, pred_bs, informed_slots, pred_slots, pred_tl):

    sl_name = ["destination", "departure"]
    sl_time = ["arrive", "leave"]
    dom_travel = ["taxi", "train"]

    #print("Updating Ref : {}".format(usr))
    ref_slots = []
    for sl_key in pred_slots:
        if(pred_slots[sl_key][0] == 9):
            ref_slots.append(sl_key)
    #print("ref_slots : {}".format(ref_slots))
    
    dom_list = []
    for slot_key in pred_bs:
        dom = slot_key.split("-")[0]
        if(dom not in dom_travel and dom not in dom_list):
            dom_list.append(dom)
    #print("ref_domains : {}".format(dom_list))
    #print("prev_bs : {}".format(pred_bs))
    #print("pred_tl : {}".format(pred_tl))
    
    if(len(dom_list)==2):
        d_dep = dom_list[0]
        d_dest = dom_list[1]
        dep_key = domain+"-departure"
        dest_key = domain+"-destination"
        dep_ref = d_dep+"-name"
        dest_ref = d_dest+"-name"
        leave_key = domain+"-leave"
        arrive_key = domain+"-arrive"
        time_ref = "restaurant-time"
        
        if(dep_key not in pred_tl and dest_key not in pred_tl):
            if (dep_key in ref_slots):
                f = False
                if(dep_ref in pred_bs):
                    pred_tl[dep_key] = pred_bs[dep_ref]
                    f = True
                elif(dep_ref in informed_slots):
                    f = True
                    pred_tl[dep_key] = informed_slots[dep_ref]
                if(f and "restaurant" in dep_ref and leave_key not in pred_tl):
                    if(time_ref in pred_bs):
                        pred_tl[leave_key] = pred_bs[time_ref]
                    
            if (dest_key in ref_slots):
                f = False
                if(dest_ref in pred_bs):
                    pred_tl[dest_key] = pred_bs[dest_ref]
                    f = True
                elif(dest_ref in informed_slots):
                    pred_tl[dest_key] = informed_slots[dest_ref]
                    f = True
                if(f and "restaurant" in dest_ref and arrive_key not in pred_tl):
                    if(time_ref in pred_bs):
                        pred_tl[arrive_key] = pred_bs[time_ref]
        else:
            v_ref_0 = "none"
            v_ref_1 = "none"
            s_key = dom_list[0]+"-name"
            if(s_key in pred_bs):
                v_ref_0 = pred_bs[s_key]
            elif(s_key in informed_slots):
                v_ref_0 = informed_slots[s_key]
                
            s_key = dom_list[1]+"-name"
            if(s_key in pred_bs):
                v_ref_1 = pred_bs[s_key]
            elif(s_key in informed_slots):
                v_ref_1 = informed_slots[s_key]
            
            if(dep_key in pred_tl):
                if(dest_key in pred_bs):
                    v = pred_bs[dest_key]
                    if(v == pred_tl[dep_key]):
                        if(log_print):
                            print("Need to change the value of {}".format(dest_key))
                        if (v==v_ref_0 and v_ref_1!="none"):
                            pred_tl[dest_key] = v_ref_1
                        elif(v==v_ref_1 and v_ref_0!="none"):
                            pred_tl[dest_key] = v_ref_0
                else:
                    if(dest_key in ref_slots):
                        if(log_print):
                            print("Need to set the value of {}".format(dest_key))
                        v = pred_tl[dep_key]
                        if (v==v_ref_0 and v_ref_1!="none"):
                            pred_tl[dest_key] = v_ref_1
                        elif(v==v_ref_1 and v_ref_0!="none"):
                            pred_tl[dest_key] = v_ref_0
                        
            else:
                if(dep_key in pred_bs):
                    v = pred_bs[dep_key]
                    if(v == pred_tl[dest_key]):
                        if(log_print):
                            print("Need to change the value of {}".format(dep_key))
                        if (v==v_ref_0 and v_ref_1!="none"):
                            pred_tl[dep_key] = v_ref_1
                        elif(v==v_ref_1 and v_ref_0!="none"):
                            pred_tl[dep_key] = v_ref_0
                else:
                    if(dep_key in ref_slots):
                        if(log_print):
                            print("Need to set the value of {}".format(dep_key))
                        v = pred_tl[dest_key]
                        if (v==v_ref_0 and v_ref_1!="none"):
                            pred_tl[dep_key] = v_ref_1
                        elif(v==v_ref_1 and v_ref_0!="none"):
                            pred_tl[dep_key] = v_ref_0
                    
    
#--------------------------------------------

def getPrediction(k, d, dials):
    pred_log = {}
    dials_log = dials['dialogue']
    data_log = d['log']
    
    sys = " "
    switch_output = 1
    switch_prob = [0.0, 1.0, 0.0]
    current_domain = {}
    pred_bs = {}
    pred_bs_prev = {}
    informed_slots = {}
    
    for t in dials_log:
        i = t['turn_idx']
        idx = 2*i
        
        usr = data_log[idx]['text'].strip().lower()
        usr_norm = normalize_text(usr)
        
        span_dict_sys = {}
        if(idx>0):
            sys = data_log[idx-1]['text'].strip().lower()
            span_dict_sys = getSpanDict(idx-1, data_log)
        
        bs = getBeliefState(t['belief_state'])
        tl = getTurnLabel(t['turn_label'])
        for slot in span_dict_sys:
            informed_slots[slot] = span_dict_sys[slot][0]
        
        if(analyze):
            print("Turn : {}".format(i))
            print("Sys : {}".format(sys))
            print("Usr : {}".format(usr_norm))
        
        if(i>0):
            switch_output, switch_prob = predictSwitch(sys, usr_norm)
        
        #Run domain prediction when required
        if(len(current_domain)==0 or switch_output==1 or len(current_domain)>1):   
            p_domain = predictDomainList(sys, usr_norm)
            if(len(p_domain)>0):
                if(i==0):
                    current_domain[p_domain[0][0]] = str(p_domain[0][1])
                else:
                    current_domain = {}
                    if(len(p_domain)==1):
                        current_domain[p_domain[0][0]] = str(p_domain[0][1])
                    else:
                        for p_dom in p_domain:
                            current_domain[p_dom[0]] = str(p_dom[1])
                        
        if(switch_output==2):
            current_domain = {}
            
        pred_slots = {}
        pred_tl = {}
        if(switch_output<2):
            for dom in current_domain:
                slot_set = {}
                if dom in meta:
                    slot_set = meta[dom]
                
                for slot in slot_set:
                    slot_act, slot_act_prob  = predictSlotAction(sys, usr_norm, slot_dict[slot], domain_dict[dom])
                    sl_key = dom+"-"+slot
                    if(log_print):
                        print("Slot act of {}-{} : {} with {}".format(dom, slot, slot_act, slot_act_prob))
                    pred_slots[sl_key] = [slot_act, getStringList(slot_act_prob)]
                    if (slot_act>1):
                        #if(log_print):
                        #    print("Slot act of {}-{} : {} with {}".format(dom, slot, slot_act, slot_act_prob))
                        slot_value = predictSlotValue(sys, usr_norm, dom, slot, slot_act, pred_bs_prev, informed_slots)
                        if(slot_value!="none"):
                            pred_tl[sl_key] = slot_value
                
                if(dom=="taxi"):
                    updateReferenceTravel(usr_norm, dom, pred_bs_prev, informed_slots, pred_slots, pred_tl)
                                
        for sl_key in pred_tl:
            if(isUnseen(sl_key, pred_tl[sl_key], pred_bs_prev)):
                pred_bs[sl_key] = pred_tl[sl_key]
        
        if(analyze):
            print("Switch output : {} - {}".format(switch_output, switch_prob))
            print("Current domains : {}".format(current_domain))
            print("Current slots : {}".format(pred_slots))
            
        if(analyze):
            print("GT TL : {}".format(tl))
            print("PR TL : {}".format(pred_tl))
            print("GT BS : {}".format(bs))
            print("PR BS : {}".format(pred_bs))
            print("------------")
                
        pred_log[i] = {}
        pred_log[i]['a_sys'] = sys
        pred_log[i]['a_usr'] = usr
        pred_log[i]['a_usr_norm'] = usr_norm
        pred_log[i]['switch'] = [switch_output, getStringList(switch_prob)]
        pred_log[i]['domains'] = current_domain.copy()
        pred_log[i]['slots'] = pred_slots.copy()
        pred_log[i]['gt_turn'] = tl.copy()
        pred_log[i]['pr_turn'] = pred_tl.copy()
        pred_log[i]['gt'] = bs.copy()
        pred_log[i]['pr'] = pred_bs.copy()
        
        pred_bs_prev = pred_bs.copy()
            
    return pred_log        
    

#--------------------------------------------

#Load raw data
dialog_data_file = os.path.join(in_dir, 'data.json')
dialog_data = load_json(dialog_data_file)
dialog_id_list = list(set(dialog_data.keys()))
test_list_file = os.path.join(in_dir, 'testListFile.txt')
test_id_list = load_list_file(test_list_file)
print('# of test dialogs :', len(test_id_list))
test_data = [(k,v) for k, v in dialog_data.items() if k in test_id_list]
assert(len(test_data) == len(test_id_list))

#Load test dials data
dials_path = os.path.join(in_dir, "test_dials.json")
data = load_json(dials_path)
dials_data = {}
for i,d in enumerate(data):
    dials_data[d['dialogue_idx']] = d
print('# of test dials dialogs :', len(dials_data))

analyze=False # Set True to analyze a single prediction
log_print = False
result = {}
if(analyze):
    #log_print = True #Set True to print more details
    # Set dialogue id to be analysed
    idx = 'PMUL2437.json'
    for k,d in test_data:
        if(k in dials_data and k==idx):
            print(k)
            pred_log = getPrediction(k, d, dials_data[k])
            result[k] = pred_log
            break
    filename = os.path.join(out_dir, "unit_test.json")
    print("Output filename : {}".format(filename))
else:
    j=0
    now = datetime.datetime.now()
    print("Starting evaluation of test data at {}".format(now.strftime("%Y-%m-%d %H:%M:%S")))
    for k,d in test_data:
        if k in dials_data:
            pred_log = getPrediction(k, d, dials_data[k])
            result[k] = pred_log
            j=j+1
            if(j%100==0):
                now = datetime.datetime.now()
                print("Iteration {} completed at {}".format(j, now.strftime("%Y-%m-%d %H:%M:%S")))

result_file = open(filename, "w")
result_file.write(json.dumps(result, indent=4, sort_keys=True))
result_file.close()
    
print("done")
#--------------------------------------------