"""
Description :
Compute joint accuracy

Run Command:
# python compute_accuracy.py -key=<unique of the prediction file> -path=<result directory>
"""
#--------------------------------------------

import os
import json
import pandas as pd
import argparse
import re
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('-key','--key', help='string to identify the slot request result file', required=True)
parser.add_argument('-path','--path', help='path of the result file', required=True)

args = vars(parser.parse_args())
result_key = args['key']
result_dir = args['path']

result_file = os.path.join(result_dir, "log_test_"+result_key+".json")
print("Path of the result : {}".format(result_file))
    
if not os.path.isfile(result_file):
    print ("Result file does not exist. Please provide correct key.")
    exit(0)
    
dataset_config = os.path.join('data', 'trippy_label_variant', 'multiwoz21.json')
with open(dataset_config, "r", encoding='utf-8') as f:
    raw_config = json.load(f)
class_types = raw_config['class_types']
slot_list = raw_config['slots']
label_maps = raw_config['label_maps']

time_slots = ["time", "leave", "arrive"]

#-----------------------------------

def loadJson(data_file):
    if os.path.isfile(data_file):
        with open(data_file, 'r') as read_file:
            data = json.load(read_file)
            return data

# Slot Accuracy Computation taken from TRADE model
def getSlotAcc(gt, pr):
    miss_gold = 0
    miss_slot = []
    for g in gt:
        if g not in pr:
            miss_gold += 1
            miss_slot.append(g)
        else:
            is_match = isMatch(gt[g], pr[g], g)
            if(not is_match):
                miss_gold += 1
                miss_slot.append(g)
            
    wrong_pred = 0
    for p in pr:
        if p not in gt and p not in miss_slot:
            wrong_pred += 1
    ACC_TOTAL = 30
    ACC = 30 - miss_gold - wrong_pred
    ACC = ACC / float(ACC_TOTAL)
    return ACC

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

def isMatch(v1, v2, key):
    is_match = False
    if(v1==v2 or v1 in v2 or v2 in v1):
        is_match = True
    else:
        v3 = re.sub("b and b","bed and breakfast", v1)
        v3 = re.sub("(^the | |-|'|\"|:)", "", v3)
        v4 = re.sub("b and b","bed and breakfast", v2)
        v4 = re.sub("(^the | |-|'|\"|:)", "", v4)
        if(v3==v4 or v3 in v4 or v4 in v3):
            is_match = True
        else:
            slot = key.split("-")[1]
            if (slot in time_slots):
                v3 = normalize_time(v1.lower())
                v4 = v2.replace(" ","")
                if(v3==v4):
                    is_match = True
                else:
                    try:
                        if(":" in v3 and ":" in v4):
                            t1 = datetime.strptime(v3, '%H:%M')
                            t2 = datetime.strptime(v4, '%H:%M')
                            t_diff = abs((t1 - t2).total_seconds() / 60.0)
                            if(t_diff<=15):
                                is_match = True
                    except:
                        print("{} {} {}".format(key, v1, v2))
            else:
                v1 = re.sub("^the ", "", v1)
                v2 = re.sub("^the ", "", v2)
                v2 = v2.replace(" - ","-")
                if v1 in label_maps:
                    for value_label_variant in label_maps[v1]:
                        if (v2 in value_label_variant or value_label_variant in v2):
                            is_match = True

                if(not is_match and v2 in label_maps):
                    for value_label_variant in label_maps[v2]:
                        if (v1 in value_label_variant or value_label_variant in v1):
                            is_match = True
    return is_match

def getMatch(gt, pr):
    if(len(gt)!=len(pr)):
        return 0
    if(len(gt)==0):
        return 1
    
    gt_keys = set()
    pr_keys = set()
    for key in gt:
        gt_keys.add(key)
    for key in pr:
        pr_keys.add(key)
    diff = gt_keys.symmetric_difference(pr_keys)  
    if(len(diff)>0):
        return 0
        
    f=1
    for key in gt:
        v1 = gt[key]
        v2 = pr[key]
        is_match = isMatch(v1, v2, key)
        if (not is_match):
            f=0
            break
    return f

def modifyBS(bs):
    bs_modified = {}
    for slot_key in bs:
        if(True):
            v = bs[slot_key]
            v = v.replace(" '","'")
            bs_modified[slot_key] = v
    return bs_modified
    
def modifyTurnPrediction(pr, pred_slots):
    pr_turn = {}
    for slot_key in pr:
        slot_act = pred_slots[slot_key][0]
        slot = slot_key.split("-")[1]
        if(True):
            if (slot in time_slots):
                v = pr[slot_key]
                pr_turn[slot_key] = v
            else:
                v = pr[slot_key].replace(" '","'")
                pr_turn[slot_key] = v
    return pr_turn

def isUnseen(slot_key, slot_val, bs):
    f = True
    if (slot_key in bs): 
        if(slot_val==bs[slot_key]):
            f=False
        else:
            v = bs[slot_key]
            if v in label_maps:
                for value_label_variant in label_maps[v]:
                    if slot_val == value_label_variant:
                        f = False
                        break
            
            if (f and slot_val in label_maps):
                for value_label_variant in label_maps[slot_val]:
                    if v == value_label_variant:
                        f = False
                        break
    return f

def getTurnPrediction(bs, bs_prev):
    bs_turn = {}
    for slot_key in bs:
        slot_val = bs[slot_key]
        if(isUnseen(slot_key, slot_val, bs_prev)):
            bs_turn[slot_key] = slot_val
    return bs_turn
        
# Turn Accuracy
def computeTurnAccuracy(result_file):   
    dst_res = loadJson(result_file)
    total = 0
    c1 = 0
    c2 = 0
    c3=0
    sa_list = []
    
    analyze = False # Set true to analyze
    #analyze = True
    did='MUL2053.json' #Set a dialogue id to analyse
    for idx in dst_res:
        if(analyze):
            if(idx!=did):
                continue
        pr = {}
        gt_prev = {}
        pr_prev = {}
        for turn in dst_res[idx]:
            total+=1
            
            gt = modifyBS(dst_res[idx][turn]['gt'])
            gt_turn = getTurnPrediction(gt, gt_prev)
            pr_turn = modifyTurnPrediction(dst_res[idx][turn]['pr_turn'], dst_res[idx][turn]['slots'])
            for slot_key in pr_turn:
                pr[slot_key] = pr_turn[slot_key]
            pr_turn = getTurnPrediction(pr, pr_prev)

            m = getMatch(gt, pr)
            c1+=m
            
            sa = getSlotAcc(gt, pr)
            c3+=sa
            sa_list.append(sa)
            
            n = getMatch(gt_turn, pr_turn)
            c2+=n
            
            if(analyze):
                print(idx)
                print("Turn : {}".format(turn))
                print("Sys : {}".format(dst_res[idx][turn]['a_sys']))
                print("Usr : {}".format(dst_res[idx][turn]['a_usr_norm']))
                print("GT : {}".format(gt))
                print("PR : {}".format(pr))
                print("GT_Turn : {}".format(gt_turn))
                print("PR_Turn : {}".format(pr_turn))
                print("BS Match : {}".format(m))
                print("TL Match : {}".format(n))
                print("-----------")
            
            gt_prev = gt.copy()
            pr_prev = pr.copy()
        
    joint_acc = c1*100.0/total
    turn_acc= c2*100.0/total
    slot_acc = sum(sa_list)*100.0/len(sa_list)
    print("{} {} {}".format(total, c1, c2))
    print("Joint Accuracy : {}".format(joint_acc))
    print("Turn Accuracy : {}".format(turn_acc))
    print("Slot Accuracy : {}".format(slot_acc))
    
print("-----------")
computeTurnAccuracy(result_file)
print("done")