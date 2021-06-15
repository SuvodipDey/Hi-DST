"""
Description :
Generates the train/test/validation data for slot action and slot value prediction model.

Run Command:
python create_slot_data.py -path=<path of input data> -out=<output directory>
"""
#--------------------------------------------
import os
import json
import argparse
import spacy
import pandas as pd
import random
import six, re
import torch
import traceback
from transformers import BertTokenizer

random.seed(1234)
default_path = os.path.join('data', 'mwz2.1')
default_out_path = "data"
default_mwz_ver = "2.1"

slot_detail = {'Type': 'type', 'Price': 'price', 'Parking': 'parking', 'Stay': 'stay', 'Day': 'day',
               'People': 'people', 'Post': 'post', 'Addr': 'address', 'Dest': 'destination', 'Arrive': 'arrive',
               'Depart': 'departure', 'Internet': 'internet', 'Stars': 'stars', 'Phone': 'phone', 'Area': 'area',
               'Leave': 'leave', 'Time': 'time', 'Ticket': 'ticket', 'Ref': 'reference', 'Food': 'food',
               'Name': 'name', 'Department': 'department', 'Fee': 'fee', 'Id': 'id', 'Car': 'car'}

domain_slot_dict = {
    'hotel': {'Type', 'Area', 'Phone', 'Day', 'Parking', 'Stars', 'Post', 'People', 'Price', 'Stay', 'Addr', 'Name',
              'Ref', 'Internet'}, 'police': {'Name', 'Phone', 'Post', 'Addr'},
    'train': {'Arrive', 'Day', 'Leave', 'Time', 'People', 'Ticket', 'Id', 'Ref', 'Dest', 'Depart'},
    'attraction': {'Type', 'Area', 'Phone', 'Fee', 'Post', 'Addr', 'Name'},
    'restaurant': {'Area', 'Phone', 'Day', 'Food', 'Post', 'Time', 'Addr', 'Price', 'People', 'Name', 'Ref'},
    'hospital': {'Post', 'Phone', 'Addr', 'Department'}, 'taxi': {'Arrive', 'Phone', 'Leave', 'Car', 'Dest', 'Depart'}}

meta = {'attraction': {'name', 'type', 'area'},
        'hotel': {'name', 'type', 'parking', 'area', 'day', 'stay', 'internet', 'people', 'stars', 'price'},
        'restaurant': {'name', 'food', 'area', 'day', 'time', 'people', 'price'},
        'taxi': {'arrive', 'departure', 'leave', 'destination'},
        'train': {'arrive', 'day', 'leave', 'destination', 'departure', 'people'}
        }

attraction_type = ['sport', 'entertainment', 'cinema', 'museum', 'theatre', 'church', 'boat', 'architecture', 'college',
                   'park', 'theater', 'camboats', 'concert', 'park', 'concert', 'hiking', 'historical', 'gallery',
                   'nightclub', 'special', 'swimming', 'gastropub', 'outdoor', 'pool', 'pub', 'club', 'swim', 'hall',
                   'movie']
hotel_type = ["hotel", "guesthouse", "guest house", "lodge"]

u_slots = set()
for d in meta:
    for u in meta[d]:
        u_slots.add(u)

qa_slots = u_slots.difference({'parking', 'internet'})
print("Unique slots : {}".format(u_slots))
print("QA slots : {}".format(qa_slots))

ignore_domain = ['booking']
domain_set = {'police', 'restaurant', 'hotel', 'taxi', 'attraction', 'train', 'hospital'}
spacy_en = spacy.load('en_core_web_sm')

dataset_config = os.path.join('trippy_label_variant', 'multiwoz21.json')
with open(dataset_config, "r", encoding='utf-8') as f:
    raw_config = json.load(f)
class_types = raw_config['class_types']
slot_list = raw_config['slots']
label_maps = raw_config['label_maps']

analyze = False

# --------------------------------------------

slot_domain_dict = {}
for dom in domain_slot_dict:
    for s in domain_slot_dict[dom]:
        if s not in slot_domain_dict:
            slot_domain_dict[slot_detail[s]] = set()
        slot_domain_dict[slot_detail[s]].add(dom)

print(slot_domain_dict)


# --------------------------------------------

def loadJson(data_file):
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


def isUnseen(slot_key, slot_val, bs):
    f = True
    if (slot_key in bs):
        if (slot_val == bs[slot_key]):
            f = False
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


def getBeliefState(belief_state):
    bs = {}
    for l in range(len(belief_state)):
        for sv in belief_state[l]['slots']:
            b_key = sv[0]
            if ("-book" in b_key):
                b_key_l = b_key.split(" ")
                b_key = b_key_l[0].split("-")[0] + "-" + correctSlotName(b_key_l[1])
            else:
                b_key = b_key.split("-")[0] + "-" + correctSlotName(b_key.split("-")[1])
            if (sv[1] != 'none'):
                bs[b_key] = sv[1]

    return cleanBeliefState(bs)


def getTurnPrediction(bs, bs_prev):
    bs_turn = {}
    for slot_key in bs:
        slot_val = bs[slot_key]
        if (isUnseen(slot_key, slot_val, bs_prev)):
            bs_turn[slot_key] = slot_val
    return bs_turn


def getDomainSlots(domain):
    s = set()
    for slot in domain_slot_dict[domain]:
        s.add(slot_detail[slot])
    return s


def cleanBeliefState(belief_state):
    bs = {}
    for k, v in belief_state.items():
        if (v != 'none'):
            bs[k] = v
    return bs


def cleanDialogAct(dialog_act):
    dst = {}
    for k in dialog_act:
        if (dialog_act[k] == "do n't care" or dialog_act[k] == "do nt care"):
            dst[k] = "dontcare"
        else:
            dst[k] = dialog_act[k]
    return dst


def correctSlotName(slot):
    if (slot == "arriveby"):
        return "arrive"
    elif (slot == "leaveat"):
        return "leave"
    elif (slot == "pricerange"):
        return "price"
    else:
        return slot


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text) if not tok.is_space]


def getSpanValue(domain, slot, txt, span_dict):
    span_val = " "
    start_idx = -1
    end_idx = -1
    span_key = domain + "-" + slot
    if (span_key in span_dict):
        if (str(span_dict[span_key][1]).isnumeric() and str(span_dict[span_key][1]).isnumeric()):
            tokens = tokenize_en(txt.lower())
            start_idx = span_dict[span_key][1]
            end_idx = span_dict[span_key][2]
            span_val = ' '.join(tokens[start_idx: end_idx + 1])
    return span_val


def isValidAnnotation(d_log):
    flag = False
    domain_set = set()
    for i, t in enumerate(d_log):
        if ('dialog_act' in t.keys()):
            if (len(list(t['dialog_act'])) > 0):
                ds = getDomain(list(t['dialog_act']))
                domain_set = domain_set.union(ds)
    if (len(domain_set) > 0):
        flag = True
    return flag


def getTurnLabel(tl):
    turn_label = {}
    for l in range(len(tl)):
        sv = tl[l]
        b_key = sv[0]
        if ("-book" in b_key):
            b_key_l = b_key.split(" ")
            b_key = b_key_l[0].split("-")[0] + "-" + correctSlotName(b_key_l[1])
        else:
            b_key = b_key.split("-")[0] + "-" + correctSlotName(b_key.split("-")[1])
        turn_label[b_key] = sv[1]
    return cleanBeliefState(turn_label)


def getDialogueAct(da):
    day = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    d_act = {}
    for k, v in da.items():
        dom_actual = k.split("-")[0].lower()
        if (len(v) > 0):
            for slots in v:
                if (len(slots) > 0 and slots[0] != 'none'):
                    if (dom_actual != 'general'):
                        if (slot_detail[slots[0]] == "day" and slots[1].lower() in day):
                            d_act[dom_actual + "-" + slot_detail[slots[0]]] = slots[1].lower()
                        else:
                            d_act[dom_actual + "-" + slot_detail[slots[0]]] = slots[1].lower()

    return cleanDialogAct(d_act)


def getDomain(dialog_act):
    domain_set = set()
    for d in dialog_act:
        t = d.split("-")[0].lower()
        if (t not in ignore_domain):
            domain_set.add(t)
    return domain_set

# ---------------------------------

# From bert.tokenization (TF code)
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
    text = re.sub("(\d{1})(a\.?m\.?|p\.?m\.?)", r"\1 \2", text)  # am/pm without space
    text = re.sub("(^| )(\d{1,2}) (a\.?m\.?|p\.?m\.?)", r"\1\2:00 \3", text)  # am/pm short to long form
    text = re.sub("(^| )(at|from|by|until|after) ?(\d{1,2}) ?(\d{2})([^0-9]|$)", r"\1\2 \3:\4\5",
                  text)  # Missing separator
    text = re.sub("(^| )(\d{2})[;.,](\d{2})", r"\1\2:\3", text)  # Wrong separator

    text = re.sub("(^| )(\d{2}):(\d{2})/", r"\1\2:\3", text)  # Wrong separator
    text = re.sub("(^| )(\d{1}) (\d{2})", r"\1\2:\3", text)  # Wrong separator
    text = re.sub("(^| )(\d{2}):!(\d{1})", r"\1\2:1\3", text)  # Wrong format

    text = re.sub("(^| )(at|from|by|until|after) ?(\d{1,2})([;., ]|$)", r"\1\2 \3:00\4",
                  text)  # normalize simple full hour time
    text = re.sub("(^| )(\d{1}:\d{2})", r"\g<1>0\2", text)  # Add missing leading 0
    # Map 12 hour times to 24 hour times
    text = re.sub("(\d{2})(:\d{2}) ?p\.?m\.?",
                  lambda x: str(int(x.groups()[0]) + 12 if int(x.groups()[0]) < 12 else int(x.groups()[0])) +
                            x.groups()[1], text)
    text = re.sub("(^| )24:(\d{2})", r"\g<1>00:\2", text)  # Correct times that use 24 as hour
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
    text = re.sub("archaelogy", "archaeology", text)  # Systematic typo
    text = re.sub("mutliple", "multiple", text)  # Systematic typo
    # text = re.sub("guesthouse", "guest house", text) # Normalization
    text = re.sub("(^| )b ?& ?b([.,? ]|$)", r"\1bed and breakfast\2", text)  # Normalization
    text = re.sub("bed & breakfast", "bed and breakfast", text)  # Normalization
    return text


def getValidAnnotations(path, ver, mode, data):
    dials_path = os.path.join(path, "mwz" + ver, mode + "_dials.json")
    dials = loadJson(dials_path)
    dials_data = {}
    for i, d in enumerate(dials):
        dials_data[d['dialogue_idx']] = d

    final_data = []

    c = 0
    for k, d in data:
        if (isValidAnnotation(d['log'])):
            c += 1
    print("Actual data : {}".format(len(data)))
    print("Valid data : {}".format(c))


def isReferral(slot, value_label, seen_slots):
    ref = "none"

    if slot == 'hotel-stars' or slot == 'hotel-internet' or slot == 'hotel-parking':
        return ref
    for s in seen_slots:
        # Avoid matches for slots that share values with different meaning.
        # hotel-internet and -parking are handled separately as Boolean slots.
        if s == 'hotel-stars' or s == 'hotel-internet' or s == 'hotel-parking':
            continue
        if re.match("(hotel|restaurant)-people", s) and slot == 'hotel-stay':
            continue
        if re.match("(hotel|restaurant)-people", slot) and s == 'hotel-stay':
            continue
        if slot != s and (slot not in seen_slots or seen_slots[slot] != value_label):
            if seen_slots[s] == value_label:
                ref = s
                break

            if value_label in label_maps:
                for value_label_variant in label_maps[value_label]:
                    if seen_slots[s] == value_label_variant:
                        ref = s
                        break

            if seen_slots[s] in label_maps:
                for value_label_variant in label_maps[seen_slots[s]]:
                    if value_label == value_label_variant:
                        ref = s
                        break

    return ref


def inUtterance(val, utt, utt_norm):
    f = False
    if (val.isnumeric()):
        pattern = "^{} | {} | {}.".format(val, val, val)
        if (re.search(pattern, utt)):
            f = True
        elif (val in label_maps):
            for value_label_variant in label_maps[val]:
                if (value_label_variant in utt or value_label_variant in utt_norm):
                    f = True
                    break
    else:
        if (val in utt or val in utt_norm):
            f = True
        elif (val in label_maps):
            for value_label_variant in label_maps[val]:
                if (value_label_variant in utt or value_label_variant in utt_norm):
                    f = True
                    break
    return f


def getValueDecision(sys, usr, dom, sl, val, span_dict_usr, span_dict_sys, seen_slots, informed_slots):
    slot_act = -2
    slot_ref = "none"

    span_val = ""
    span_val2 = ""
    span_val_actual = ""

    usr_norm = normalize_text(usr)
    span_key = dom + "-" + sl
    if (span_key in span_dict_usr):
        span_val_actual = span_dict_usr[span_key][0]
        span_val = getSpanValue(dom, sl, usr, span_dict_usr)
        span_val2 = getSpanValue(dom, sl, usr_norm, span_dict_usr)

    if (span_key in span_dict_usr):
        if (span_val == span_val_actual or span_val2 == span_val_actual):
            slot_act = 7
        elif (span_val_actual in label_maps):
            for value_label_variant in label_maps[span_val_actual]:
                if (span_val == value_label_variant or span_val2 == value_label_variant):
                    slot_act = 7
                    break

    if (slot_act == -2 and sl == "people" and val == "1"):
        slot_act = 5

    if (slot_act == -2 and inUtterance(val, usr, usr_norm)):
        slot_act = 7

    if (slot_act == -2 and span_key in span_dict_sys):
        span_val_actual = span_dict_sys[span_key][0]
        span_val = getSpanValue(dom, sl, sys, span_dict_sys)
        span_val2 = getSpanValue(dom, sl, normalize_text(sys), span_dict_sys)

        if (span_key in span_dict_sys):
            if (span_val == span_val_actual or span_val2 == span_val_actual):
                slot_act = 8
            elif (span_val_actual in label_maps):
                for value_label_variant in label_maps[span_val_actual]:
                    if (span_val == value_label_variant or span_val2 == value_label_variant):
                        slot_act = 8
                        break

    if (slot_act == -2 and inUtterance(val, sys, normalize_text(sys))):
        slot_act = 8

    ref = isReferral(span_key, val, seen_slots)
    if (slot_act == -2 and ref != "none"):
        slot_act = 9
        slot_ref = ref

    # Check if reference is present in system informed slots
    ref = isReferral(span_key, val, informed_slots)
    if (slot_act == -2 and ref != "none"):
        slot_act = 9
        slot_ref = ref

    return slot_act, slot_ref


def getAct(sys, usr, dom, sl, val, span_dict_usr, span_dict_sys, seen_slots, informed_slots):
    slot_value = val.lower()
    slot_ref = "none"

    if (slot_value == "?"):
        return 1, slot_ref
    elif (slot_value == "dontcare"):
        return 2, slot_ref
    elif (slot_value == "yes" or slot_value == "free"):
        return 3, slot_ref
    elif (slot_value == "no"):
        return 4, slot_ref
    elif (dom == "hotel" and sl == "type" and val == "hotel"):
        return 6, slot_ref
    else:
        return getValueDecision(sys, usr, dom, sl, slot_value, span_dict_usr, span_dict_sys, seen_slots, informed_slots)

#Get annotated start and end index of slot values
def getSpanDict(i, log):
    span_dict = {}
    if (i < 0):
        return span_dict

    t = log[i]
    span_info_len = 0
    if ('span_info' in t.keys()):
        span_info_len = len(t['span_info'])
    for idx in range(span_info_len):
        dom = t['span_info'][idx][0].split("-")[0].lower()
        if t['span_info'][idx][1] in slot_detail:
            sl = slot_detail[t['span_info'][idx][1]]
            span_key = dom + "-" + sl
            if (span_key not in span_dict):
                v = t['span_info'][idx][2].lower()
                span_value = [v, t['span_info'][idx][3], t['span_info'][idx][4]]
                span_dict[span_key] = span_value
            else:
                #Handling multiple slot value for a given domain-slot pair
                v = "{}$${}".format(span_dict[span_key][0], t['span_info'][idx][2].lower())
                start_idx = "{}$${}".format(span_dict[span_key][1], t['span_info'][idx][3])
                end_idx = "{}$${}".format(span_dict[span_key][2], t['span_info'][idx][4])
                span_value = [v, start_idx, end_idx]
                span_dict[span_key] = span_value
    return span_dict

# ------------------------------------------------------------

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#Map of domain-slot pair to question. Here "domain" is a placeholder for the actual domain.
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

#Convert a domain-slot pair to a question
def getQuestion(dom, slot, is_ref):
    q = ""
    if (is_ref):
        q = "What is the reference point of {} {}?".format(dom, slot)
    else:
        q = question_dict[slot]
        q = q.replace("domain", dom)
    return q.lower()

def find_sub_list(sub_lst, lst):
    results = []
    sub_len = len(sub_lst)
    try:
        for idx in (i for i, e in enumerate(lst) if e == sub_lst[0]):
            if lst[idx:idx + sub_len] == sub_lst:
                results.append((idx, idx + sub_len - 1))
    except:
        results = []
    return results

def find_lengthy_answer(tokens, context_inp, start_index, end_index):
    j = start_index
    k = end_index
    out_indexes = []
    for r in range(5):
        if (j - r - 1 < 0):
            break
        lengthy_ans = ' '.join(tokens[j - r - 1: k + 1])
        ans_enc = tokenizer.encode_plus(lengthy_ans)
        ans_inp = ans_enc["input_ids"]
        ans_inp = ans_inp[1:-1]
        out_indexes = find_sub_list(ans_inp, context_inp)[0]
        if (len(out_indexes) > 0):
            break

    if (len(out_indexes) == 0):
        j = start_index
        k = end_index
        for r in range(5):
            if (k + r + 1 == len(tokens) - 1):
                break
            lengthy_ans = ' '.join(tokens[j: k + r + 1])
            ans_enc = tokenizer.encode_plus(lengthy_ans)
            ans_inp = ans_enc["input_ids"]
            ans_inp = ans_inp[1:-1]
            out_indexes = find_sub_list(ans_inp, context_inp)[0]
            if (len(out_indexes) > 0):
                break
    return out_indexes

#Get the start and end index at bert token level to finetine the pre-trained SQuAD model
def getSpanIndex(text, dom, slot, span_value, start_index, end_index, is_ref):
    match = 0
    multi_match = 0

    tokens = tokenize_en(text)
    question = getQuestion(dom, slot, is_ref)
    context_enc = tokenizer.encode_plus(question, text)
    input_ids = context_enc["input_ids"]

    m = 0
    n = 0
    for i in range(len(input_ids)):
        if (m == 0 and input_ids[i] == 102):
            m = i
        if (m != 0 and input_ids[i] == 102):
            n = i

    context_inp = input_ids[m:n + 1]
    ans_enc = tokenizer.encode_plus(span_value)
    ans_inp = ans_enc["input_ids"]
    ans_inp = ans_inp[1:-1]

    out_indexes = find_sub_list(ans_inp, context_inp)
    decoded_ans = ""
    idx1 = 0
    idx2 = 0

    if (len(out_indexes) > 0):
        match = 1

    if (len(out_indexes) == 1):
        idx1 = out_indexes[0][0] + m
        idx2 = out_indexes[0][1] + m
        decoded_ans = tokenizer.decode(input_ids[idx1:idx2 + 1], clean_up_tokenization_spaces=True)

    if (len(out_indexes) > 1):
        multi_match = 1

        out_indexes2 = find_lengthy_answer(tokens, context_inp, start_index, end_index)
        if (len(out_indexes2) == 0):
            out_indexes2 = out_indexes
        context_inp2 = context_inp[out_indexes2[0]:out_indexes2[1] + 1]
        out_indexes3 = find_sub_list(ans_inp, context_inp2)[0]

        idx1 = m + out_indexes2[0] + out_indexes3[0]
        idx2 = m + out_indexes2[0] + out_indexes3[0] + len(ans_inp) - 1
        decoded_ans = tokenizer.decode(input_ids[idx1:idx2 + 1], clean_up_tokenization_spaces=True)

    return match, multi_match, idx1, idx2, decoded_ans


def isReferneceMentioned(usr, slot_ref):
    f = False
    val = ""

    d_ref = slot_ref.split("-")[0]
    if (d_ref in usr):
        f = True
        val = d_ref
    elif (d_ref == "hotel"):
        for v in hotel_type:
            if (v in usr):
                f = True
                val = v
                break
    elif (d_ref == "attraction"):
        for v in attraction_type:
            if v in usr:
                f = True
                val = v
                break
            elif (v in label_maps):
                for value_label_variant in label_maps[v]:
                    if (value_label_variant in usr):
                        f = True
                        val = value_label_variant
                        break
    return f, val


def get_token_pos(usr, value_label):
    tok_list = tokenize_en(usr)
    start_idx = -1
    end_idx = -1
    span_val = ""

    label_list = [item for item in map(str.strip, re.split("(\W+)", value_label)) if len(item) > 0]
    len_label = len(label_list)
    for i in range(len(tok_list) + 1 - len_label):
        t = ' '.join(tok_list[i:i + len_label])
        if (tok_list[i:i + len_label] == label_list or value_label in t):
            start_idx = i
            end_idx = i + len_label - 1
            span_val = t
            break

    return start_idx, end_idx, span_val

#Extract span information of slot values
def getSpanInfo(text, span_dict, final_data_val, slot_act):
    c = 0
    for span_key in span_dict:
        dom = span_key.split("-")[0]
        sl = span_key.split("-")[1]

        if not (dom in meta and sl in meta[dom]):
            continue

        span_val_actual = span_dict[span_key][0]
        s_tokens = tokenize_en(span_val_actual)
        if (sl in ["area", "price", "day", "people", "stay"] and len(s_tokens) > 1):
            continue

        if (sl == "stars"):
            span_val_actual = span_val_actual.split("-")[0]

        g = False
        if (str(span_dict[span_key][1]).isnumeric() and str(span_dict[span_key][2]).isnumeric()):
            start_idx = span_dict[span_key][1]
            end_idx = span_dict[span_key][2]
            span_val = getSpanValue(dom, sl, text, span_dict)
            span_val2 = getSpanValue(dom, sl, normalize_text(text), span_dict)

            if (span_val == span_val_actual or span_val2 == span_val_actual):
                g = True
            elif (span_val_actual in label_maps): #Check for label variant
                for value_label_variant in label_maps[span_val_actual]:
                    if (span_val == value_label_variant or span_val2 == value_label_variant):
                        g = True
                        break

        if (g):
            match, multi_match, idx1, idx2, decoded_ans = getSpanIndex(normalize_text(text),
                                                                       dom, sl, span_val2, start_idx, end_idx, False)
            if (analyze):
                print("Span :: {} : {} : {} : {} : {} : {}".format(span_key, idx1, idx2,
                                                                   span_val_actual, span_val2, decoded_ans))
            if (idx1 > -1 and idx2 > -1):
                c += 1
                question = getQuestion(dom, sl, False)
                final_data_val.append(
                    [question, normalize_text(text), dom, sl, slot_act, span_val2, decoded_ans, idx1, idx2])
    return c

def getBS(metadata):
    bs = {}
    for dom in metadata:
        # print(dom)
        book = metadata[dom]['book']
        semi = metadata[dom]['semi']
        for s in semi:
            if (len(semi[s]) > 0):
                slot = correctSlotName(s.lower())
                key = dom + "-" + slot
                bs[key] = semi[s][0]
        booked = book['booked']
        for s in book:
            if (s != 'booked'):
                if (len(book[s]) > 0):
                    slot = correctSlotName(s.lower())
                    key = dom + "-" + slot
                    bs[key] = book[s][0]
    return bs

# ------------------------------------------------------------

#Generate slot action slot value data
def getSlotActionData(path, out_dir, mode, data):
    dials_path = os.path.join(path, mode + "_dials.json")
    dials = loadJson(dials_path)
    dials_data = {}
    for i, d in enumerate(dials):
        dials_data[d['dialogue_idx']] = d

    final_data = []

    final_data_val = []
    c_valid = 0
    ref_count = 0

    c_val1 = 0
    c_val2 = 0

    for k, d in data:
        try:
            if (isValidAnnotation(d['log']) and k in dials_data):
                c_valid += 1
                dials_log = []
                if k in dials_data:
                    dials_log = dials_data[k]['dialogue']

                sys = " " #Set system utterance of turn 0 as " "
                seen_slots = {}
                informed_slots = {}
                bs_prev = {}

                for i, t in enumerate(d['log']):
                    if (i % 2 == 0 and i < len(d['log']) - 1): #User turn

                        usr = t['text'].strip().lower()
                        usr_norm = normalize_text(usr) #Normalize user utterance

                        # Get span_info
                        span_dict_usr = getSpanDict(i, d['log'])
                        span_dict_sys = getSpanDict(i - 1, d['log'])

                        # Get dialouge act
                        dialog_act = getDialogueAct(t['dialog_act'])

                        # Get turn label
                        turn_label = {}
                        idx = int((i + 1) / 2)
                        if (mwz_ver == "2.2"):
                            metadata = d['log'][i + 1]['metadata']
                            bs = getBS(metadata)
                            turn_label = getTurnPrediction(bs, bs_prev)
                            bs_prev = bs.copy()
                        else:
                            if (len(dials_log) > 0):
                                turn_label = getTurnLabel(dials_log[idx]['turn_label'])

                        if (analyze):
                            print("-----------------------")
                            print("Turn {}".format(idx))
                            print("Sys : {}".format(sys))
                            print("Usr : {}".format(usr))
                            print("TL : {}".format(turn_label))

                        # Union of turn label and dialogue act
                        tl_set = set(turn_label.keys())
                        da_set = set(dialog_act.keys())
                        slot_set = tl_set.union(da_set)

                        if (analyze):
                            print("slot_set : {}".format(slot_set))

                        if (analyze):
                            print("Usr span_dict : {}".format(span_dict_usr))
                            print("Sys span_dict : {}".format(span_dict_sys))

                        cur_dict = {}
                        prev_seen_slots = seen_slots.copy()

                        # Update informed slots by the system so far
                        for slot in span_dict_sys:
                            informed_slots[slot] = span_dict_sys[slot][0]

                        # Populate slot-value data
                        c_val = getSpanInfo(usr, span_dict_usr, final_data_val, 7)
                        c_val1 += c_val
                        c_val = getSpanInfo(sys, span_dict_sys, final_data_val, 8)
                        c_val2 += c_val

                        # Adding positive samples for slot-action
                        for slot in slot_set:
                            dom = slot.split("-")[0]
                            if dom not in meta:
                                continue
                            sl = slot.split("-")[1]
                            if (dom not in cur_dict):
                                cur_dict[dom] = set()
                            cur_dict[dom].add(sl)

                            slot_act = 0
                            val1 = " "
                            val2 = " "
                            if (slot in turn_label):
                                val1 = turn_label[slot].lower()
                            if (slot in dialog_act):
                                val2 = dialog_act[slot].lower()
                            val = val1 if (val1 != " ") else val2

                            #Get slot action
                            slot_act, slot_ref = getAct(sys, usr, dom, sl, val, span_dict_usr, span_dict_sys,
                                                        prev_seen_slots, informed_slots)

                            if (analyze):
                                print("Act :: {} : {} : {} : {} : {}".format(k, slot, val, slot_ref, slot_act))

                            if (slot_act > 0):
                                final_data.append([sys, usr_norm, dom, sl, val, slot_ref, slot_act])

                            if (val != "?"):
                                seen_slots[slot] = val

                            #Find start and end index of reference domain
                            if (slot_act == 9 and dom in meta):
                                r_flag, r_val = isReferneceMentioned(usr_norm, slot_ref)
                                question = getQuestion(dom, sl, True)
                                if (r_flag):
                                    start_idx, end_idx, span_val = get_token_pos(usr_norm, r_val)
                                    if (start_idx > -1 and end_idx > -1):
                                        ref_count += 1
                                        match, multi_match, idx1, idx2, decoded_ans = getSpanIndex(usr_norm, dom, sl,
                                                                                                   span_val,
                                                                                                   start_idx, end_idx,
                                                                                                   True)
                                        if (analyze):
                                            print("Ref :: {} : {} : {} : {}".format(r_val, idx1, idx2, decoded_ans))
                                        final_data_val.append(
                                            [question, usr_norm, dom, sl, slot_act, span_val, decoded_ans, idx1, idx2])
                                else:
                                    final_data_val.append([question, usr_norm, dom, sl, slot_act, " ", " ", 0, 0])

                        # Adding negative samples for slot-action and slot_value
                        dom_set = set()
                        for dom in cur_dict:
                            dom_set.add(dom)
                        for dom in cur_dict:
                            if dom in meta:
                                all_sets = meta[dom]
                            else:
                                all_sets = getDomainSlots(dom)
                            set_diff = all_sets.difference(cur_dict[dom])
                            if len(set_diff) > 0:
                                ls = list(set_diff)
                                random.shuffle(ls)
                                nc = int(len(all_sets) / 3)
                                if (nc == 0):
                                    nc = 1
                                neg = len(ls) if len(ls) < nc else nc
                                for set_diff_idx in range(neg):
                                    final_data.append([sys, usr_norm, dom, ls[set_diff_idx], " ", " ", 0])
                                    if (ls[set_diff_idx] in qa_slots):
                                        question = getQuestion(dom, ls[set_diff_idx], False)
                                        final_data_val.append(
                                            [question, usr_norm, dom, ls[set_diff_idx], 7, " ", " ", 0, 0])

                            # Add negative samples for domain
                            cur_dom_set = set()
                            cur_dom_set.add(dom)
                            # For co-existing domains
                            dom_diff = dom_set.difference(cur_dom_set)
                            for d_neg in dom_diff:
                                for st in cur_dict[dom]:
                                    if (st not in cur_dict[d_neg]):
                                        final_data.append([sys, usr_norm, d_neg, st, " ", " ", 0])

                            # For other domains
                            for st in cur_dict[dom]:
                                dom_diff = slot_domain_dict[st].difference(dom_set)
                                if len(dom_diff) > 0:
                                    ls = list(dom_diff)
                                    random.shuffle(ls)
                                    final_data.append([sys, usr_norm, ls[0], st, " ", " ", 0])

                    else:
                        sys = t['text'].strip().lower()

                if (analyze):
                    break
        except Exception as e:
            print("!!Error!!")
            print(e)
            print(k)
            print(idx)
            traceback.print_exc()

    df = pd.DataFrame(final_data, columns=['sys', 'usr', 'domain', 'slot', 'val', 'slot_ref', 'slot_act'])
    out_file = os.path.join(out_dir, mode + "_slot_act.tsv")
    df.to_csv(out_file, sep="\t", index=False)
    print("{} data generated".format(mode))
    print("Valid conversations : {}".format(c_valid))
    print("Shape of slot act data : {}".format(df.shape))
    print(df.groupby('slot_act').count())

    df = pd.DataFrame(final_data_val,
                      columns=['question', 'text', 'domain', 'slot', 'slot_act', 'val', 'dec_val', 'idx1', 'idx2'])
    out_file = os.path.join(out_dir, mode + "_slot_value.tsv")
    df.to_csv(out_file, sep="\t", index=False)
    print("Shape of slot value data : {}".format(df.shape))
    print("Ref count : {}".format(ref_count))
    print("Val for 7 : {}".format(c_val1))
    print("Val for 8 : {}".format(c_val2))
    print(df.groupby('slot').count())

# --------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('-path', '--path', help='Name of the input directory containing the input files.', required=False,
                    default=default_path)
parser.add_argument('-ver', '--ver', help='Version of the MultiWOZ dataset', required=False, default=default_mwz_ver)
parser.add_argument('-out', '--out', help='Name of the output directory.', required=False, default=default_out_path)
args = vars(parser.parse_args())

path = args['path']
out_dir = args['out']
mwz_ver = args['ver']

print("Input directory : {}".format(path))
print("Output directory : {}".format(out_dir))
print("MultiWOZ version : {}".format(mwz_ver))

if (not os.path.isdir(path)):
    print("Input directory {} does not exist.".format(path))
    exit(0)

if (not os.path.isdir(out_dir)):
    print("Creating output directory : {}".format(out_dir))
    os.mkdir(out_dir)

# Load raw data
dialog_data_file = os.path.join(path, 'data.json')
if (mwz_ver == "2.2"):
    dialog_data_file = os.path.join(path, 'data2.2.json')

dialog_data = loadJson(dialog_data_file)
dialog_id_list = list(set(dialog_data.keys()))
valid_list_file = os.path.join(path, 'valListFile.txt')
test_list_file = os.path.join(path, 'testListFile.txt')
valid_id_list = list(set(load_list_file(valid_list_file)))
test_id_list = load_list_file(test_list_file)
train_id_list = [did for did in dialog_id_list if did not in (valid_id_list + test_id_list)]
print('# of train dialogs:', len(train_id_list))
print('# of valid dialogs:', len(valid_id_list))
print('# of test dialogs :', len(test_id_list))

train_data = [(k, v) for k, v in dialog_data.items() if k in train_id_list]
valid_data = [(k, v) for k, v in dialog_data.items() if k in valid_id_list]
test_data = [(k, v) for k, v in dialog_data.items() if k in test_id_list]
assert (len(train_data) == len(train_id_list))
assert (len(valid_data) == len(valid_id_list))
assert (len(test_data) == len(test_id_list))

#Generate training data
getSlotActionData(path, out_dir, "train", train_data)
#Generate test data
getSlotActionData(path, out_dir, "test", test_data)
#Generate validation data
getSlotActionData(path, out_dir, "dev", valid_data)

print("done")
# --------------------------------
