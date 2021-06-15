"""
Description :
Generates the train/test/validation data for domain change and domain prediction model.

Run Command:
python create_domain_data.py -path=<path of input data> -out=<output directory>
"""
#--------------------------------------------
import os
import json
import argparse
import pandas as pd
import six, re
import random

random.seed(1234)
ignore_domain = ['booking']
all_domains = {'hotel', 'police', 'train', 'attraction', 'restaurant', 'hospital', 'taxi'}
default_path = os.path.join('data', 'mwz2.1')
default_out_path = os.path.join('data', 'domain_data')

#--------------------------------------------

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

#---------------------------------------------
def getDomain(dialog_act):
    domain_set = set()
    for d in dialog_act:
        t = d.split("-")[0].lower()
        if(t not in ignore_domain):
            domain_set.add(t)
    return domain_set

def isValidAnnotation(d_log):
    flag = False
    domain_set = set()
    for i, t in enumerate(d_log):
        if ('dialog_act' in t.keys()):
            if(len(list(t['dialog_act']))>0):
                ds = getDomain(list(t['dialog_act']))
                domain_set = domain_set.union(ds)
    if(len(domain_set)>0):
        flag = True
    return flag

def correctSlotName(slot):
    if(slot=="arriveby"):
        return "arrive"
    elif(slot=="leaveat"):
        return "leave"
    elif(slot=="pricerange"):
         return "price"
    else:
        return slot
    
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
    return turn_label

def getLabel(ds_sys, ds_usr, tl, ds_usr_prev, tl_prev):    
    g = {'general'}
    label = 1
    if(len(ds_usr.symmetric_difference(g))==0):
        label = 2
    elif(len(ds_sys.symmetric_difference(g))==0 and len(ds_usr)==0):
        label = 2
    else:
        if('general' in ds_sys):
            ds_sys.remove('general')
        if('general' in ds_usr):
            ds_usr.remove('general')
        if('general' in ds_usr_prev):
            ds_usr_prev.remove('general')
            
        s1 = ds_usr.symmetric_difference(ds_sys)
        s2 = ds_usr.symmetric_difference(tl)
        s3 = ds_sys.symmetric_difference(tl)
        
        if(len(tl)>0 and len(tl_prev)>0 and len(tl.symmetric_difference(tl_prev))>0):
            label = 1
        elif(len(tl)>0 and len(tl_prev)>0 and len(tl.symmetric_difference(tl_prev))==0):
            label = 0
        elif(len(ds_usr)>0 and len(ds_usr_prev)>0 and len(ds_usr.symmetric_difference(ds_usr_prev))>0):
            label = 1
        elif(len(ds_usr)>0 and len(ds_usr_prev)>0 and len(ds_usr.symmetric_difference(ds_usr_prev))==0):
            label = 0
        elif(len(s1)==0 and len(s2)==0):
            label = 0
        elif(len(s1)==0 and len(tl)==0):
            label = 0
        elif(len(s2)==0 and len(ds_sys)==0):
            label = 0
        elif(len(s3)==0 and len(ds_usr)==0):
            label = 0
        elif(len(ds_sys)>=0 and len(ds_usr)==0 and len(tl)==0):
            label = 0
        elif(len(ds_usr)>=0 and len(ds_sys)==0 and len(tl)==0):
            label = 0
        else:
            label = 1
    return label  

def getLabeledDomains(ds_usr, domain_tl):
    d_pos = ds_usr.union(domain_tl)
    
def generateDomainData(path, out_dir, mode, data):
    dials_path = os.path.join(path, mode+"_dials.json")
    dials = loadJson(dials_path)
    dials_data = {}
    for i,d in enumerate(dials):
        dials_data[d['dialogue_idx']] = d
    print("#Convesations in {} dials : {}".format(mode, len(dials_data)))
    
    data_domain = [] #Data for domain model
    data_switch = [] #Data for switch model
    
    for k,d in data:
        if(isValidAnnotation(d['log'])):
            ds_sys = set()
            ds_usr = set()
            ds_usr_prev = set()
            domain_tl = set()
            domain_tl_prev = set() 
            sys = " "
            
            log = []
            if k in dials_data:
                log = dials_data[k]['dialogue']
            
            for i, t in enumerate(d['log']):
                if(i%2==0):
                    usr = normalize_text(t['text'].strip())
                    ds_usr_prev = ds_usr.copy()
                    ds_usr = getDomain(t['dialog_act'])
                    
                    idx = int((i+1)/2)
                    tl = {}
                    if(len(log)>0):
                        tl = getTurnLabel(log[idx]['turn_label'])
                  
                    domain_tl_prev = domain_tl.copy()
                    domain_tl = getDomain(tl)
                    
                    if(i==0):
                        label = 1
                    else:
                        label = getLabel(ds_sys, ds_usr, domain_tl, ds_usr_prev, domain_tl_prev)
                    
                    #Set switch data
                    if(i>0):
                        data_switch.append([k, sys, usr, label])
                    
                    #Set domain data
                    if(label==1):
                        d_pos = ds_usr.union(domain_tl)
                        if('general' in d_pos):
                            d_pos.remove('general')
                        if(len(d_pos)>0):
                            for dom in d_pos:
                                data_domain.append([k, sys, usr, dom, 1])
                            d_neg = all_domains.difference(d_pos)
                            for dom in d_neg:
                                data_domain.append([k, sys, usr, dom, 0])
                        
                    #print("{} : {} : {} : {} : {} : {}".format(i, sys+"#"+usr, ds_sys, ds_usr, domain_tl, label))
                else:
                    sys = t['text'].strip().lower()
                    ds_sys = getDomain(t['dialog_act'])

    df = pd.DataFrame(data_switch, columns =['idx','sys', 'usr', 'label'])
    print("Shape of {} switch data : {}".format(mode, df.shape))
    #print(df.head())
    print(df.groupby('label').count())
    out_file = os.path.join(out_dir, mode+"_switch.tsv")
    df.to_csv(out_file, sep="\t", index=False)
    
    df = pd.DataFrame(data_domain, columns =['idx','sys', 'usr', 'domain', 'label'])
    print("Shape of {} domain data : {}".format(mode, df.shape))
    #print(df.head())
    print(df.groupby('label').count())
    out_file = os.path.join(out_dir, mode+"_domain.tsv")
    df.to_csv(out_file, sep="\t", index=False)
            
#--------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('-path','--path', help='Name of the input directory containing the input files.', required=False, default=default_path)
parser.add_argument('-out','--out', help='Name of the output directory.', required=False, default=default_out_path)
args = vars(parser.parse_args())

path = args['path']
out_dir = args['out']

print("Input directory : {}".format(path))
print("Output directory : {}".format(out_dir))

if(not os.path.isdir(path)):
    print("Input directory {} does not exist.".format(path))
    exit(0)
    
if(not os.path.isdir(out_dir)):
    print("Creating output directory : {}".format(out_dir))
    os.mkdir(out_dir) 

#Load raw data
dialog_data_file = os.path.join(path, 'data.json')
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
train_data = [(k,v) for k, v in dialog_data.items() if k in train_id_list]
valid_data = [(k,v) for k, v in dialog_data.items() if k in valid_id_list]
test_data = [(k,v) for k, v in dialog_data.items() if k in test_id_list]
assert(len(train_data) == len(train_id_list))
assert(len(valid_data) == len(valid_id_list))
assert(len(test_data) == len(test_id_list))

#Generate domain and switch model data
generateDomainData(path, out_dir, "train", train_data)
generateDomainData(path, out_dir, "dev", valid_data)
generateDomainData(path, out_dir, "test", test_data)

print("done")
#--------------------------------