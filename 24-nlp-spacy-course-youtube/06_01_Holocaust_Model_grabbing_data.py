# Train a model from scratch to try to reconigze the concentrations camps as PLACE.
# We use a list of camps from wiki-pedia

import json
import glob
import re

def load_data(file):
    with open(file,'r',encoding='utf-8') as f:
        data = json.load(f)
    return data

def write_data(file, data):
    with open(file,'w',encoding='utf-8') as f:
        json.dump(data, f, indent=4)

camps = []
with open('./sources/camps.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        result = re.findall(r"[A-Z].*? ", line)[0].replace(' ','')
        camps.append(result)

write_data('./sources/camps.json',camps)