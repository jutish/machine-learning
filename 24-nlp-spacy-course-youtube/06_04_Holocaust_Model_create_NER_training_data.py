import spacy
import json
import random
import re

# Function to save our training dataset
def save_data(file, data):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

# Load our custom holocaust_model and use our entity ruler to create the 
# training data for the NER.
nlp = spacy.load('holocaust_model')

# Load holocaust text and clean it a few!
with open('./sources/camps_text.txt','r',encoding='utf-8') as f:
    text = f.read()
# This is what spacy expect to train a NER
# TRAIN_DATA = [(text, {'entities':[(start, end, label)]})]
TRAIN_DATA = []
for sent in text.split('\n\n'):
    results = []
    entities = []
    sent = sent.strip()
    sent = sent.replace('\n',' ')
    sent = re.sub('[\(\[].*?[\)\]]','',sent) # remove [05] [03] this kind of references
    doc = nlp(sent)
    for ent in doc.ents:  # Find entities using our previous created entity ruler
        entities.append((ent.start_char, ent.end_char, ent.label_))
    if len(entities) > 0:
        results = [sent, {'entities': entities}]
        TRAIN_DATA.append(results)

save_data('./sources/holocaust_ner_training_data.json',TRAIN_DATA)