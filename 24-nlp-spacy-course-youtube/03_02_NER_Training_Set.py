################################################
# 04 => using spaCy's Named Entity Recognition #
# Create training data                         #
################################################

import spacy
import json
import random

# Load a json file
def load_json(file):
    with open(file,'r') as f:
        data = json.load(f)
    return(data)

def save_data(file, data):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

# Create training dataset
def test_model(model, text):
    # TRAIN_DATA = [(text, {'entities':[(start, end, label)]})] # this is what spacy expect to train
    doc = model(text)
    results = []
    entities = []
    for ent in doc.ents:
        entities.append((ent.start_char, ent.end_char, ent.label_))
    if len(entities) > 0:
        results = [text, {'entities': entities}]
    return results

TRAIN_DATA = []
nlp = spacy.load('hp_ner') # Load our customize model
with open('./sources/harry_potter.txt','r') as f:
    text = f.read()
    chapters = text.split('CHAPTER ')[:]
    for chapter in chapters:
        chapter_num, chapter_title = chapter.split('\n\n',)[0:2]
        segments = chapter.split('\n\n')[2:]  # Only text without the title
        hits = []
        for segment in segments:
            segment = segment.strip()
            segment = segment.replace('\n',' ')
            results = test_model(nlp, segment)
            if results != None and len(results) > 0:
                TRAIN_DATA.append(results)

save_data('./sources/training_data.json', TRAIN_DATA)