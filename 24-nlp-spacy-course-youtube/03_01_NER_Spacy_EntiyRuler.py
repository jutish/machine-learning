################################################
# 03 => Introduction to Machine Learning NER   #
# 04 => using spaCy's Named Entity Recognition #
################################################
import spacy
import json
from spacy.lang.en import English
from spacy.pipeline import EntityRuler  # Allows create rule to find entities

# Load a json file
def load_json(file):
    with open(file,'r') as f:
        data = json.load(f)
    return(data)

def save_data(file, data):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

# Generate a better list of characters
def generate_better_characters(file):
    data = load_json(file)
    new_characters = []
    for item in data:
        new_characters.append(item)
    for item in data:
        item = item.replace('The','').replace('the','').replace('and','')\
            .replace('And','')
        item = item.split() # by default use whitespace to split
        for name in item:
            name = name.strip() # clean whitespace before and after words
            new_characters.append(name)
        if '(' in item:
            names = item.split('(')
            for name in names:
                name = name.replace(')','').strip()
        if ',' in item:
            names = item.split(',')
            for name in names:
                name = name.replace('and','').strip()
                if ' ' in name:
                    new_names = name.split()
                    for x in new_names:
                        x = x.strip()
                        new_characters.append(x)
                new_characters.append(name) 
    final_characters = []
    titles = ['Dr.','Professor','Mr.','Mrs.','Ms.','Miss','Aunt','Uncle',
    'Mr. and Mrs.']
    for character in new_characters:
        final_characters.append(character)
        for title in titles:
            titled_char = f'{title} {character}'
            final_characters.append(titled_char)
    return list(set(final_characters))  # remove duplicates

def create_training_data(file, type):
    data = generate_better_characters(file)
    patterns = []
    for item in data:
        pattern = {
            'label':type,
            'pattern':item
        }
        patterns.append(pattern)
    return patterns

def generate_rules(patterns):
    nlp = English()  # create an empty model
    ruler = nlp.add_pipe('entity_ruler')
    ruler.add_patterns(patterns)
    nlp.to_disk('hp_ner')

# patterns = create_training_data('./sources/hp_characters.json','PERSON')
# generate_rules(patterns)

def test_model(model, text):
    doc = model(text)
    results = []
    for ent in doc.ents:
        results.append(ent.text)
    return results

ie_data = {}  # Save characters by chapter
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
            for result in results:
                hits.append(result)
            ie_data[chapter_num] = hits

save_data('./sources/hp_data.json', ie_data)