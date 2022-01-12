import spacy
import json
import random
from collections import defaultdict

# Create a custom entity ruler based on holocaust list "camps.json"
with open('./sources/camps.json', 'r', encoding='utf-8') as f:
    camps = json.load(f)

# Create patterns for our "entity ruler"
patterns = []
label = 'CONC_CAMP'
for camp in camps:
    pattern = {
        'label': label,
        'pattern': camp
    }
    patterns.append(pattern)

# Save our model with a new pipe of type entity ruler called "camps_entity_ruler"
nlp = spacy.blank('en')
entity_ruler = nlp.add_pipe('entity_ruler', name='camps_entity_ruler')
entity_ruler.add_patterns(patterns)
nlp.to_disk('holocaust_model')

# Test our entity ruler
results = []
with open('./sources/camps_text.txt','r',encoding='utf-8') as f:
    text = f.read()
    tokenize_text = nlp(text)

# Count matches in the text
result_count = defaultdict(int)
for ent in tokenize_text.ents:  # .ents use our recently created entity_ruler
    # results.append(ent.text)
    result_count[ent.text] += 1
print(result_count)