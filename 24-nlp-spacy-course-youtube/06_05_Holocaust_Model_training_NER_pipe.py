import spacy
import json
import random
import re
from spacy.util import minibatch
from spacy.training.example import Example

def load_data(file):
    with open(file,'r',encoding='utf-8') as f:
        data = json.load(f)
        return data

def train_spacy(TRAINING_DATA, epochs):
    # nlp = spacy.blank('en')
    nlp = spacy.load('./holocaust_model')
    ner = nlp.add_pipe('ner', name='ner_conc_camp')
    ner.add_label('CAMP')
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe!='ner_conc_camp']
    with nlp.disable_pipes(*other_pipes): # disable other pipes to not affect them
        optimizer = nlp.begin_training()
        for epoch in range(epochs):
            print(f'Starting epoch: {epoch}')
            random.shuffle(TRAINING_DATA)
            losses = {}
            for batch in minibatch(TRAINING_DATA, size=8):
                for text, annotations in TRAINING_DATA:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    nlp.update([example],
                        drop=0.2,
                        sgd=optimizer,
                        losses=losses)
            print(losses)
    return  nlp        

# TRAINING_DATA = load_data('./sources/holocaust_ner_training_data.json')
# nlp = train_spacy(TRAINING_DATA, 5)
# nlp.to_disk('holocaust_model')

# Test the NER component of the model.

test = "The first concentration camps in Germany were established soon after\
Hitler's appointment as chancellor in January 1933. In the weeks after the\
Nazis came to power, the SA (Sturmabteilung; commonly known as the Storm\
Troopers), the SS (Schutzstaffel; Protection Squadrons—the elite guard of the\
Nazi party), the police, and local civilian authorities organized numerous\
detention camps to incarcerate real and perceived political opponents of Nazi\
policy.German authorities established camps all over Germany on an ad hoc basis to\
handle the masses of people arrested as alleged subversives. The SS established\
larger camps in Oranienburg, north of Berlin; Esterwegen, near Hamburg; Dachau,\
northwest of Munich; and Lichtenburg, in Saxony. In Berlin itself, the Columbia\
Haus facility held prisoners under investigation by the Gestapo (the German\
secret state police) until 1936."\

nlp = spacy.load('holocaust_model',disable=['camps_entity_ruler'])
for ent in nlp(test).ents:
    print(ent.text, ent.label_)