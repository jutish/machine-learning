import spacy
from spacy.language import Language
from spacy.tokens import Span

main_nlp = spacy.blank('en')

@Language.factory('en_ner')
def get_english_ner(nlp, name):
    english_nlp = spacy.load('en_core_web_sm')
    return english_nlp.get_pipe('ner')

main_nlp.add_pipe('en_ner') # dln -> dates, locations, and norp (Nationalities or religious or political groups)

# Create a function that only check for dates, locations, person and norp in our Doc object
@Language.component('en_narrow')
def en_narrow(doc):
    l = []
    for ent in doc.ents:
        if ent.label_ in ['DATE','NORP','PERSON']:
            l.append(ent)
        elif ent.label_ == 'GPE':
            new_ent = Span(doc, ent.start, ent.end, label='LOCATION')
            l.append(new_ent)
    l = tuple(l)
    doc.ents = l
    return doc

# Add our custom function
main_nlp.add_pipe('en_narrow', after='en_ner')

# Test our model
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

doc = main_nlp(test) # this run all the pipes include our custom function 'en_narrow'
for ent in doc.ents:
    print(ent.text, ent.label_)

main_nlp.to_disk('model_with_custom_pipe')
