import spacy
from spacy.language import Language
from spacy.tokens import Span

# Create a function that only check for dates, locations and onrp in our Doc object
@Language.component('en_narrow')
def en_narrow(doc):
    l = []
    for ent in doc.ents:
        if ent.label_ in ['DATE','NORP','CONC_CAMP','PERSON']:
            l.append(ent)
        elif ent.label_ == 'GPE':
            new_ent = Span(doc, ent.start, ent.end, label='LOCATION')
            l.append(new_ent)
    l = tuple(l)
    doc.ents = l
    return doc

@Language.factory('en_ner')
def get_english_ner(nlp, name):
    english_nlp = spacy.load('en_core_web_sm')
    return english_nlp.get_pipe('ner')