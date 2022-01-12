import spacy
from spacy.language import Language
from spacy.tokens import Span

# Accuracy
#  Precision: True Positive vs False Positive
#  Recall: True Negative vs False Negative


main_nlp = spacy.blank('en')
english_nlp = spacy.load('en_core_web_sm')
camps_nlp = spacy.load('holocaust_model')

@Language.factory('en_dln')
def get_english_ner(nlp, name):
    return english_nlp.get_pipe('ner')

@Language.factory('camps_ner')
def get_camps_ner(nlp, name):
    return camps_nlp.get_pipe('ner_conc_camp')

main_nlp.add_pipe('en_dln') # dln -> dates, locations, and norp (Nationalities or religious or political groups)
main_nlp.add_pipe('camps_ner', before='en_dln')

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

# Add our custom function
main_nlp.add_pipe('en_narrow', after='en_dln')

vocabs = ['CONC_CAMP']
for vocab in vocabs:
    main_nlp.vocab.strings.add(vocab)

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

print(main_nlp.pipe_names)

doc = main_nlp(test) # this run all the pipes include our custom function 'en_narrow'
for ent in doc.ents:
    print(ent.text, ent.label_)