import spacy
from spacy.language import Language

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

# Load our model
holocaust_nlp = spacy.load('holocaust_model',disable=['camps_entity_ruler'])

# Load spacy model and add our tag 'CON_CAMP'
main_nlp = spacy.load('en_core_web_sm')
vocabs = ['CONC_CAMP']
for vocab in vocabs:
    main_nlp.vocab.strings.add(vocab)

# Use a factory decorator to define our NER Factory
@Language.factory('camps_ner')
def get_camps_ner(nlp, name):
    return holocaust_nlp.get_pipe('ner_conc_camp')

# Add our NER factory to spacy model
main_nlp.add_pipe('camps_ner',before='ner')

# Test that our pipe is really working.
for ent in main_nlp(test).ents:
    print(ent.text, ent.label_)
