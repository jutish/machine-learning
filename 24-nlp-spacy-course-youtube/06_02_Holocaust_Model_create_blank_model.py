import spacy

# Create an empty model
nlp = spacy.blank('en')

# Create a new NER object
ner = nlp.add_pipe('ner', name='conc_camps_ner')
# Add a label, this could be a list.
ner.add_label('CONC_CAMP')
nlp.to_disk('holocaust_model')
