import spacy
import numpy as np
import re
from spacy.matcher import Matcher
from spacy import displacy


with open('./sources/alice.txt','r',encoding='utf-8') as f:
    text = f.read().replace('\n\n',' ').replace('\n',' ')
    chapters = text.split('CHAPTER ')[1:]

#
chapter1 = chapters[0]

# Create spaCy model
nlp = spacy.load('en_core_web_lg')

# Get sentences 
doc = nlp(chapter1)
sentences = list(doc.sents)
sentence = sentences[0]

# Extract entities of the book. Persons and things. We can use also do this 
# in "doc.ents"to get the whole chapter entities.
ents = list(sentence.ents)
print(ents[0])
print(ents[0].label)  # Return a numerical number
print(ents[0].label_) # Return what our entity is a PERSON
print(ents[0].text)   # Return the text

# We can make a list of PERSONS in the whole chapter.
chapter_ents = list(doc.ents)
persons = [people for people in chapter_ents if people.label_ == 'PERSON']
 
# Tokenizacion
# for token in sentence:
#     print(token.text, token.pos_) # .pos_ part of speach NOUN, PUNCT, ADV, etc.

# Example extracting all the NOUNS
nouns = [token for token in sentence if token.pos_ == 'NOUN']
# print(nouns)

# Extract NOUNS chunks for Example "a litle door" is a NOUN chunk
# "that lovely garden" is other good example
# print(list(doc.noun_chunks))

# Working with verbs using patterns
# Look for phrases with adv follow by a verb.
# Note: In the video use a library called 'textacy' but I couldn't installed it!
#       the actual version of Spacy support patterns so this library is not 
#       anymore necesary
matcher = Matcher(nlp.vocab)
# Define a pattern and add to the model given a name
patterns = [{'POS':'ADV'},{'POS':'VERB'}]  # Exactly the same as the video https://www.youtube.com/watch?v=VgGHwIWu-kU&list=PL2VXyKi-KpYvuOdPwXR-FZfmZ0hjoNSUo&index=7
matcher.add('ADV_VERB', [patterns])
# More complex like She run faster (NOUN, VERB, ADV)
patterns = [{'POS':'NOUN'},{'POS':'VERB'},{'POS':'ADV'}]
matcher.add('NOUN_VERB_ADV',[patterns])

# Print matches
matches = matcher(doc)
for match_id, start, end in matches:
    string_id = nlp.vocab.strings[match_id]  # Get string representation
    span = doc[start:end]  # The matched span
    print(match_id, string_id, start, end, span.text)

# Lematization
# Lemma is the root of a verb or noun. Boys --> boy | running --> run 
# Example extracting all the NOUNS
words = [(word, word.lemma_) for word in sentence if word.pos_ == 'VERB']
print(words)

# Reduce a text to its "lemmas" it is a good way to analyse frecuency of words
words_lemma = np.array([lemma[1] for lemma in words])
unique, counts = np.unique(words_lemma, return_counts=True)
counts = dict(zip(unique, counts))
print(counts)

# Display using "displacy" with 'dep' and 'ent' options.
short_sentence = sentences[4]
print(short_sentence)
html_dep = displacy.render(short_sentence, style='dep')  # dep --> Show relations in an arch way
with open('style_dep_sentence_4.html', 'w') as f:
    f.write(html_dep)

# The optios are not necesary but we can do this.
colors = {'PERSON':'#4E000'} # Color for the background
options = {'ents':['PERSON'], 'colors':colors} # Filter what to see
chapter = doc
html_ent = displacy.render(chapter, style='ent', options=options)  # ent --> Mark entities on the text
with open('style_ent_sentence_4.html', 'w') as f:
    f.write(html_ent)

# # Finding quotes on text
# def find_sents(text=chapter1):
#     doc = nlp(text)
#     sentences = list(doc.sents)
#     return sentences

# def get_quotes(text):
#     quotes = re.findall(r" '(.*?)")

# found_sents = find_sents()
# for sent in found_sents:
#     print(sent)