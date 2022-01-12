# #   NAMED ENTITY RECOGNITION SERIES   #
# #             Lesson 01               #
# #        Introduction to NER          #
# #               with                  #
# #        Dr. W.J.B. Mattingly         #

# Series Outline:
# 01 => Introduction to NER
# 02 => Gazetteer and NER in Python (Rules-Based NER)
# 03 => Introduction to Machine Learning NER
# 04 => using spaCy's Named Entity Recognition
# 05 => What's under the hood of spaCy?
# 06 => Identifying the weaknesses in spaCy's NER
# 07 => Introduction to Word Vectors
# 08 => Generating Custom Word Vectors in Gensim
# 09 => Importing Custom Word Vectors from Gensim into spaCy
# 10 => Training spaCy's NER on new domain-specific texts
# 11 => Creating New Entity Labels in spaCy
# 12 => Generating New Training Data Quickly
# 13 => Training and Deploying a Domain NER Model

######################################################
# 02 => Gazetteer and NER in Python (Rules-Based NER)#
######################################################

import json

with open('./sources/harry_potter.txt','r') as f:
    text = f.read().split('\n\n')[3:4]
    # print(text)

character_names = []
with open('./sources/hp_characters.json','r') as f:
    characters = json.load(f)
    for character in characters:
        names  = character.split()
        for name in names:
            if "and" != name and "the" != name and "The" != name:
                name = name.replace(",", "").strip()
                character_names.append(name)
    # print(characters)

# Clean data
for segment in text:
    segment = segment.strip()
    segment = segment.replace('\n',' ')
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for letter in segment:  # iterate letter by letter.
        if letter in punc:
            segment = segment.replace(letter, "")
    words = segment.split()
    i = 0
    for word in words:
        if word in character_names:
            if words[i-1][0].isupper():
                print (f"Found Character(s): {words[i-1]} {word}")
            else:
                print (f"Found Character(s): {word}")

        i=i+1