# https://www.kaggle.com/matleonard/intro-to-nlp 
# NLP with spaCy
# spaCy is the  leading library for NLP, and it has quickly become one of 
# the most popular Python frameworks. Most people find it intuitive, and it 
# has excellent documentation.

import spacy
nlp = spacy.load('en_core_web_sm')

# With the model loaded, you can process text like this:
doc = nlp("Tea is healthy and calming, don't you think?")

# Tokenizing
# This returns a document object that contains tokens. A token is a unit of text
# in the document, such as individual words and punctuation. SpaCy splits
# contractions like "don't" into two tokens, "do" and "n't". You can see the
# tokens by iterating through the document. 
for token in doc:
    print(token)

# Iterating through a document gives you token objects. Each of these tokens
# comes with additional information. In most cases, the important ones are
# token.lemma_ and token.is_stop.

# Text preprocessing 
# There are a few types of preprocessing to improve how we
# model with words. The first is "lemmatizing." (token.lemma_) 
# The "lemma" of a word is its
# base form. For example, "walk" is the lemma of the word "walking". So, when
# you lemmatize the word walking, you would convert it to walk.

# It's also common to remove stopwords. (token.is_stop) 
# Stopwords are words that occur
# frequently in the language and don't contain much information. English
# stopwords include "the", "is", "and", "but", "not".

# With a spaCy token, token.lemma_ returns the lemma, while token.is_stop
# returns a boolean True if the token is a stopword (and False otherwise).
print(f"Token \t\tLemma \t\tStopword".format('Token', 'Lemma', 'Stopword'))
print("-"*40)
for token in doc:
    print(f"{str(token)}\t\t{token.lemma_}\t\t{token.is_stop}")

# Why are lemmas and identifying stopwords important? Language data has a lot of
# noise mixed in with informative content. In the sentence above, the important
# words are tea, healthy and calming. Removing stop words might help the
# predictive model focus on relevant words. Lemmatizing similarly helps by
# combining multiple forms of the same word into one base form
# ("calming", "calms", "calmed" would all change to "calm").

# However, lemmatizing and dropping stopwords might result in your models
# performing worse. So you should treat this preprocessing as part of your
# hyperparameter optimization process.

# Pattern Matching Another 
# common NLP task is matching tokens or phrases within
# chunks of text or whole documents. You can do pattern matching with regular
# expressions, but spaCy's matching capabilities tend to be easier to use.

# To match individual tokens, you create a Matcher. When you want to match a
# list of terms, it's easier and more efficient to use PhraseMatcher. For
# example, if you want to find where different smartphone models show up in
# some text, you can create patterns for the model names of interest. First you
# create the PhraseMatcher itself.

from spacy.matcher import PhraseMatcher
# The matcher is created using the vocabulary of your model. (nlp)
matcher = PhraseMatcher(nlp.vocab, attr='LOWER')

# The matcher is created using the vocabulary of your model. Here we're using the
# small English model you loaded earlier. Setting attr='LOWER' will match the
# phrases on lowercased text. This provides case insensitive matching.

# Next you create a list of terms to match in the text. The phrase matcher needs
# the patterns as document objects. The easiest way to get these is with a list
# comprehension using the nlp model.

terms = ['Galaxy Note', 'iPhone 11', 'iPhone XS', 'Google Pixel']
patterns = [nlp(t) for t in terms]
matcher.add('TerminologyList', patterns)

# Then you create a document from the text to search and use the phrase matcher
# to find where the terms occur in the text.
# The matches here are a tuple of the match id and the positions of the start
# and end of the phrase.
text_doc = nlp("Glowing review overall, and some really interesting side-by-side "
               "photography tests pitting the iPhone 11 Pro against the "
               "Galaxy Note 10 Plus and last year’s iPhone XS and Google Pixel 3.")
matches = matcher(text_doc)
print(matches)

match_id, start, end = matches[0]

# Exercise

# Solution: You could group reviews by what menu items they mention, and then
# calculate the average rating for reviews that mentioned each item. You can
# tell which foods are mentioned in reviews with low scores, so the restaurant
# can fix the recipe or remove those foods from the menu.

import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher

# Read reviews
data = pd.read_json('restaurant.json')

# Get a sample
index_of_review_to_test_on = 14
text_to_test_on = data.text.iloc[index_of_review_to_test_on]

# Load Spacy Model
nlp = spacy.blank('en')

# Create the tokenized version of text_to_test_on
review_doc = nlp(text_to_test_on)

# Create the PhraseMatcher object. The tokenizer is the first argument. 
# Use attr = 'LOWER' to make consistent capitalization
matcher = PhraseMatcher(nlp.vocab, attr='LOWER')

menu = ["Cheese Steak", "Cheesesteak", "Steak and Cheese", "Italian Combo", "Tiramisu", "Cannoli",
        "Chicken Salad", "Chicken Spinach Salad", "Meatball", "Pizza", "Pizzas", "Spaghetti",
        "Bruchetta", "Eggplant", "Italian Beef", "Purista", "Pasta", "Calzones",  "Calzone",
        "Italian Sausage", "Chicken Cutlet", "Chicken Parm", "Chicken Parmesan", "Gnocchi",
        "Chicken Pesto", "Turkey Sandwich", "Turkey Breast", "Ziti", "Portobello", "Reuben",
        "Mozzarella Caprese",  "Corned Beef", "Garlic Bread", "Pastrami", "Roast Beef",
        "Tuna Salad", "Lasagna", "Artichoke Salad", "Fettuccini Alfredo", "Chicken Parmigiana",
        "Grilled Veggie", "Grilled Veggies", "Grilled Vegetable", "Mac and Cheese", "Macaroni",  
         "Prosciutto", "Salami"]

# Create a list of tokens for each item in the menu
pattern = [nlp(text) for text in menu]
matcher.add('MENU',pattern)

# Find matches in the review_doc
matches = matcher(review_doc)
# print(review_doc)
for match in matches:
    print(f'Token N°: {match[1]} {review_doc[match[1]:match[2]]}')

# Matching on the whole dataset
# Now run this matcher over the whole dataset and collect ratings for each menu
# item. Each review has a rating, review.stars. For each item that appears in the
# review text (review.text), append the review's rating to a list of ratings for
# that item. The lists are kept in a dictionary item_ratings.

# To get the matched phrases, you can reference the PhraseMatcher documentation
# for the structure of each match object:

# A list of (match_id, start, end) tuples, describing the matches. A match tuple
# describes a span doc[start:end]. The match_id is the ID of the added match
# pattern.
from collections import defaultdict
items_review = defaultdict(list)
for idx, row in data.iterrows():
    text = nlp(row.text)
    matches = matcher(text)
    items = set([text[match[1]:match[2]].text.lower() for match in matches])
    for item in items:
        items_review[item].append(row.stars)
items_mean = {key: sum(items_review[key])/len(items_review[key]) for key in items_review.keys()}
items_count = {key: len(items_review[key]) for key in items_review.keys()}

# Worst item
worst = min(items_mean, key=items_mean.get)
print(f'{worst} - Mean Stars:{items_mean[worst]} Reviews Count:{items_count[worst]}')

items = sorted(items_count, key=items_count.get, reverse=True)
for item in items:
    print(f"{item:>25}{items_count[item]:>5}")

sorted_ratings = sorted(items_mean, key=items_mean.get)
print("Worst rated menu items:")
for item in sorted_ratings[:10]:
    print(f"{item:20} Ave rating: {items_mean[item]:.2f} \tcount: {items_count[item]}")
    
print("\n\nBest rated menu items:")
for item in sorted_ratings[-10:]:
    print(f"{item:20} Ave rating: {items_mean[item]:.2f} \tcount: {items_count[item]}")

# Solution: The less data you have for any specific item, the less you can trust
# that the average rating is the "real" sentiment of the customers. This is
# fairly common sense. If more people tell you the same thing, you're more likely
# to believe it. It's also mathematically sound. As the number of data points
# increases, the error on the mean decreases as 1 / sqrt(n).