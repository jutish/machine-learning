# Text Classification with SpaCy A common task in NLP is text classification. This
# is "classification" in the conventional machine learning sense, and it is
# applied to text. Examples include spam detection, sentiment analysis, and
# tagging customer queries.

# In this tutorial, you'll learn text classification with spaCy. The classifier
# will detect spam messages, a common functionality in most email clients. Here
# is an overview of the data you'll use:

import pandas as pd

# Loading the spam data
# ham is the label for non-spam messages
spam = pd.read_csv('spam.csv')
print(spam.head())

# Bag of Words Machine learning models don't learn from raw text data. Instead,
# you need to convert the text to something numeric.

# The simplest common representation is a variation of one-hot encoding. You
# represent each document as a vector of term frequencies for each term in the
# vocabulary. The vocabulary is built from all the tokens (terms) in the corpus
# (the collection of documents).

# As an example, take the sentences "Tea is life. Tea is love." and "Tea is
# healthy, calming, and delicious." as our corpus. The vocabulary then is
# {"tea", "is", "life", "love", "healthy", "calming", "and", "delicious"}
# (ignoring punctuation).

# For each document, count up how many times a term occurs, and place that count
# in the appropriate element of a vector. The first sentence has "tea" twice and
# that is the first position in our vocabulary, so we put the number 2 in the
# first element of the vector. Our sentences as vectors then look like

# v1=[22110000]
# v2=[11001111]
 
# This is called the bag of words representation. You can see that documents with
# similar terms will have similar vectors. Vocabularies frequently have tens of
# thousands of terms, so these vectors can be very large.

# Another common representation is TF-IDF (Term Frequency - Inverse Document
# Frequency). TF-IDF is similar to bag of words except that each term count is
# scaled by the term's frequency in the corpus. Using TF-IDF can potentially
# improve your models. You won't need it here. Feel free to look it up though!

################# Building a Bag of Words model ###########################

# Building a Bag of Words model Once you have your documents in a bag of words
# representation, you can use those vectors as input to any machine learning
# model. spaCy handles the bag of words conversion and building a simple linear
# model for you with the TextCategorizer class.

# The TextCategorizer is a spaCy pipe. Pipes are classes for processing and
# transforming tokens. When you create a spaCy model with nlp = spacy.load
# ('en_core_web_sm'), there are default pipes that perform part of speech
# tagging, entity recognition, and other transformations. When you run text
# through a model doc = nlp("Some text here"), the output of the pipes are
# attached to the tokens in the doc object. The lemmas for token.lemma_ come from
# one of these pipes.

# You can remove or add pipes to models. What we'll do here is create an empty
# model without any pipes (other than a tokenizer, since all models always have a
# tokenizer). Then, we'll create a TextCategorizer pipe and add it to the empty
# model.

import spacy

# Create an empty model
nlp = spacy.blank('en')

# Add the TextCategorizer to the empty model
textcat = nlp.add_pipe('textcat')

# Next we'll add the labels to the model. Here "ham" are for the real
# messages, "spam" are spam messages.

# Add labels to text classifier
textcat.add_label('ham')
textcat.add_label('spam')

# Training a Text Categorizer Model Next, you'll convert the labels in the data to
# the form TextCategorizer requires. For each document, you'll create a
# dictionary of boolean values for each class.

# For example, if a text is "ham", we need a dictionary {'ham': True, 'spam':
# False}. The model is looking for these labels inside another dictionary with
# the key 'cats'.
train_texts = spam['text'].values
train_labels = [{'cats':{'ham': label == 'ham',
                         'spam': label == 'spam'}}
                 for label in spam['label']]

# Then we combine the texts and labels into a single list.
# we get an array of tuples with the form [('Text...',{'ham':True, 'spam':False})]
train_data = list(zip(train_texts, train_labels))

# Now you are ready to train the model. First, create an optimizer using
# nlp.begin_training(). spaCy uses this optimizer to update the model. In
# general it's more efficient to train models in small batches. spaCy provides
# the minibatch function that returns a generator yielding minibatches for
# training. Finally, the minibatches are split into texts and labels, then used
# with nlp.update to update the model's parameters.
from spacy.util import minibatch
from spacy.training.example import Example

# Create an optimizer using nlp.begin_training()
# spacy.util.fix_random_seed(1)
# optimizer = nlp.begin_training()

# # Create the batch generator with batch size = 8
# batches = minibatch(train_data, size=8)
# for batch in batches:
#     # Each batch is a list of (text, label)
#     for text, labels in batch:
#         doc = nlp.make_doc(text)
#         example = Example.from_dict(doc, labels)
#         nlp.update([example], sgd=optimizer)

# This is just one training loop (or epoch) through the data. The model will
# typically need multiple epochs. Use another loop for more epochs, and
# optionally re-shuffle the training data at the begining of each loop.

import random
random.seed(1)
spacy.util.fix_random_seed(1)
optimizer = nlp.begin_training()
epoches = 10
losses = {}
for epoch in range(epoches):
    random.shuffle(train_data)
    # Create the batch generator with batch size = 8
    batches = minibatch(train_data, size=8)
    # Iterate through minibatches
    for batch in batches:
        for text, labels in batch:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, labels)
            nlp.update([example], sgd=optimizer, losses=losses)
    print(f'Epoch: {epoch} - Loss: {losses}')

# Making Predictions 
# Now that you have a trained model, you can make predictions
# with the predict() method. The input text needs to be tokenized with
# nlp.tokenizer. Then you pass the tokens to the predict method which returns
# scores. The scores are the probability the input text belongs to the classes.
texts = ["Are you ready for the tea party????? It's gonna be wild",
         "URGENT Reply to this message for GUARANTEED FREE TEA" ]
docs = [nlp.tokenizer(text) for text in texts]

# Use textcat to get the scores for each doc
textcat = nlp.get_pipe('textcat')
scores = textcat.predict(docs)

# The scores are used to predict a single class or label by choosing the label
# with the highest probability. You get the index of the highest probability
# with scores.argmax, then use the index to get the label string from
# textcat.labels.
# From the scores, find the label with the highest score/probability
predicted_labels = scores.argmax(axis=1)
print([textcat.labels[label] for label in predicted_labels])