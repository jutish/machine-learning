import pandas as pd
import spacy
import random
from spacy.util import minibatch
from spacy.training.example import Example

# Load data and split it into a training and validation set
def load_data(csv_file, split=0.9):
    data = pd.read_csv(csv_file)
    # Shuffle data
    train_data = data.sample(frac=1, random_state=7)
    texts = train_data.text.values #  return a numpy array
    labels = [{'POSITIVE': bool(y), 'NEGATIVE': not bool(y)}
        for y in train_data.sentiment.values]
    split = int(len(train_data) * split)
    train_labels = [{'cats':labels} for labels in labels[:split]]
    val_labels = [{'cats':labels} for labels in labels[split:]]
    return texts[:split], train_labels, texts[split:], val_labels

train_texts, train_labels, val_texts, val_labels = load_data('yelp_ratings.csv')

# Create an empty model
nlp = spacy.blank('en')

# Add the TextCategorizer to the empty model
textcat = nlp.add_pipe('textcat')

# Add labels to text classifier
textcat.add_label('NEGATIVE')
textcat.add_label('POSITIVE')

# Implement a function train that updates a model with training data.
def train(model, train_data, optimizer, batch_size=8):
    losses = {}
    random.seed(1)
    random.shuffle(train_data)
    # train_data is a list of tuples [(text0, label0), (text1, label1), ...]
    for batch in minibatch(train_data, size=batch_size):
        # split batch into text and labels
        for text, labels in batch:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, labels)
            # Update model with texts and labels
            model.update([example], sgd=optimizer, losses=losses)
    return losses

# Fix seed for reproducibility
spacy.util.fix_random_seed(1)
random.seed(1)

# This may take a while to run!
optimizer = nlp.begin_training()
train_data = list(zip(train_texts, train_labels))
losses = train(nlp, train_data, optimizer)
print(losses['textcat'])

# We can try this slightly trained model on some example text and look at the
# probabilities assigned to each label.
text = "This tea cup was full of holes. Do not recommend."
doc = nlp(text)
print(doc.cats)

# Making predictions
# First, tokenize the texts using nlp.tokenizer().
# Then, pass those docs to the TextCategorizer which you can get from nlp.get_pipe().
# Use the textcat.predict() method to get scores for each document, 
# then choose the class with the highest score (probability) as the predicted class.
def predict(nlp, texts):
    # Use the model's tokenizer to tokenize each input text
    docs = [nlp.tokenizer(text) for text in texts]
    # Use textcat to get the scores for each doc
    textcat = nlp.get_pipe('textcat')
    prediction = textcat.predict(docs)  # Return for each text [[prob class 1, prob class 2],..,]
    # From the scores, find the class with the highest score/probability
    predicted_class = prediction.argmax(axis=1)
    return predicted_class

# Predict using a few texts
texts = val_texts[34:38]
predictions = predict(nlp, texts)
for p, t in zip(predictions, texts):
    print(f"{textcat.labels[p]}: {t} \n")

# Evaluate the model Implement a function that evaluates a TextCategorizer
# model. This function evaluate takes a model along with texts and labels. It
# returns the accuracy of the model, which is the number of correct predictions
# divided by all predictions.

# First, use the predict method you wrote earlier to get the predicted class for
# each text in texts. Then, find where the predicted labels match the
# true "gold-standard" labels and calculate the accuracy.
def evaluate(model, texts, labels):
    """ Returns the accuracy of a TextCategorizer model. 
    
        Arguments
        ---------
        model: ScaPy model with a TextCategorizer
        texts: Text samples, from load_data function
        labels: True labels, from load_data function
    
    """
    # Get predictions from textcat model (using your predict method)
    predicted_class = predict(model, texts)
    
    # From labels, get the true class as a list of integers (POSITIVE -> 1, NEGATIVE -> 0)
    true_class = [int(cat['cats']['POSITIVE']) for cat in labels]

    # A boolean or int array indicating correct predictions
    correct_predictions = predicted_class == true_class
    # The accuracy, number of correct predictions divided by all predictions
    accuracy = correct_predictions.mean()
    
    return accuracy

accuracy = evaluate(nlp, val_texts, val_labels)
print(f"Accuracy: {accuracy:.4f}")

# This may take a while to run!
n_iters = 5
for i in range(n_iters):
    losses = train(nlp, train_data, optimizer)
    accuracy = evaluate(nlp, val_texts, val_labels)
    print(f"Loss: {losses['textcat']:.3f} \t Accuracy: {accuracy:.3f}")