# %% [code]
! pip install ml-datasets

# %% [code]
# Import libraries
import spacy
from spacy.tokens import DocBin
from ml_datasets import imdb

# %% [code]
# Import data
train_data, valid_data = imdb()

# %% [code]
print(train_data[0])

# %% [code]
def make_docs(data):
    docs = []
    for doc, label in nlp.pipe(data, as_tuples=True):
        if label == 'neg':
            doc.cats['positive'] = 0
            doc.cats['negative'] = 1
        else:
            doc.cats['positive'] = 1
            doc.cats['negative'] = 0
        docs.append(doc)
    return docs

# %% [code]
nlp = spacy.load('en_core_web_sm')

# %% [code]
num_texts = 500
train_docs = make_docs(train_data[:num_texts])
doc_bin = DocBin(docs=train_docs)
doc_bin.to_disk('train.spacy')

valid_docs = make_docs(valid_data[:num_texts])
doc_bin = DocBin(docs=valid_docs)
doc_bin.to_disk('valid.spacy')

# %% [code]
!python -m spacy init config config.cfg --lang en --pipeline ner,textcat --optimize efficiency

# %% [code]
# Training our model https://spacy.io/api/cli#train
# The output will save out the best model from all epochs, as well as the final pipeline.
!python -m spacy train config.cfg --paths.train ./train.spacy --paths.dev ./valid.spacy --output ./output

# %% [code] {"execution":{"iopub.status.busy":"2022-03-07T14:52:07.236993Z","iopub.execute_input":"2022-03-07T14:52:07.238066Z","iopub.status.idle":"2022-03-07T14:52:07.579936Z","shell.execute_reply.started":"2022-03-07T14:52:07.238014Z","shell.execute_reply":"2022-03-07T14:52:07.578998Z"}}
# Testing our custom training model with unseeing data num_texts+1
test_text = train_data[num_texts + 2]

# Load our best model
nlp = spacy.load('./output/model-best')

doc = nlp(test_text[0])
print(doc.cats)
print(test_text)
