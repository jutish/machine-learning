# To load a custom word vector to an Spacy model we have 2 options.

# Method N°1: Using a comand line shell
# Go to the path where we want to work and where the word_vector is.
#                        1      2     3                     4                        5
# type: python -m spacy init vectors en ./sources/word2vec_hp_ner_model_03.txt ./hp_model_test
# 1: create a blank model
# 2: create a vectors pipeline
# 3: set the model language
# 4: set vector source created on Gensim or other tool
# 5: set output where the component will be saved

# Method N°2: Use a python function to call the same command shell

import subprocess
import sys

model_name = './hp_model_test'
word_vectors = './sources/word2vec_hp_ner_model_03.txt'

# Usage: python -m spacy init vectors [OPTIONS] LANG VECTORS_LOC OUTPUT_DIR
# Try 'python -m spacy init vectors --help' for help.
def load_word_vectors(model_name, word_vectors):
    subprocess.run([sys.executable,
        '-m',
        'spacy',
        'init',
        'vectors',
        'en',
        word_vectors,
        model_name])

load_word_vectors(model_name, word_vectors)