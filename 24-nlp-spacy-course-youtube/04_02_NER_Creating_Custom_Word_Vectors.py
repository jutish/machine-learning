import json
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import multiprocessing

# Here we use our "hp.json", previously created training dataset, to get a 
# word vector model based on Harry Potter first book.
def training(model_name):
    with open('./sources/hp.json','r',encoding='utf-8') as f:
        texts = json.load(f)
    sentences = texts
    cores = multiprocessing.cpu_count()
    w2v_model = Word2Vec(min_count=2,
        window=2,
        vector_size = 500,
        sample=6e-5,
        alpha=0.03,
        min_alpha=0.0007,
        negative=20,
        workers=cores-1)
    w2v_model.build_vocab(texts)
    w2v_model.train(texts, total_examples=w2v_model.corpus_count, epochs=30)
    w2v_model.save(f'sources/{model_name}.model') # save the model
    w2v_model.wv.save_word2vec_format(f'sources/word2vec_{model_name}.txt')

# Training the model.
# training('hp_ner_model_03')

# Return a top of 10 of words wich have similar word_vectors to "word"
# We load the vector using KeyedVector.
def gen_similarity(word):
    word = word.lower()
    model = KeyedVectors.load_word2vec_format('./sources/word2vec_hp_ner_model_03.txt',
        binary=False)
    results = model.most_similar(positive=[word])
    print(results)

gen_similarity('ron')

