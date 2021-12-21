import json
import spacy
import random
from spacy.util import minibatch
from spacy.training.example import Example

# Load a json file
def load_json(file):
    with open(file,'r') as f:
        data = json.load(f)
    return(data)

def save_data(file, data):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def train_spacy(data, epochs, batch_size=8):
    TRAINING_DATA = data
    nlp = spacy.blank('en')
    if 'ner' not in nlp.pipe_names:
        ner = nlp.add_pipe('ner', last=True)
    for _, annotations in TRAINING_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for epoch in range(epochs):
            print('Starting epoch: '+ str(epoch))
            random.shuffle(TRAINING_DATA)
            losses = {}
            for batch in minibatch(TRAINING_DATA, size=batch_size):
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    nlp.update(
                        [example],
                        drop=0.2,  # Prevent overfitting
                        sgd=optimizer,
                        losses=losses,
                        )
            print(losses)
    return nlp

# TRAINING_DATA = load_json('./sources/training_data.json')
# nlp = train_spacy(TRAINING_DATA, 30)
# nlp.to_disk('hp_ner_model')

# Test our custom model
test = "When the first novel of the series, Harry Potter and the Philosopher's\
Stone, opens, it is apparent that some significant event has taken place in the\
wizarding world – an event so very remarkable that even Muggles\
(non-magical people) notice signs of it. The full background to this event and\
Harry Potter's past is revealed gradually throughout the series. After the\
introductory chapter, the book leaps forward to a time shortly before Harry\
Potter's eleventh birthday, and it is at this point that his magical background\
begins to be revealed.\
\
Despite Harry's aunt and uncle's desperate prevention of Harry learning about\
his abilities,[15] their efforts are in vain. Harry meets a half-giant, Rubeus\
Hagrid, who is also his first contact with the wizarding world. Hagrid reveals\
himself to be the Keeper of Keys and Grounds at Hogwarts as well as some of\
Harry's history.[15] Harry learns that, as a baby, he witnessed his parents'\
murder by the power-obsessed dark wizard Lord Voldemort (more commonly known by\
the magical community as You-Know-Who or He-Who-Must-Not-Be-Named, and by Albus\
Dumbledore as Tom Marvolo Riddle) who subsequently attempted to kill him as\
well.[15] Instead, the unexpected happened: Harry survived with only a\
lightning-shaped scar on his forehead as a memento of the attack, and Voldemort\
disappeared soon afterwards, gravely weakened by his own rebounding curse.\
\
As its inadvertent saviour from Voldemort's reign of terror, Harry has become a\
living legend in the wizarding world. However, at the orders of the venerable\
and well-known wizard Albus Dumbledore, the orphaned Harry had been placed in\
the home of his unpleasant Muggle relatives, the Dursleys, who have kept him\
safe but treated him poorly, including confining him to a cupboard without\
meals and treating him as their servant. Hagrid then officially invites Harry\
to attend Hogwarts School of Witchcraft and Wizardry, a famous magic school in\
Scotland that educates young teenagers on their magical development for seven\
years, from age eleven to seventeen.\
\
With Hagrid's help, Harry prepares for and undertakes his first year of study at\
Hogwarts. As Harry begins to explore the magical world, the reader is\
introduced to many of the primary locations used throughout the series. Harry\
meets most of the main characters and gains his two closest friends: Ron\
Weasley, a fun-loving member of an ancient, large, happy, but poor wizarding\
family, and Hermione Granger, a gifted, bright, and hardworking witch of\
non-magical parentage.[15][16] Harry also encounters the school and his friend Esteban told him"

nlp = spacy.load('./hp_ner_model')
doc = nlp(test)
for ent in doc.ents:
    print(ent.text, ent.label_)