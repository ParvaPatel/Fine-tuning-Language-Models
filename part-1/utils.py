import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.

def get_synonym(word):
    synsets = wordnet.synsets(word)
    if len(synsets) > 0:
        lemmas = synsets[0].lemmas()
        if len(lemmas) > 0:
            synonym = lemmas[0].name().replace("_", " ")
            if synonym.lower() != word.lower():
                return synonym
    return word


def introduce_typo(word):
    if len(word) <= 3:
        return word
    i = random.randint(0, len(word) - 2)
    word = list(word)
    word[i], word[i + 1] = word[i + 1], word[i]
    return "".join(word)


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    # raise NotImplementedError
    
    text = example["text"]

    # Tokenize properly (better than split)
    words = word_tokenize(text)

    new_words = []
    for word in words:
        r = random.random()

        # 25% synonym replacement
        if r < 0.25:
            new_words.append(get_synonym(word))

        # next 15% typo
        elif r < 0.40:
            new_words.append(introduce_typo(word))

        else:
            new_words.append(word)

    # Detokenize back to sentence
    detokenizer = TreebankWordDetokenizer()
    example["text"] = detokenizer.detokenize(new_words)
    
    ##### YOUR CODE ENDS HERE ######

    return example