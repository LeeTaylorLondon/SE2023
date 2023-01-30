# Author: Lee Taylor
import gensim
import gzip
import pickle
from ModelInput import *
from object_storage import *
from gensim.models import KeyedVectors
import numpy as np


def readfile(fd="Dataset/trial.data.v1.txt"):
    """ Returns a list of lines from a text file """
    rv = []
    with open(fd, "r", errors="replace") as f:
        for line in f:
            rv.append(line)
    return rv

def process_lines(lines):
    """ Returns two lists of words, and image names. """
    words, imgs = [], []
    for line in lines:
        line_split = line.split()
        for word in line_split:
            if word.__contains__('.jpg') or \
                word.__contains__('.png'):
                imgs.append(word)
            else:
                words.append(word)
    return words, imgs

def process_lines_obj(lines):
    """ Returns an object. """
    objs = []
    for i, line in enumerate(lines):
        words, imgs = [], []
        line_split = line.split()
        for word in line_split:
            if word.__contains__('.jpg') or \
                word.__contains__('.png'):
                imgs.append(word)
            else:
                words.append(word)
        try:
            _ = DataObject(i, [words[0]], words[1:], imgs)
            # _.create_embeddings(tokenization)
            objs.append(_)
        except ValueError as e:
            print(e)
    return objs

def print_list(list, name):
    print(f"\n---- {name} ----\n")
    for _ in list:
        print(_)
    pass

def tokenization(word, more_words, out=False):
    # Example of `words` parameter
    # words = ["I", "am", "a", "sentence", "to", "be", "tokenized"]
    # words = ["andromeda"]

    tokenized_words = [word[0]]
    for w in more_words:
        tokenized_words.append(w)

    words = tokenized_words

    # Tokenize the words
    # tokenized_words = [_.split() for _ in words]

    # Train the word2vec model
    model = gensim.models.Word2Vec(tokenized_words,
                                   min_count=1, vector_size=1,
                                   window=5, sg=0)
    # Print the vocabulary of the model
    vocab = model.wv.index_to_key
    if out:
        print(vocab)
    # Print the vector of a word
    try:
        vector = model.wv.get_vector(words[0])
    except KeyError:
        vector = [0, 0, 0]
    if out:
        print(vector)
    # Return value
    return vector

def get_word2vec_embedding(words, model):
    # Make sure the words are in lowercase
    words = [word.lower() for word in words]

    # Check if all the words are in the vocabulary of the word2vec model
    for word in words:
        if word not in model.index_to_key:
            return np.zeros(shape=(300))

    # Return the average of the word embeddings
    return sum([model[word] for word in words]) / len(words)


if __name__ == '__main__':
    lines = readfile()
    words, imgs = process_lines(lines)
    # data_objs = process_lines_obj(lines)

    # for obj in data_objs:
    #     print(obj)

    # store_obj(data_objs, "Object Storage/word_list_pkl.gz")

    word_list = load_obj("Object Storage/word_list_pkl.gz")

    for obj in word_list:
        print(obj)

    # data_objs = load_obj("Object Storage/word_dict_.pkl.gz")
    # print(data_objs.values())

    # Load the word2vec model
    model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)

    # Example usage
    # words = ["dog", "cat", "horse"]
    # words = ['andromeda', 'andromeda', 'tree']
    words = ['andromeda']
    embedding = get_word2vec_embedding(words, model)

    if embedding is not None:
        print(embedding)
        print(len(embedding))
    else:
        print("One or more words not found in vocabulary.")