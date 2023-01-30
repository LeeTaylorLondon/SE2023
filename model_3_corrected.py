import tensorflow as tf
from tensorflow.keras import layers
from object_storage import *
from gensim.models import KeyedVectors
import numpy as np
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Reshape
from word_embedding import get_word2vec_embedding
from model_3c_results import results
import math


# Load object containing words, image names (IN) can be used
# with object `image dictionary` to retrieve the actual image.
obj_list = load_obj("Object Storage/word_list_pkl.gz")

# Dict `img_` is (`image_name` : `np.array shape -> (1, 7, 7, 512)`)
img_dict = load_obj("Object Storage/image_dict_.pkl.gz")

# `word_list` init. empty list to contain all words
word_list = []

# Dict `win` is (`line_number` : `[3_words, 10_image_names]`)
win = {}
for obj in obj_list:
    # dict `win`
    obj.update_words()
    win.update(
        {obj.line_number: [obj.words, obj.images]}
    )
    # list `word_list`
    for word in obj.words:
        word_list.append(word)

# Populate ModelInput objects with `goldimg`
with open("Dataset/trial.gold.v1.txt") as f:
    gold_lines = f.readlines()
# Clean
for i,v in enumerate(gold_lines):
    gold_lines[i] = v.strip()
# Set gold image
for i in range(len(gold_lines)):
    obj_list[i].goldimg = gold_lines[i]
# Check obj_
print(gold_lines)
for obj in obj_list:
    print(obj)

# Check word list
print(f"`word_list` = {word_list}\n")

# Create dict `joint_dict` -> (i, [[word1, word2, word3], [image_embed]])
i = 0
joint_dict = {}
for value in list(win.values()):
    words, images = value[0], value[1]
    for image_name in images:
        try:
            joint_dict.update({i: [words, img_dict[image_name]]})
        except KeyError as e:
            print(f"KeyError: {e}")
        i += 1

"""  """

# def condense_array(arr):
#     # dense_layer = Dense(300, activation='relu', input_shape=(None, 25088, 1))
#     dense_layer = Dense(300, activation='relu', input_shape=(None, 1, 300))
#     return dense_layer(arr)
#
# # Load pre-trained word2vec model
# word2vec_model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)
#
# # Define vocab & embedding size
# word_index = {word: i for i, word in enumerate(word_list)}
# vocab_size = len(word_index) + 1
# embedding_size = 300
#
# embedding_matrix = np.zeros((vocab_size, embedding_size))
# for word, i in word_index.items():
#     if i >= vocab_size:
#         print(f"Index {i} out of bounds for vocab size {vocab_size}")
#         continue
#     if word in word2vec_model.index_to_key:
#         embedding_matrix[i] = word2vec_model.get_vector(word)
#     else:
#         print(f"{word} not in vocab")
#
# # Input layers for the 3 words
# word1_input = layers.Input(shape=(1,), dtype='float32', name='word1_input')
# word2_input = layers.Input(shape=(1,), dtype='float32', name='word2_input')
# word3_input = layers.Input(shape=(1,), dtype='float32', name='word3_input')
#
# # Embedding layer
# embedding_layer = layers.Embedding(input_dim=vocab_size,
#                                    output_dim=embedding_size,
#                                    input_length=1,
#                                    weights=[embedding_matrix])
# word1_embed = embedding_layer(word1_input)
# word2_embed = embedding_layer(word2_input)
# word3_embed = embedding_layer(word3_input)
#
# # Input layer for the image embed
# img_input = Input(shape=(1, 300), name='img_input')
#
# # Concatenate the word embeddings and image embed
# concatenated = Concatenate()([word1_embed, word2_embed, word3_embed, img_input]) # `image_embed`
# concatenated = Flatten()(concatenated)
#
# # Final dense layer for prediction
# prediction = Dense(1, activation='sigmoid')(concatenated)
#
# # Build the model
# model = Model(inputs=[word1_input, word2_input, word3_input, img_input], outputs=prediction)
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.summary()
#
# def sigmoid(x):
#     return 1 / (1 + math.exp(-x))
#
# def char_to_int(string):
#     int_values = [ord(char) for char in string]
#     rv = ''
#     for iv in int_values:
#         rv = rv + str(iv)
#     # rv = int(rv) % 1
#     rv = sigmoid(int(rv))
#     rv = np.array(rv, dtype='float32')
#     return rv
#
#
# all = []
# for data_obj in obj_list:
#     """
#     For each 3 words predict 10 images -> [x1, x2, ..., x9, x10]
#
#     """
#     single = []
#     for image_name in data_obj.images:
#         try:
#             # Define input
#             embeds = [
#                 char_to_int(data_obj.words[0]),
#                 char_to_int(data_obj.words[1]),
#                 char_to_int(data_obj.words[2]),
#                 # list(img_dict.values())[0]
#                 img_dict[image_name]
#             ]
#             # Reshape all
#             for i, e in enumerate(embeds[:-1]):
#                 e = np.reshape(e, (1, -1))
#                 embeds[i] = e
#             # Redefine + reshape `image`
#             img = list(img_dict.values())[0]
#             img = np.reshape(img, (1, -1))[0]
#             img = np.reshape(img, (1, -1))
#             img = condense_array(img)
#             embeds[-1] = img
#             # Reshape all once more
#             for i in range(len(embeds)):
#                 embeds[i] = np.reshape(embeds[i], (1, 1, -1))
#             # Pass the input data to the model for prediction
#             prediction = model.predict(embeds)
#             single.append(float(prediction[0]))
#             print("Prediction:", prediction)
#         except:
#             print(f"\nERROR\n{embeds[:3]}\nERROR\n")
#             single.append([None])
#     all.append(single)
#
# for single_ in all:
#     print(single_)

matches = []
for i, arr in enumerate(results):
    max_index = arr.index(max(arr))
    matched_image = obj_list[i].images[max_index]
    gold_img = obj_list[i].goldimg
    if matched_image == gold_img: matches.append(1)
    else: matches.append(0)

print(matches)
print(f"Percentage Match: {(matches.count(1) / len(matches) * 100)}%")

# for single_ in all:
    # gold_lines[single_.index(max(single_))]

