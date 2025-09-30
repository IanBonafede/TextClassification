
# IMPORTANT---------------------
# pip install numpy==1.16.1 or else will not work
# after done:pip install numpy=1.18.1 

import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

(train_data, train_labels) , (test_data, test_labels) = data.load_data(num_words=88000)

word_index = data.get_word_index() # gets tuples with string of word and integer of word

word_index = {k:(v+3) for k, v in word_index.items()}# split the tuple into key and value, 3 special keys or characters
word_index["<PAD>"] = 0 #if movie reviews different length, short ones we can add padding that tf wont care about
word_index["START"] = 1
word_index["<UNK>"] = 2 #if word is unknown
word_index["<UNUSED>"] = 3

#swap keys and values for an int to point to a word so we can see what they are in the data!!
reversed_word_index = dict([(value, key) for (key, value) in word_index.items()]) 

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

def decode_review(text):
    return " ".join([reversed_word_index.get(i, "?") for i in text])


# model here
#uncomment to train
#comment to test
'''
model = keras.Sequential()
model.add(keras.layers.Embedding(88000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

fitmodel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

results = model.evaluate(test_data, test_labels)

print(results)

model.save("Text_Classification_Model.h5")
'''
#testing model from saved model
#uncomment to test
#comment to train
def review_encode(s):
    encoded = [1]

    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)
    return encoded

model = keras.models.load_model("Text_Classification_Model.h5")

with open("Text_Classification_test.txt", encoding="utf-8") as f:
    for line in f.readlines():
        nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"", "").strip().split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250)
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])



'''
test_review = test_data[0]
predict = model.predict([test_review])
print("Review: ")
print(decode_review(test_review))
print("Prediction: " + str(predict[0]))
print("Actual: " + str(test_labels[0]))
print(results)
'''