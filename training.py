import nltk
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import random
import json
import pickle
import re

nltk.download('punkt')


with open('./json/general.json', encoding='utf-8') as file:
    general = json.load(file)

with open('./json/college.json', encoding='utf-8') as file:
    college = json.load(file)

with open('./json/entertainment.json', encoding='utf-8') as file:
    entertainment = json.load(file)

with open('./json/healthcare.json', encoding='utf-8') as file:
    healthcare = json.load(file)

with open('./json/swears.json', encoding='utf-8') as file:
    swears = json.load(file)

with open('./json/bots.json', encoding='utf-8') as file:
    bots = json.load(file)

with open("./json/depression.json", encoding="utf-8") as file:
    depression = json.load(file)


def decontract(sentence):
    sentence = re.sub(r"'t", " not", sentence)
    sentence = re.sub(r"'re", " are", sentence)
    sentence = re.sub(r"'s", " is", sentence)
    sentence = re.sub(r"'d", " would", sentence)
    sentence = re.sub(r"'ll", " will", sentence)
    sentence = re.sub(r"'t", " not", sentence)
    sentence = re.sub(r"'ve", " have", sentence)
    sentence = re.sub(r"'m", " am", sentence)
    sentence = re.sub(
        r"[?,:,|,!,\",',(,),*,+,\,,\-,.,/,;,[,\],^,_,{,}]", "", sentence)
    sentence = re.sub("\s\s+", " ", sentence)
    sentence = sentence.lower()
    return sentence


words = []
classes = []
documents = []

for intent in [*general, *college, *entertainment, *healthcare, *swears, *bots, *depression]:
    for pattern in intent["patterns"]:
        pattern = decontract(pattern)
        pattern = nltk.word_tokenize(pattern)
        words.extend(pattern)
        documents.append((pattern, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = sorted(set(words))

classes = sorted(set(classes))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]

    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

with open("training_data.pkl", "wb") as f:
    pickle.dump((words, classes), f)


model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer="adam", metrics=['accuracy'])

history = model.fit(np.array(train_x), np.array(train_y), epochs=len(train_x[0]), batch_size=15)
model.save("chatbot.h5", history)

model.evaluate(np.array(train_x),np.array(train_y), steps=27)