import nltk
import re
import pickle
import numpy as np
import json
from tensorflow.keras.models import load_model

with open("./json/general.json", encoding="utf-8") as file:
    general = json.load(file)

with open("./json/college.json", encoding="utf-8") as file:
    college = json.load(file)

with open("./json/entertainment.json", encoding="utf-8") as file:
    entertainment = json.load(file)

with open("./json/healthcare.json", encoding="utf-8") as file:
    healthcare = json.load(file)

with open("./json/swears.json", encoding="utf-8") as file:
    swears = json.load(file)

with open("./json/bots.json", encoding="utf-8") as file:
    bots = json.load(file)

intents = [*general, *college, *entertainment, *healthcare, *swears, *bots]

with open("./training_data.pkl", "rb") as f:
    words, classes = pickle.load(f)

model = load_model("./chatbot.h5")


def decontract(sentence):
    sentence = re.sub(r"'t", " not", sentence)
    sentence = re.sub(r"'re", " are", sentence)
    sentence = re.sub(r"'s", " is", sentence)
    sentence = re.sub(r"'d", " would", sentence)
    sentence = re.sub(r"'ll", " will", sentence)
    sentence = re.sub(r"'t", " not", sentence)
    sentence = re.sub(r"'ve", " have", sentence)
    sentence = re.sub(r"'m", " am", sentence)
    sentence = re.sub(r"[?,:,|,!,\",',(,),*,+,\,,\-,.,/,;,[,\],^,_,{,}]", "", sentence)
    sentence = re.sub("\s\s+", " ", sentence)
    sentence = sentence.lower()
    return sentence


def bag_of_words(sentence):
    sentence = decontract(sentence)
    sentence_words = nltk.word_tokenize(sentence)
    bag = [0] * len(words)

    for word in sentence_words:
        for i, w in enumerate(words):
            if word == w:
                bag[i] = 1

    return np.array(bag)


def predict_class(sentence):
    response = model.predict(np.array([bag_of_words(sentence)]))[0]
    results = [[i, r] for i, r in enumerate(response) if r > 0.05]
    results_list = []

    if len(results) == 0:
        results_list.append({"intent": "no_answer", "probability": str(0.9)})
        return results_list

    results.sort(key=lambda x: x[1], reverse=True)

    for r in results:
        results_list.append({"intent": classes[r[0]], "probability": str(r[1])})

    return results_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]["intent"]
    for i in intents_json:
        if i["tag"] == tag:
            return random.choice(i["responses"])


if __name__ == "__main__":
    ints = predict_class("I am sad")
    print(get_response(ints, intents))
