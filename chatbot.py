from urllib import request
import streamlit as st
import streamlit.components.v1 as components
import random
import json
import torch
import os
import numpy as np
import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('C:/Users/DELL/myenv/qna_updated_final.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "C:/Users/DELL/myenv/data.pth"
data = torch.load(FILE)

def tokenize(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    stemming = find the root form of the word
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):

    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag

all_words = []
tags = []
xy = []
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))


import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out
    

input_size = 108
hidden_size = 8
output_size = 34
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

print("Let's test! (type 'quit' to exit)")

def chat():
    while True:
        # sentence = "do you use credit cards?"
        sentence = name
        if sentence == "quit":
            break

        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    if patterns == intent["patterns"]: 
                        botvalue = f"{random.choice(intent['responses'])}"
                    
                    
        else:
            botvalue = f"I do not understand..."

        return botvalue

st.title("Dummy bot")
form = st.form(key='my-form')
name = form.text_input('Enter input')
submit = form.form_submit_button('Submit')

if name == '/train':
    os.system('C:/Users/DELL/myenv/train.py')
    st.write(trained())
    
    st.write("trained data successful")
elif submit:
	st.write("User: "+ name)
	st.write("Dummy: "+ chat())
else:
    st.write("there is some error in your code")

