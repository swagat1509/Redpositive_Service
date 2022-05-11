from flask import Flask, render_template, request
import re
import torch
import pickle
import numpy as np
import pandas as pd
import network
from network import LSTMNetwork
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# define function for saving a file
def save_file(name, obj):
    """
    Function to save an object as pickle file
    """
    with open(name, 'wb') as f:
        pickle.dump(obj, f)

# define function for loading a file
def load_file(name):
    """
    Function to load a pickle object
    """
    return pickle.load(open(name, "rb"))



app = Flask(__name__)


@app.route('/')
def hello_word():
    return render_template('home.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        text = request.form['Enter Text']
        input_text = text.lower()
        input_text = re.sub(r"[^\w\d'\s]+", " ", input_text)
        input_text = re.sub("\d+", "", input_text)
        input_text = re.sub(r'[x]{2,}', "", input_text)
        input_text = re.sub(' +', ' ', input_text)
        tokens = word_tokenize(input_text)
        tokens = ['<pad>']*(500-len(tokens))+tokens
        idx_token = []
        vocabulary = load_file("files/vocabulary.pkl")
        embeddings = load_file("files/embeddings.pkl")
        for token in tokens:
            if token in vocabulary:
                idx_token.append(vocabulary.index(token))
            else:
                idx_token.append(vocabulary.index('<unk>'))

        token_emb = embeddings[idx_token,:]
        inp = torch.from_numpy(token_emb)
        inp = torch.unsqueeze(inp, 0)
        label_encoder = load_file("files/label_encoder.pkl")
        num_classes = len(label_encoder.classes_)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        input_size = 100
        
        hidden_size = 50
        # Create model object
        model = LSTMNetwork(input_size, hidden_size, num_classes)

        # Load trained weights
        lstm_model_path = 'model/lstm_model.pkl'
        model.load_state_dict(torch.load(lstm_model_path,map_location=torch.device('cpu')))

        # Move the model to GPU if available
        # if torch.cuda.is_available():
        #     model = model.cuda()
            
        # Forward pass
        out = torch.squeeze(model(inp))

        # Find predicted class
        prediction = label_encoder.classes_[torch.argmax(out)]
        print(f"Predicted  Class: {prediction}")





    return render_template('home.html',passtext = prediction)


if __name__ == '__main__':
    app.run(debug = True, port=8000)