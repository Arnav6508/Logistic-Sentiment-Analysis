import pickle
import numpy as np
from utils_main import predict_tweet

def test_one_tweet(tweet):

    with open("theta.pkl", "rb") as file:
        theta = pickle.load(file)

    with open("freqs.pkl", "rb") as file:
        freqs = pickle.load(file)

    y_pred = predict_tweet(tweet, freqs, theta)

    if y_pred>0.5: print('Positive sentiment')
    else: print('Negative sentiment')
