import numpy as np
from utils import *
from preprocess import preprocess

def build_freqs(tweets, ys):
    freqs = {}
    
    for tweet,y in zip(tweets,ys):
        for word in preprocess(tweet):
            pair = (word,int(y))
            if pair in freqs: freqs[pair] += 1
            else: freqs[pair] = 1
    return freqs

def train(x, y, freqs, theta, lr, num_iters):
    m = len(x)
    X = np.zeros((m,3))
    for i in range(m):
        X[i:] = extract_features(x[i], freqs)
    
    loss, theta = gradient_descent(X, y, theta, lr, num_iters)
    return loss, theta

def predict_tweet(tweet, freqs, theta):
    x = extract_features(tweet, freqs)
    y_pred = sigmoid(np.dot(x, theta))
    return y_pred

def test(x, y, freqs, theta):
    y_hat = []
    for tweet in x:
        y_pred = predict_tweet(tweet, freqs, theta)
        if y_pred>0.5: y_hat.append(1)
        else: y_hat.append(0)

    accuracy = ((np.array(y_hat) == np.squeeze(y,-1)).sum())/len(y_hat)
    return accuracy

