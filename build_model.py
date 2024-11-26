import numpy as np
from load import load_data
from utils_main import train, test, build_freqs
import pickle
import os

lr = os.getenv('lr')
num_iters = os.getenv('num_iters')

def build_model():

    train_x, test_x, train_y, test_y = load_data()
    freqs = build_freqs(train_x,train_y)

    X = np.array(train_x)
    Y = np.array(train_y)

    loss, theta = train(X, Y, freqs, np.zeros((3,1)), lr, num_iters)
    accuracy = test(test_x, test_y, freqs, theta)

    with open("theta.pkl", "wb") as file:
        pickle.dump(theta, file)

    with open("freqs.pkl", "wb") as file:
        pickle.dump(freqs, file)

    print("accuracy:", accuracy)

if __name__ == "__main__":
    build_model()