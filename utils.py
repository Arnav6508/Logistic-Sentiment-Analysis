import numpy as np
from preprocess import preprocess

def sigmoid(z):
    return 1/(1+np.exp(-z))

def loss(y,h):
    m = y.shape[0]
    return -(1/m)*(np.dot(y.transpose(),np.log(h))+np.dot((1-y).transpose(),np.log(1-h)))

def differential(y,x,h):
    m = y.shape[0]
    return np.dot(x.transpose(),h-y)/m

def gradient_descent(x, y, theta, lr, num_iters):
    for i in range(num_iters):
        z = np.dot(x, theta)
        h = sigmoid(z)
        j = loss(y,h)
        theta = theta - lr*differential(y,x,h)
        if(i%100 == 0): print(j)
    return float(j),theta

def extract_features(tweet, freqs):
    tweet = preprocess(tweet)

    feature = np.zeros(3) # bias, pos, neg
    feature[0] = 1
    
    for word in tweet:
        feature[1] += freqs.get((word,1),0)
        feature[2] += freqs.get((word,0),0)

    feature = np.expand_dims(feature,axis = 0)
    return feature
