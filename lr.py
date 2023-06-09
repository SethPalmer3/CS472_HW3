#!/usr/bin/python
#
# CIS 472/572 - Logistic Regression Template Code
#
# Author: Daniel Lowd <lowd@cs.uoregon.edu>
# Date:   2/9/2018
#
# Please use this code as the template for your solution.
#
import sys
import re
from math import log
from math import exp
from math import sqrt

MAX_ITERS = 100


# Load data from a file
def read_data(filename):
    f = open(filename, 'r')
    p = re.compile(',')
    data = []
    header = f.readline().strip()
    varnames = p.split(header)
    namehash = {}
    for l in f:
        example = [int(x) for x in p.split(l.strip())]
        x = example[0:-1]
        y = example[-1]
        data.append((x, y))
    return (data, varnames)

def logistic(n): # Logistic function
    if n < -200:
        return 0.0
    return 1 / (1 + exp(-n))

def logit(model, x):
    (w,b) = model
    h = 0.0
    for i in range(len(x)):
        h+=w[i]*x[i]
    h+=b
    return h

def log_err(y_true, y_pred):
    return logistic(-y_true*y_pred)

def dot(w,z):
    assert len(w) == len(z)
    sm = 0.0
    for i in range(len(w)):
        sm += w[i]*z[i]
    return sm


# Train a logistic regression model using batch gradient descent
'''
eta = learning rate
l2_reg_weight = lambda = L2 normaliztion scale
'''
def train_lr(data, eta, l2_reg_weight):
    numvars = len(data[0][0])
    numexamples = len(data)
    w = [0.0] * numvars
    b = 0.0

    for _ in range(MAX_ITERS):
        pred_vect = [0.0] * numexamples
        for d in range(numexamples):
            # place computation in pred_vect
            pred_vect[d] = logit((w,b),data[d][0])
        # compute the error y - y_hat
        error = [log_err(data[x][1],pred_vect[x]) for x in range(numexamples)]
        # compute gradient w/ weight regulation dot(train_feat.T, error) / len(data) + l2_reg_weight / len(data) * weights
        g_w = [0.0] *len(w)
        g_b = 0.0
        for p in range(numvars): # run for each weight : w_i
            for e in range(numexamples): # iterate down the examples
                g_w[p] += -data[e][1] * error[e] * data[e][0][p]
            g_w[p] += (l2_reg_weight * w[p])
        for e in range(numexamples):
            g_b += -data[e][1] * error[e]

        # adjust weights : w -= eta * gradient
        for wu in range(numvars):
            w[wu] -= eta * g_w[wu] 
        b -= eta * g_b

    return (w, b)


# Predict the probability of the positive label (y=+1) given the
# attributes, x.
def predict_lr(model, x):
    s = logit(model,x)
    return s

# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
    if (len(argv) != 5):
        print('Usage: lr.py <train> <test> <eta> <lambda> <model>')
        sys.exit(2)
    (train, varnames) = read_data(argv[0])
    (test, testvarnames) = read_data(argv[1])
    eta = float(argv[2])
    lam = float(argv[3])
    modelfile = argv[4]

    # Train model
    (w, b) = train_lr(train, eta, lam)

    # Write model file
    f = open(modelfile, "w+")
    f.write('%f\n' % b)
    for i in range(len(w)):
        f.write('%s %f\n' % (varnames[i], w[i]))

    # Make predictions, compute accuracy
    correct = 0
    for (x, y) in test:
        prob = predict_lr((w, b), x)
        print(prob)
        if (prob - 0.5) * y > 0:
            correct += 1
    acc = float(correct) / len(test)
    print("Accuracy: ", acc)


if __name__ == "__main__":
    main(sys.argv[1:])
