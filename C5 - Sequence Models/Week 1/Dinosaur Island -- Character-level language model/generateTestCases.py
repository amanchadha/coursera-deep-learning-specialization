# New Generate Test Cases 
from solutions import *
import numpy as np 
import math 
import os,sys
import copy 
from random import shuffle
# import tensorflow as tf
sys.path.append('../')
sys.path.append('../../')

from grader_support import stdout_redirector
from grader_support import util


mFiles = [
    "clip.py",
    "sample.py",
    "optimize.py",
    "model.py"
]


data = open('dinos.txt', 'r').read()
data= data.lower()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
char_to_ix = { ch:i for i,ch in enumerate(sorted(chars)) }
ix_to_char = { i:ch for i,ch in enumerate(sorted(chars)) }


# set the seed to be able to replicate the same results. 
np.random.seed(3)

dWax = np.random.randn(5,3)*10
dWaa = np.random.randn(5,5)*10
dWya = np.random.randn(2,5)*10
db = np.random.randn(5,1)*10
dby = np.random.randn(2,1)*10
gradients = {"dWax": dWax, "dWaa": dWaa, "dWya": dWya, "db": db, "dby": dby}
gradients1 = copy.deepcopy(gradients)
gradients = clip(gradients, 10)

# generating test cases for sampling function
vocab_size = 27
n = 23
n_a = 50
a0 = np.random.randn(n_a, 1) * 0.2
i0 = 1 # first character is ix_to_char[i0]
Wax = np.random.randn(n_a, vocab_size)
Waa = np.random.randn(n_a, n_a)
Wya = np.random.randn(vocab_size, n_a)
b = np.random.randn(n_a, 1)
by = np.random.randn(vocab_size, 1)
parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
indexes = sample(parameters, char_to_ix, 0)

# # generating test cases for optimize function
vocab_size = 27
n_a = 50 
a_prev = np.random.randn(n_a, 1) * 0.2
Wax = np.random.randn(n_a, vocab_size) * 0.4
Waa = np.random.randn(n_a, n_a)
Wya = np.random.randn(vocab_size, n_a)
b = np.random.randn(n_a, 1)
by = np.random.randn(vocab_size, 1)
parameters2 = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
parameters3 = copy.deepcopy(parameters2)
X = [12,3,5,11,22,3]
Y = [4,14,11,22,25, 26]
loss, g, a_last = optimize(X, Y, a_prev, parameters2, learning_rate = 0.01)

# generating the model. Killing the print statements.
with stdout_redirector.stdout_redirected():
	# generating the model
	with open("dinos.txt") as f:
		examples = f.readlines()
		np.random.seed(0)
		np.random.shuffle(examples)
		a = model(examples, ix_to_char, char_to_ix, 200)

def generateTestCases():
	testCases = {
	    'clip': { 
	        'partId': 'sYLqC',
	        'testCases': [
	            {
	                'testInput': (gradients1, 10),
	                'testOutput': gradients
	            }
	        ]
	    },
	    'sample': { 
	        'partId': 'QxiNo',
	        'testCases': [
	            {
	                'testInput': (parameters, char_to_ix, 0),
	                'testOutput': indexes
	            }
	        ]
	    },
	    'optimize': { 
	        'partId': 'x2pxm',
	        'testCases': [
	            {
	                'testInput': (X, Y, a_prev, parameters3),
	                'testOutput': (loss, g, a_last)
	            }
	        ]
	    },
	    'model': { 
	        'partId': 'mJTOb',
	        'testCases': [
	            {
	                'testInput': (examples, ix_to_char, char_to_ix, 200),
	                'testOutput': a
	            }
	        ]
	    }
	}
	return testCases

