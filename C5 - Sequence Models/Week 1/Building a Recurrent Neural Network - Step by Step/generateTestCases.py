# New Generate Test Cases 
from solutions import *
import numpy as np 
import math 
import os,sys
# import tensorflow as tf
sys.path.append('../')
sys.path.append('../../')

from grader_support import stdout_redirector
from grader_support import util


mFiles = [
    "rnn_cell_forward.py",
    "rnn_forward.py",
    "lstm_cell_forward.py",
    "lstm_forward.py"
]

# generating test cases for rnn_cell_forward 


xt = np.random.randn(3,6)
a_prev = np.random.randn(4,6)
Waa = np.random.randn(4,4)
Wax = np.random.randn(4,3)
Wya = np.random.randn(2,4)
ba = np.random.randn(4,1)
by = np.random.randn(2,1)
parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}

a_next, yt_pred, cache = rnn_cell_forward(xt, a_prev, parameters)

# -------------------------------------------------------

# generating test cases for rnn_forward


x1 = np.random.randn(3,6,4)
a01 = np.random.randn(4,6)
parameters1 = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}
a, y_pred, caches = rnn_forward(x1, a01, parameters1)

# -------------------------------------------------------
# generate test cases for lstm_cell_forward
xt1 = np.random.randn(3,4)
a_prev1 = np.random.randn(5,4)
c_prev1 = np.random.randn(5,4)
Wf = np.random.randn(5, 5+3)
bf = np.random.randn(5,1)
Wi = np.random.randn(5, 5+3)
bi = np.random.randn(5,1)
Wo = np.random.randn(5, 5+3)
bo = np.random.randn(5,1)
Wc = np.random.randn(5, 5+3)
bc = np.random.randn(5,1)
Wy = np.random.randn(2,5)
by = np.random.randn(2,1)
parameters2 = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}
a_next_lstm, c_next_lstm, yt_lstm, cache_lstm = lstm_cell_forward(xt1, a_prev1, c_prev1, parameters2)

# -------------------------------------------------------
# generate test cases for lstm_cell_forward
# lstm_forward

np.random.seed(1)
x2 = np.random.randn(3,10,4)
a02 = np.random.randn(5,10)
Wf = np.random.randn(5, 5+3)
bf = np.random.randn(5,1)
Wi = np.random.randn(5, 5+3)
bi = np.random.randn(5,1)
Wo = np.random.randn(5, 5+3)
bo = np.random.randn(5,1)
Wc = np.random.randn(5, 5+3)
bc = np.random.randn(5,1)
Wy = np.random.randn(2,5)
by = np.random.randn(2,1)

parameters3 = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}

af, yf, cf, cachesf = lstm_forward(x2, a02, parameters3)

def generateTestCases():
	testCases = {
	    'rnn_cell_forward': { 
	        'partId': 'KrqbT',
	        'testCases': [
	            {
	                'testInput': (xt, a_prev, parameters),
	                'testOutput': (a_next, yt_pred, cache)
	            }
	        ]
	    },
	    'rnn_forward': { 
	        'partId': 'CzGAI',
	        'testCases': [
	            {
	                'testInput': (x1, a01, parameters1),
	                'testOutput': (a, y_pred, caches)
	            }
	        ]
	    },
	    'lstm_cell_forward': { 
	        'partId': '7tvdt',
	        'testCases': [
	            {
	                'testInput': (xt1, a_prev1, c_prev1, parameters2),
	                'testOutput': (a_next_lstm, c_next_lstm, yt_lstm, cache_lstm)
	            }
	        ]
	    },
	    'lstm_forward': { 
	        'partId': 'SAQvR',
	        'testCases': [
	            {
	                'testInput': (x2, a02, parameters3),
	                'testOutput': (af, yf, cf, cachesf)
	            }
	        ]
	    }
	}
	return testCases

