# New Generate Test Cases 
from .solutions import *
import numpy as np 
np.random.seed(3)

import math 

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense,  LSTM, Reshape

# import copy 
# from keras.callbacks import History 
# import tensorflow as tf


#from grader_support import stdout_redirector
#from grader_support import util

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

# with suppress_stdout_stderr():
n_a = 64 
n_values = 90 
LSTM_cell = LSTM(n_a, return_state=True) # Used in Step 2.C
densor = Dense(n_values, activation='softmax') # Used in Step 2.D
x_initializer = np.zeros((1, 1, 90))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))
reshapor = Reshape((1, n_values))  

# ================================================================================================
# generating the test cases for dj model 
'''
def djmodel_gen():
	m = 60
	a0 = np.zeros((m, n_a))
	c0 = np.zeros((m, n_a))
	djmodelx = djmodel(Tx = 30 , LSTM_cell=LSTM_cell, densor=densor, reshapor=reshapor)
	opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
	djmodelx.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	cp = djmodelx.count_params()
	ml = len(djmodelx.layers)
	print(cp, ml)
	return (cp, ml)

# ================================================================================================
'''
# GENERATING TEST CASES FOR THE MUSIC INFERENCE MODEL 
'''
def music_inference_model_gen():
	im = music_inference_model(LSTM_cell, densor, Ty=10)
	opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
	im.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	cp1 = im.count_params()
	ml1 = len(im.layers)
	m_out1 = np.asarray((cp1, ml1))
	print(m_out1)
	return m_out1
'''
# ================================================================================================

# generating the test cases for predicted_and_sample

inference_model = music_inference_model(LSTM_cell, densor, 13)
results, indices = predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer)

def generateTestCases():
	testCases = {
	    'djmodel': {
	        'partId': 'iz6sX',
	        'testCases': [
	            {
	                'testInput': (30, LSTM_cell, densor, reshapor),
	                'testOutput': (45530, 36)
	            }
	        ]
	    },
	    'music_inference_model': { 
	        'partId': 'MtuL2',
	        'testCases': [
	            {
	                'testInput': (LSTM_cell, densor, 10),
	                'testOutput': (45530, 32)
	            }
	        ]
	    },
		'predict_and_sample': { 
	        'partId': 'tkaiA',
	        'testCases': [
	            {
	                'testInput': (inference_model, x_initializer, a_initializer, c_initializer),
	                'testOutput': (results, indices)
	            }
	        ]
	    },
	}
	return testCases

