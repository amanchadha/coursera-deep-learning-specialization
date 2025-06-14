from __future__ import print_function
import tensorflow as tf
#import tensorflow.keras.backend as K
from tensorflow.keras.layers import RepeatVector
import sys
from music21 import *
import numpy as np
from grammar import *
from preprocess import *
from qa import *


def data_processing(corpus, values_indices, m = 60, Tx = 30):
    # cut the corpus into semi-redundant sequences of Tx values
    Tx = Tx 
    N_values = len(set(corpus))
    np.random.seed(0)
    X = np.zeros((m, Tx, N_values), dtype=np.bool)
    Y = np.zeros((m, Tx, N_values), dtype=np.bool)
    for i in range(m):
#         for t in range(1, Tx):
        random_idx = np.random.choice(len(corpus) - Tx)
        corp_data = corpus[random_idx:(random_idx + Tx)]
        for j in range(Tx):
            idx = values_indices[corp_data[j]]
            if j != 0:
                X[i, j, idx] = 1
                Y[i, j-1, idx] = 1
    
    Y = np.swapaxes(Y,0,1)
    Y = Y.tolist()
    return np.asarray(X), np.asarray(Y), N_values 

def next_value_processing(model, next_value, x, predict_and_sample, indices_values, abstract_grammars, duration, max_tries = 1000, temperature = 0.5):
    """
    Helper function to fix the first value.
    
    Arguments:
    next_value -- predicted and sampled value, index between 0 and 77
    x -- numpy-array, one-hot encoding of next_value
    predict_and_sample -- predict function
    indices_values -- a python dictionary mapping indices (0-77) into their corresponding unique value (ex: A,0.250,< m2,P-4 >)
    abstract_grammars -- list of grammars, on element can be: 'S,0.250,<m2,P-4> C,0.250,<P4,m-2> A,0.250,<P4,m-2>'
    duration -- scalar, index of the loop in the parent function
    max_tries -- Maximum numbers of time trying to fix the value
    
    Returns:
    next_value -- process predicted value
    """

    # fix first note: must not have < > and not be a rest
    if (duration < 0.00001):
        tries = 0
        while (next_value.split(',')[0] == 'R' or 
            len(next_value.split(',')) != 2):
            # give up after 1000 tries; random from input's first notes
            if tries >= max_tries:
                #print('Gave up on first note generation after', max_tries, 'tries')
                # np.random is exclusive to high
                rand = np.random.randint(0, len(abstract_grammars))
                next_value = abstract_grammars[rand].split(' ')[0]
            else:
                next_value = predict_and_sample(model, x, indices_values, temperature)

            tries += 1
            
    return next_value


def sequence_to_matrix(sequence, values_indices):
    """
    Convert a sequence (slice of the corpus) into a matrix (numpy) of one-hot vectors corresponding 
    to indices in values_indices
    
    Arguments:
    sequence -- python list
    
    Returns:
    x -- numpy-array of one-hot vectors 
    """
    sequence_len = len(sequence)
    x = np.zeros((1, sequence_len, len(values_indices)))
    for t, value in enumerate(sequence):
        if (not value in values_indices): print(value)
        x[0, t, values_indices[value]] = 1.
    return x
