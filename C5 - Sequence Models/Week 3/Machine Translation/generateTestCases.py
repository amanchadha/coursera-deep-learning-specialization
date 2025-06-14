# New Generate Test Cases 
import numpy as np 
import math 
import os,sys
from testCase import get_testCase
# import copy 
# from keras.callbacks import History 
# import tensorflow as tf
sys.path.append('../')
sys.path.append('../../')

from grader_support import stdout_redirector
from grader_support import util

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

# This grader is for the Emojify assignment

mFiles = [
    "one_step_attention.py",
    "model.py"
]
class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in 
    Python, i.e. will suppress all print, even if the print originates in a 
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).      

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)
np.random.seed(3)
with suppress_stdout_stderr():
    from solutions import *
    from testCase import get_testCase
    n_a = 64
    n_s = 128
    m = 10
    dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)

    human_vocab_size = len(human_vocab)
    machine_vocab_size = len(machine_vocab)

    im = model(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size)
    cp1 = im.count_params()
    mi1 = len(im.inputs)
    mo1 = len(im.outputs)
    ml1 = len(im.layers)
    m_out1 = np.asarray((cp1, mi1, mo1, ml1))


    # GRADED FUNCTION: one_step_attention

    m_out2 = get_testCase()

def generateTestCases():
	testCases = {
	    'one_step_attention': {
	        'partId': 'zcQIs',
	        'testCases': [
	            {
	                'testInput': 0,
	                'testOutput': m_out2
	            }
	        ]
	    },
	    'model': { 
	        'partId': 'PTKef',
	        'testCases': [
	            {
	                'testInput': (Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size),
	                'testOutput': m_out1
	            }
	        ]
	    }
       }
	return testCases

