# New Generate Test Cases 
import numpy as np 
import math 
import os,sys
sys.path.append('../')
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from td_utils import *
import warnings
warnings.filterwarnings("ignore")

from pydub import AudioSegment

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

with suppress_stdout_stderr():
  from solutions import *

# import copy 
# from keras.callbacks import History 
# import tensorflow as tf
sys.path.append('../../')

from grader_support import stdout_redirector
from grader_support import util




# This grader is for the Emojify assignment

mFiles = [
    "is_overlapping.py",
    "insert_audio_clip.py",
    "insert_ones.py",
    "create_training_example.py",
    "model.py"
]

np.random.seed(3)

# generating the testCases for is_overlapping
overlap1 = is_overlapping((900, 2500), [(2000, 2550), (260, 949)])
overlap2 = is_overlapping((2306, 2307), [(824, 1532), (1900, 2305), (3424, 3656)])

# generating the test cases for the insert_audio_clip
# (2732, 3452), (4859, 5579)
a = AudioSegment.from_wav("activate.wav")
b = AudioSegment.from_wav("background.wav")
audio_clip, segment_time = insert_audio_clip(b, a, [(3790, 4400)])
audio_clip.export('test.wav', format = 'wav')

inserted = graph_spectrogram('test.wav')

# generate the testCases for insert_ones
arr1 = insert_ones(np.zeros((1, Ty)), 9)

# generate the test Cases for create_training_example

n = AudioSegment.from_wav("negative.wav")

A = []
N = []
A.append(a)
N.append(n)

with stdout_redirector.stdout_redirected():
    a, s = create_training_example(b, A, N)


# generating the test cases for the model 
with suppress_stdout_stderr():
    model = model(input_shape = (Tx, n_freq))
    ml = len(model.layers)
    cp = model.count_params()
    mi = len(model.inputs)
    mo = len(model.outputs)

def generateTestCases():
	testCases = {
	    'is_overlapping': {
	        'partId': 'S8DvY',
	        'testCases': [
	            {
	                'testInput': ((900, 2500), [(2000, 2550), (260, 949)]),
	                'testOutput': overlap1
	            },
                {
                    'testInput': ((2306, 2307), [(824, 1532), (1900, 2305), (3424, 3656)]),
                    'testOutput': overlap2
                }
	        ]
	    },
	    'insert_audio_clip': { 
	        'partId': 'BSIWi',
	        'testCases': [
	            {
	                'testInput': ("activate.wav", "background.wav"),
	                'testOutput': inserted
	            }
	        ]
	    },
	    'insert_ones': { 
	        'partId': '2Kdnr',
	        'testCases': [
	            {
	                'testInput': (np.zeros((1, Ty)), 9) ,
	                'testOutput': arr1
	            }
	        ]
	    },
	    'create_training_example': { 
	        'partId': 'v097u',
	        'testCases': [
	            {
	                'testInput': (b, A, N),
	                'testOutput': (a,s)
	            }
	        ]
	    },
      'model': { 
          'partId': '0Txcd',
          'testCases': [
              {
                  'testInput': (Tx, n_freq),
                  'testOutput': np.asarray([cp, ml, mi, mo])
              }
          ]
       }
       }
	return testCases

