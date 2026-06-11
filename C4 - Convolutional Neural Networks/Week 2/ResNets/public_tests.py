from termcolor import colored
import tensorflow as tf
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity
import numpy as np

def identity_block_test(target):
    np.random.seed(1)
    #X = np.random.randn(3, 4, 4, 6).astype(np.float32)
    X1 = np.ones((1, 4, 4, 3)) * -1
    X2 = np.ones((1, 4, 4, 3)) * 1
    X3 = np.ones((1, 4, 4, 3)) * 3

    X = np.concatenate((X1, X2, X3), axis = 0).astype(np.float32)

    A3 = target(X,
                f = 2,
                filters = [4, 4, 3],
                initializer=lambda seed=0:constant(value=1),
                training=False)


    A3np = A3.numpy()
    assert tuple(A3np.shape) == (3, 4, 4, 3), "Shapes does not match. This is really weird"
    assert np.all(A3np >= 0), "The ReLu activation at the last layer is missing"
    resume = A3np[:,(0,-1),:,:].mean(axis = 3)

    assert np.floor(resume[1, 0, 0]) == 2 * np.floor(resume[1, 0, 3]), "Check the padding and strides"
    assert np.floor(resume[1, 0, 3]) == np.floor(resume[1, 1, 0]),     "Check the padding and strides"
    assert np.floor(resume[1, 1, 0]) == 2 * np.floor(resume[1, 1, 3]), "Check the padding and strides"
    assert np.floor(resume[1, 1, 0]) == 2 * np.floor(resume[1, 1, 3]), "Check the padding and strides"

    assert resume[1, 1, 0] - np.floor(resume[1, 1, 0]) > 0.7, "Looks like the BatchNormalization units are not working"

    assert np.allclose(resume, 
                       np.array([[[0.0,       0.0,       0.0,        0.0],
                                  [0.0,       0.0,       0.0,        0.0]],
                                 [[192.71236, 192.71236, 192.71236,  96.85619],
                                  [ 96.85619,  96.85619,  96.85619,  48.9281 ]],
                                 [[578.1371,   578.1371,  578.1371,  290.56854],
                                  [290.56854,  290.56854, 290.56854, 146.78427]]]), atol = 1e-5 ), "Wrong values with training=False"
    
    np.random.seed(1)
    A4 = target(X,
                f = 3,
                filters = [3, 3, 3],
                initializer=lambda seed=7:constant(value=1),
                training=True)
    A4np = A4.numpy()
    resume = A4np[:,(0,-1),:,:].mean(axis = 3)
    assert np.allclose(resume, 
                         np.array([[[0.,         0.,        0.,      0.,        ],
                                  [0.,         0.,        0.,        0.,        ]],
                                 [[0.37394285, 0.37394285, 0.37394285, 0.37394285],
                                  [0.37394285, 0.37394285, 0.37394285, 0.37394285]],
                                 [[3.2379014,  4.1394243,  4.1394243,  3.2379014 ],
                                  [3.2379014,  4.1394243,  4.1394243,  3.2379014 ]]]), atol = 1e-5 ), "Wrong values with training=True"

    print(colored("All tests passed!", "green"))
