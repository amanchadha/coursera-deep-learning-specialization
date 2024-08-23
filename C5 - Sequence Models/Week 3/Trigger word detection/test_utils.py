from termcolor import colored

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Conv1D

# Compare the two inputs
def comparator(learner, instructor):
    layer = 0
    for a, b in zip(learner, instructor):
        if tuple(a) != tuple(b):
            print(colored("Test failed", attrs=['bold']),
                  f"at layer: {layer}",
                  "\n Expected value \n\n", colored(f"{b}", "green"), 
                  "\n\n does not match the input value: \n\n", 
                  colored(f"{a}", "red"))
            raise AssertionError("Error in test") 
        layer += 1
    print(colored("All tests passed!", "green"))

# extracts the description of a given model
def summary(model):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    result = []
    for layer in model.layers:
        descriptors = [layer.__class__.__name__, layer.output_shape, layer.count_params()]
        if (type(layer) == Conv1D):
            descriptors.append(layer.padding)
            descriptors.append(layer.activation.__name__)
            descriptors.append(layer.strides)
            descriptors.append(layer.kernel_size)
            descriptors.append(layer.kernel_initializer.__class__.__name__)
           
        if (type(layer) == Conv2D):
            descriptors.append(layer.padding)
            descriptors.append(layer.activation.__name__)
            descriptors.append(layer.kernel_initializer.__class__.__name__)
            
        if (type(layer) == MaxPooling2D):
            descriptors.append(layer.pool_size)
            descriptors.append(layer.strides)
            descriptors.append(layer.padding)
            
        if (type(layer) == Dropout):
            descriptors.append(layer.rate)
            
        if (type(layer) == ZeroPadding2D):
            descriptors.append(layer.padding)
            
        if (type(layer) == Dense):
            descriptors.append(layer.activation.__name__)
            
        if (type(layer) == LSTM):
            descriptors.append(layer.input_shape)
            descriptors.append(layer.activation.__name__)
            
        if (type(layer) == RepeatVector):
            descriptors.append(layer.n)
            
        if (type(layer) == TimeDistributed):
            descriptors.append(layer.layer.activation.__name__)  
            
        if (type(layer) == GRU):
            descriptors.append(layer.return_sequences)    
            
        result.append(descriptors)
    return result