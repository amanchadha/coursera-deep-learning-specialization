def inference_model(LSTM_cell, densor, n_x = 90, n_a = 64, Ty = 100):
    """
    Uses the trained "LSTM_cell" and "densor" from model() to generate a sequence of values.
    
    Arguments:
    LSTM_cell -- the trained "LSTM_cell" from model(), Keras layer object
    densor -- the trained "densor" from model(), Keras layer object
    n_x -- number of unique values
    n_a -- number of units in the LSTM_cell
    Ty -- number of time steps to generate
    
    Returns:
    inference_model -- Keras model instance
    """
    
    # Define the input of your model with a shape 
    x0 = Input(shape=(1, n_x))
    
    # Define s0, initial hidden state for the decoder LSTM
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    x = x0

    ### START CODE HERE ###
    # Step 1: Create an empty list of "outputs" to later store your predicted values (≈1 line)
    outputs = []
    
    # Step 2: Loop over Ty and generate a value at every time step
    for t in range(Ty):
        
        # Step 2.A: Perform one step of LSTM_cell (≈1 line)
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        
        # Step 2.B: Apply Dense layer to the hidden state output of the LSTM_cell (≈1 line)
        out = densor(a)

        # Step 2.C: Append the prediction "out" to "outputs" (≈1 line)
        outputs.append(out)
        
        # Step 2.D: Set the prediction "out" to be the next input "x". You will need to use RepeatVector(1). (≈1 line)
        x = RepeatVector(1)(out)
        
    # Step 3: Create model instance with the correct "inputs" and "outputs" (≈1 line)
    inference_model = Model(inputs=[x0, a0, c0], outputs=outputs)
    ### END CODE HERE ###
    
    return inference_model


inference_model = inference_model(LSTM_cell, densor)


x1 = np.zeros((1, 1, 90))
x1[:,:,35] = 1
a1 = np.zeros((1, n_a))
c1 = np.zeros((1, n_a))
predicting = inference_model.predict([x1, a1, c1])


indices = np.argmax(predicting, axis = -1)
results = to_categorical(indices, num_classes=90)