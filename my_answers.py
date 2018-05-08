import numpy as np
import string # for problem 2
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import keras


# DONE function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
  
    element_counter = 0
    # cut up series into input/output pairs
    while element_counter + window_size < len(series):
        X.append(series[element_counter:element_counter+window_size])
        y.append(series[element_counter+window_size])
        #X = np.append(X,[series[element_counter:element_counter+window_size]])    
        #y = np.append(y,series[element_counter+window_size])
        element_counter += 1
        
    # reshape each 
    X = np.asarray(X)
    #X.shape = (np.shape(X)[0:2])
    X.shape = (np.shape(X)[0:window_size])
    y = np.asarray(y)
    y.shape = (len(y),1)
    
    return X, y
# DONE: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    lstm_size = 5  #of hidden units
    model = Sequential()
    model.add(LSTM(lstm_size,input_shape = (window_size,1))) #layer 1 LSTM
    #model.add(LSTM(lstm_size,dropout=0.5, unroll=True,input_shape = (window_size,1))) #layer 1 LSTM
    model.add(Dense(1)) #layer 2 fully connected add activation='sigmoid' as second param?
    #things I tried that sucked:
        #1 making the dense layer a sigmoid
        #2 adding dropout .5 AND unroll =true to LSTM results in a much worse result.  Perhaps dropout by itself would help
        #but I didn't try it since I already have an acceptable result
    return model


### DONE: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    remove_these = set(text)
    #the two sets of characters we want to remove from the set above
    punctuation = ['!', ',', '.', ':', ';', '?', ' '] # i added ' ' as an element here since we want to keep spaces, i think
    ascii_lowercase = set(string.ascii_lowercase) #i imported the string module in file header
    # pull out the characters we want to keep in our text
    remove_these = remove_these - set(punctuation) - ascii_lowercase
    # this was the print of what characters we remove...i wasn't too sure about the letters with accent marks
    # per the instructions in the todo for this method, they gotta go though
    #{'1', '2', '4', '7', '$', '8', 'â', '%', '/', '&', "'", '(', 'à', 'é', '-', '3', '*', '0', 'è', '"', '@', '9', '6', ')', '5'}
    for item in remove_these:
        text = text.replace(item,' ')
    return text

### DONE: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    element_counter = 0
    # cut up series into input/output pairs
    while element_counter + window_size < len(text):
        inputs.append(text[element_counter:element_counter+window_size])
        outputs.append(text[element_counter+window_size])
        element_counter += step_size
        
    return inputs,outputs

# DONE build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    lstm_size = 200  #of hidden units
    model = Sequential()
    model.add(LSTM(lstm_size,input_shape = (window_size,num_chars))) #layer 1 LSTM
    #model.add(LSTM(lstm_size,dropout=0.5,input_shape = (window_size,num_chars))) #layer 1 LSTM
    model.add(Dense(num_chars)) #layer 2 fully connected , linear module set hidden units to num_chars
    model.add(Activation('softmax')) #softmax activation
    return model
