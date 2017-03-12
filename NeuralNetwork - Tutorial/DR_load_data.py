# data loader for the digit recognition

# import third-party libraries
import numpy as np
import pandas as pd

def load_data():
    
    # loading data
    train_data  = pd.read_csv("../samle_data_4000.csv")
    # testdata    = pd.read_csv("../test.csv")
    # commenting this as there is no test data in the location
    
    # selecting training and validation samples using a random split
    train_7 = train_data.sample(frac=0.7)
    val_7   = train_data.loc[~train_data.index.isin(train_7.index)]
    

    # getting the training features and corresponding labels as a tuple
    y_train         = train_7['label'].ravel()
    train_7         = train_7.drop(['label'], axis = 1)
    x_train         = tuple(map(tuple, train_7.values))
    training_data   = zip(x_train, y_train)

    y_validation    = val_7['label'].ravel()
    val_7           = val_7.drop(['label'], axis = 1)
    x_validation    = tuple(map(tuple, val_7.values))
    validation_data = zip(x_validation, y_validation)

    # taking a dummy y_test to have continuity in the code and data
    x_test          = tuple(map(tuple, testdata.values))
    y_test          = testdata['pixel0'].ravel() # a dummy value
    test_data       = zip(x_test, y_test)
    
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)`` in a format that is more convenient for use in our
    implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    
    tr_d, va_d, te_d    = load_data()
    
    training_inputs     = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results    = [vectorized_result(y) for y in tr_d[1]]
    training_data       = zip(training_inputs, training_results)
    # transposing the tuple
    training_data       = zip(*training_data)
    
    validation_inputs   = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data     = zip(validation_inputs, va_d[1])
    validation_data     = zip(*validation_data)
    
    test_inputs         = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data           = zip(test_inputs, te_d[1])
    
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    
    e       = np.zeros((10, 1))
    e[j]    = 1.0
    
    return e
