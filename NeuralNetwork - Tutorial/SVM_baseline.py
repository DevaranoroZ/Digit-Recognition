# Running SVM for digit recognition to establish a base-line

# importing libraries
import time # to compute the time taken to run the fit

# improting third party libraries
from sklearn import svm

# importing my libraries for loading data
import DR_load_data as data_loader



def svm_baseline():
    
    # start time:
    t = time.time()

    training_data, validation_data, test_data = data_loader.load_data()
    print "Data loaded. Time taken : %s." % (time.time() - t)
    
    t = time.time()

    # train

    clf = svm.SVC()
    clf.fit(training_data[0], training_data[1])
    print "Fit completed. Time taken : %s." % (time.time() - t)

    # test
    
    predictions = [int(a) for a in clf.predict(validation_data[0])]
    num_correct = sum(int(a == y) for a, y in zip(predictions, validation_data[1]))
    

    print "Baseline classifier using an SVM. Time taken : %s." % (time.time() - t)
    print "%s of %s values correct." % (num_correct, len(validation_data[1]))
    
    # returning clf to test the validations outside
    return clf
