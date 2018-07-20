#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn import datasets
from rmsprop_hyperopt_class import RMSPropHyperopt
from sklearn.metrics import confusion_matrix, accuracy_score

# load data
iris = datasets.load_iris()

X = iris.data
y = iris.target


# instance RMSPropHyperopt class
rmsp = RMSPropHyperopt(ramdon_state=44)

# train, test data
x_train, x_test, y_train, y_test = rmsp.train_test_custom(X, 
                                                          y, 
                                                          scaler = False, 
                                                          resample = False, 
                                                          train_size = 0.8)

# set tree parzen algorithm parameters
rmsp.set_hyperopt_params(n_EI_candidates=50,
                         gamma=0.2,
                         n_startup_jobs=20)

# hyperparameters space
params = {
        'learn_rate_min' : 0.01, 
        'learn_rate_max' : 0.5,
        'units_min' : 1,
        'units_max' : 200,
        'dropout_min' : 0,
        'dropout_max' : 0.5, 
        'batch_size_min' : 5,
        'batch_size_max' : 70,
        'nlayers' : 5
        }

# hyperparameter optimization ("best hyperparameters")
best = rmsp.hyperparameter_opt(epochs = 50, 
                               n_iters = 60, 
                               params = params)

# min categorical_crossentropy error 
best_loss = min(rmsp._trials.losses())

# train best hyperparameters
best_net = rmsp.best_nnet_train()

# epochs vs error
rmsp.plot_error()

# categorical_crossentropy error
rmsp.predict_error()

# best net architecture
best_net.architecture()

# precision, recall and f1 score
c_report = rmsp.classification_report_df()

# predict x_train and x_test
y_test_predicted = best_net.predict(x_test).argmax(axis=1)
y_train_predicted = best_net.predict(x_train).argmax(axis=1)

# test and train accuracy
test_accuracy = accuracy_score(y_test.argmax(axis=1), y_test_predicted)
train_accuracy = accuracy_score(y_train.argmax(axis=1), y_train_predicted)

print(test_accuracy)
print(train_accuracy)

# test confusion matrix
print(confusion_matrix(y_test.argmax(axis=1), y_test_predicted))

# train cunfusion matrix
print(confusion_matrix(y_train.argmax(axis=1), y_train_predicted))







