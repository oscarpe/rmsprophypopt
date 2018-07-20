#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 14:35:36 2018

@author: oscarp
"""

from pprint import pprint

import theano
import numpy as np
import pandas as pd
from hyperopt import hp
from neupy import algorithms, layers, plots
from neupy.exceptions import StopTraining
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from hyperopt import tpe, fmin, Trials
from functools import partial



theano.config.floatX = 'float32'

# Treshold Error Stop
def on_epoch_end(network):
    if network.errors.last() > 10:
        raise StopTraining("Training was interrupted. Error is to high.")

###################### Prior Custom Functions #############################
    
# Uniform int Prior
def uniform_int(name, lower, upper):
# `quniform` returns:
# round(uniform(low, high) / q) * q
    return hp.quniform(name, lower, upper, q=1)

# Log Uniform int Prior 
def loguniform_int(name, lower, upper):
    # Do not forget to make a logarithm for the
    # lower and upper bounds.
    return hp.qloguniform(name, np.log(lower), np.log(upper), q=1)

#########################  RMSPropHyperopt class #############################

class RMSPropHyperopt():
    
    # Initializer
    def __init__(self,  
                 ramdon_state = 44):
        
        self._ramdon_state = ramdon_state
        
    
    
    ####################### Prior Initialization #############################
    
    # Paremeters generations
    # Default Values 
    def gen_params(self, 
                   learn_rate_min = 0.01, 
                   learn_rate_max = 0.5,
                   units_min = 1,
                   units_max = 200,
                   dropout_min = 0,
                   dropout_max = 0.5, 
                   batch_size_min = 5,
                   batch_size_max = 70,
                   nlayers = 10):


        parameter_space = {
            'step': hp.uniform('step', learn_rate_min, learn_rate_max),
            'layers': hp.choice('layers', 
                                [{'n_layers': i, 
                                  'n_units_layer': 
                                      [hp.choice('act_func_type_'+str(i)+str(j), [layers.Relu, layers.PRelu, layers.Elu, layers.Tanh, layers.Sigmoid])(uniform_int('n_units_layer_'+str(i)+str(j), units_min, units_max)) for j in (np.arange(i)+1)]} for i in (np.arange(nlayers) + 1)]),
            
            'dropout': hp.uniform('dropout', dropout_min, dropout_max),
            'batch_size': loguniform_int('batch_size', batch_size_min, batch_size_max),
        }
        
        return parameter_space

    ##################### Train Network constructor ##########################
    
    def gennetwork_constructor(self, 
                               input_units = None, 
                               softmax_resp = None,
                               epochs = 50):
        
        """Build a train network function
            
        Input arguments:
            input_layers -- features size
            softmax_resp -- labels class response size
            epochs -- epochs number
            
        Output:
            train_network -- function to minimize using hyperopt
        """
        x_train = self._x_train 
        y_train = self._y_train
        x_test = self._x_test, 
        y_test = self._y_test
        
        
        # Output function using RMSProp algorithm
        def train_network(parameters):
            print("Parameters:")
            pprint(parameters)
            print()
            
            # Learning rate
            step = parameters['step']
            
            # mini batch size
            batch_size = int(parameters['batch_size'])
            
            # Dropout probability
            proba = parameters['dropout']
            
            ################# Network Structure ##############################
            
            # Input Layer
            network = layers.Input(input_units)
            
            # Adds Hidden layers with their respective activation function
            for act_fun in parameters['layers']['n_units_layer']:
                network = network > act_fun
            
            # Adds Dropout layer for overfitting control
            # Adds Final Layer Softmax function
            network = network > layers.Dropout(proba) > layers.Softmax(softmax_resp)
            
            
            ################# RMSProp Algorithm ##############################
            
            # set seed
            np.random.seed(self._ramdon_state)
            
            # RMSProp Algorithm
            mnet = algorithms.RMSProp(
                network,
        
                batch_size=batch_size,
                step=step,
                
                error='categorical_crossentropy',
                shuffle_data=True,
                
                epoch_end_signal=on_epoch_end,
            )
            
            ################### Train Network #################################
            
            # Train Network
            mnet.train(x_train, y_train, epochs=epochs)
            
            # categorical crossentropy error
            score = mnet.prediction_error(x_test, y_test)
            
            # test prediction
            y_predicted = mnet.predict(x_test).argmax(axis=1)
            
            # accuracy (sklearn)
            accuracy = metrics.accuracy_score(y_test.argmax(axis=1), y_predicted)
            
            print("Final score: {}".format(score))
            print("Accuracy: {:.2%}".format(accuracy))
            print("================================================")
            
            return score
    
        return train_network
    
    def train_test_custom(self, 
                          X, 
                          y, 
                          scaler = False, 
                          resample = False, 
                          train_size = 0.95):
        
        """train test data: Uses sklearn and imblearn
            
        Input arguments:
            X -- features np array  
            y -- response np array (n,)
            scaler -- use MinMaxScaler (bool)
            resample -- use SMOTE resample method (bool)
            train_size -- float in (0,1)
            
        Output:
            train_network -- function to minimize using hyperopt
        """
        
        
        np.random.seed(self._ramdon_state)
        self._size_labels = len(pd.Series(y).value_counts())
        
        # tranform labes to dummy vars
        target_scaler = OneHotEncoder()
        target = y.reshape((-1, 1))
        target = target_scaler.fit_transform(target).todense()
        
        # scale data
        if scaler:
            mm_scaler = MinMaxScaler()
            data = mm_scaler.fit_transform(X)
        else:
            data = X
            
        x_train, x_test, y_train, y_test = train_test_split(
                data.astype(np.float32),
                target.astype(np.float32),
                train_size= train_size, 
                stratify = y)
        
        # resample data for imbalance
        if resample:
            y_train = np.squeeze(np.asarray(y_train.argmax(axis = 1)))
    
            # SMOTE resample
            sm = SMOTE(random_state=44) 
    
            x_train, y_res = sm.fit_sample(x_train, y_train)
        
        
            # rebuild y train
            y_train = y_res.reshape((-1, 1))
            y_train = target_scaler.fit_transform(y_train).todense()
        
        self._x_train = x_train
        self._x_test = x_test
        self._y_train = y_train
        self._y_test = y_test
        
        return x_train, x_test, y_train, y_test 
    

    def set_hyperopt_params(self, 
                            n_EI_candidates=50,
                            gamma=0.2,
                            n_startup_jobs=20):
        
        """set parameters to tree parzen algorithm
            
        Input arguments:
            n_EI_candidates=50 -- Sample n_EI_candidates candidate and select 
                                  candidate that has highest Expected 
                                  Improvement (EI) (int)   
            gamma -- Use gamma * 100 % of best observations to estimate next 
                     set of parameters  (float in (0,1))
            n_startup_jobs -- First n_startup_jobs trials are going to be random
            
        """
        
        # set methodology
        self._tree = partial(
                # methodology
                tpe.suggest,
                n_EI_candidates=n_EI_candidates,
                gamma=0.2,
                n_startup_jobs=n_startup_jobs,
        )
    
    
    def hyperparameter_opt(self, epochs = 50, n_iters = 60, params = {}):
        
        """Running hyperparameter optimization
            
        Input arguments:
            X -- features np array  
            y -- response np array (n,)
            scaler -- use MinMaxScaler (bool)
            resample -- use SMOTE resample method (bool)
            train_size -- 
            
        Output:
            best -- best hyperparameters values (dict) 
        """
        
        
        input_units = self._x_train.shape[1]
        softmax_resp =  self._size_labels
        
        self._input_units = input_units
        self._softmax_resp = softmax_resp
        self._epochs = epochs
        
        parameter_space = self.gen_params(**params)

        
        train_network = self.gennetwork_constructor(input_units = input_units,
                                                    softmax_resp = softmax_resp,
                                                    epochs=epochs)
        
        # Init trials object
        trials = Trials()
        self._trials = trials
        
        # Optimizer
        best = fmin(
            train_network,
            trials=trials,
            space=parameter_space,
        
            # Set up TPE for hyperparameter optimization
            algo=self._tree,
        
            # Maximum number of iterations. Basically it trains at
            # most 200 networks before choose the best one.
            max_evals=n_iters,
            # set seed
            rstate= np.random.RandomState(self._ramdon_state),
        )
        
        print("=========================================")
        tdiff = trials.trials[-1]['book_time'] - trials.trials[0]['book_time']
        print("ELAPSED TIME: " + str(tdiff.total_seconds() / 60))
        print("====================================u=====")
        
        self._best = best
        
        return best

    def create_layer(self, tlayer):
        
        """return activation function 
            
        Input arguments:
            tlayer -- id of layer activation function  
            
        Output:
            layers.act_fun -- layer activation function 
        """
        
        if tlayer == 0:
            return layers.Relu
        elif tlayer == 1:
            return layers.PRelu
        elif tlayer == 2:
            return layers.Elu
        elif tlayer == 3:
            return layers.Tanh
        elif tlayer == 4:
            return layers.Sigmoid

            
    def create_network(self):
        
        network = layers.Input(self._input_units)
        
        best = self._best
        
        nlayers = best['layers'] + 1
        nunits = [int(best['n_units_layer_' + str(nlayers) + str(i)]) for i in (np.arange(nlayers) + 1)]
        act_fun = [best['act_func_type_' + str(nlayers) + str(i)] for i in (np.arange(nlayers) + 1)]
        
        arch = zip(nunits, act_fun)
         
        for x,y in arch:
            layer_temp = self.create_layer(y)
            network = network > layer_temp(x)
        
        network = network > layers.Dropout(best['dropout']) > layers.Softmax(self._softmax_resp)
        
        return network

    def best_nnet_train(self):
        
        best = self._best
        network = self.create_network()
        
        # set seed
        np.random.seed(self._ramdon_state)
        
        
        bnet = algorithms.RMSProp(
                network,
                batch_size=int(best['batch_size']),
                step=best['step'],
                error='categorical_crossentropy',
                shuffle_data=True,
                verbose=True,       
                epoch_end_signal=on_epoch_end,
                )
        
        bnet.architecture()
        
        bnet.train(self._x_train, 
                   self._y_train, 
                   self._x_test, 
                   self._y_test, 
                   epochs=self._epochs)
        
        self._bnet = bnet
        
        return bnet
    
    def plot_error(self):
        plots.error_plot(self._bnet)

    def predict_error(self):
        return self._bnet.prediction_error(self._x_test, self._y_test)
  
    def classification_report_df(self):
        y_test_predicted = self._bnet.predict(self._x_test).argmax(axis=1)
        report = metrics.classification_report(self._y_test.argmax(axis=1), y_test_predicted)
        
        report_data = []
        lines = report.split('\n')
        for line in lines[2:-3]:
            row = {}
            row_data = line.split('      ')[1:]
            row_data = [a.strip() for a in row_data]
            row['class'] = int(row_data[0])
            row['precision'] = float(row_data[1])
            row['recall'] = float(row_data[2])
            row['f1_score'] =  float(row_data[3])
            row['samples'] = int(row_data[4])
            report_data.append(row)
    
        return pd.DataFrame.from_dict(report_data)    
    
    def predict(self,x):   
        return self._bnet.predict(x).argmax(axis=1)


