#==================================================================================#
# Author       : Davide Mariani                                                    #  
# University   : Birkbeck College, University of London                            # 
# Programme    : Msc Data Science                                                  #
# Script Name  : ann_utils.py                                                      #
# Description  : utils for artificial neural networks implementation               #
# Version      : 0.1                                                               #
#==================================================================================#
# This file contains functions do implement and prototype artificial neural        #
# networks using tensorflow 2.0 and keras                                          #
#==================================================================================#

#importing main modules
import pandas as pd
import numpy as np
import pickle
import datetime 
import os

#importing TensorFlow
import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.regularizers import l1, l2
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tensorflow.keras.callbacks import Callback

import mlflow
import mlflow.keras


class TerminateOnBaseline(Callback):
    """Callback that terminates training when either acc or val_acc reaches a specified baseline
    """
    def __init__(self, monitor, baseline):
        super(TerminateOnBaseline, self).__init__()
        self.monitor = monitor
        self.baseline = baseline

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        metric = logs.get(self.monitor)
        if metric is not None:
            if metric >= self.baseline:
                print('Epoch %d: Reached baseline, terminating training' % (epoch))
                self.model.stop_training = True



def create_mlp_model(input_shape = 16,
                     hidden_layers_no=1, 
                     hidden_nodes=[5], 
                     hl_activations = [tf.nn.relu], 
                     random_seed=42, 
                     output_function = tf.nn.sigmoid,
                     optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-4),
                     loss_func = 'binary_crossentropy',
                     metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
                     kernel_regularizers = [],
                     dropout = None,
                    print_summary=True):
    
    """
    This function creates a multilayer percpetron using keras API.
    It requires to specify: input shape (int), number of hidden layers (int), number of hidden nodes in each of the hidden layers (list of int),
    activation function for each hidden layer (list of activation functions), random seed, output function (activation function), 
    optimizer (training optimizer function), loss function, metrics to monitor, regularizer, option to print the summary of architecture
    """
    
    weights_init = tf.keras.initializers.glorot_normal(seed=42)  
    bias_init = tf.keras.initializers.Zeros()
    
    
    input_layer = [tf.keras.layers.Dense(hidden_nodes[0], input_shape=[input_shape], activation=hl_activations[0], 
                                        kernel_initializer=weights_init, bias_initializer=bias_init)] #[tf.keras.layers.Flatten(input_shape=[X_train.shape[1]])]
    
    hidden_layers = []
    
    for i in range(1,hidden_layers_no):
            if len(kernel_regularizers)==0:
                hidden_layers.append(tf.keras.layers.Dense(hidden_nodes[i], 
                                                           activation=hl_activations[i]))
                if dropout!=None:
                    hidden_layers.append(tf.keras.layers.Dropout(dropout[i]))
            else:
                hidden_layers.append(tf.keras.layers.Dense(hidden_nodes[i], 
                                                           activation=hl_activations[i],
                                                           kernel_regularizer=kernel_regularizers[i]))
                if dropout!=None:
                    hidden_layers.append(tf.keras.layers.Dropout(dropout[i]))
    
    output_layer = [tf.keras.layers.Dense(1, activation=output_function)]
    
    model = tf.keras.Sequential(input_layer + hidden_layers + output_layer)
    
    if print_summary:
        print(model.summary())
    
    model.compile(optimizer=optimizer, #tf.keras.optimizers.RMSprop(learning_rate=1e-3),      #'adam', 
              loss=loss_func,
              metrics=metrics)
    
    return model

def main():
    print("ann_utils.py executed/loaded..")

if __name__ == "__main__":
    main()