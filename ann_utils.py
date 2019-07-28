#==================================================================================#
# Author       : Davide Mariani                                                    #  
# University   : Birkbeck College, University of London                            # 
# Programme    : Msc Data Science                                                  #
# Script Name  : ann_utils.py                                                      #
# Description  : utils for artificial neural networks implementation               #
# Version      : 0.2                                                               #
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
import matplotlib.pyplot as plt
import math

from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from models_utils import *

#importing TensorFlow
import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.optimizers import RMSprop, Adam
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
                     kernel_initializer = tf.keras.initializers.lecun_uniform(seed=42),
                     bias_initializer = tf.keras.initializers.Zeros(),
                     dropout = None,
                    print_summary=True):
    
    """
    This function creates a multilayer percpetron using keras API.
    It requires to specify: input shape (int), number of hidden layers (int), number of hidden nodes in each of the hidden layers (list of int),
    activation function for each hidden layer (list of activation functions), random seed, output function (activation function), 
    optimizer (training optimizer function), loss function, metrics to monitor, regularizer, option to print the summary of architecture
    """
   
    
    
    input_layer = [tf.keras.layers.Dense(hidden_nodes[0], input_shape=[input_shape], activation=hl_activations[0], 
                                        kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)] 

    if dropout:
        input_layer.append(tf.keras.layers.Dropout(dropout[0]))

    
    hidden_layers = []
    
    for i in range(1,hidden_layers_no): #operations on hidden layers (dropout or other regularization addition)
            if len(kernel_regularizers)==0:
                hidden_layers.append(tf.keras.layers.Dense(hidden_nodes[i], kernel_initializer=kernel_initializer,
                                                           activation=hl_activations[i]))
                if dropout!=None:
                    hidden_layers.append(tf.keras.layers.Dropout(dropout[i]))
            else:
                hidden_layers.append(tf.keras.layers.Dense(hidden_nodes[i], kernel_initializer=kernel_initializer,
                                                           activation=hl_activations[i],
                                                           kernel_regularizer=kernel_regularizers[i]))
                if dropout!=None:
                    hidden_layers.append(tf.keras.layers.Dropout(dropout[i]))
    
    output_layer = [tf.keras.layers.Dense(1, activation=output_function, kernel_initializer=kernel_initializer)]
    
    model = tf.keras.Sequential(input_layer + hidden_layers + output_layer)
    
    if print_summary:
        print(model.summary())
    
    model.compile(optimizer=optimizer, #tf.keras.optimizers.RMSprop(learning_rate=1e-3),      #'adam', 
              loss=loss_func,
              metrics=metrics)
    
    return model


def plot_epochs_graph(history_dict, metric):
    """
    This function takes as inputs a dictionary containing mpl history data
    and the name of the metric to monitor, and plot a graph using matplotlib 
    for model diagnostics
    """
    values = history_dict[metric]
    val_values = history_dict['val_'+metric]
    epochs = range(1, len(val_values)+1)

    plt.plot(epochs, values, 'bo', label='Train {:}'.format(metric))
    plt.plot(epochs, val_values, 'b', label='Validation {:}'.format(metric))
    plt.title("Training and validation {:}".format(metric))
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend()

    plt.show()


def save_tf_model(model, datafolder, model_name, prefix):
    """
    This function saves a tensorflow model in h5 format
    """

    #creating reference for output file
    year = str(datetime.datetime.now().year)[2:]
    month = str(datetime.datetime.now().month)
    if len(month)==1:
        month = '0'+month
    day = str(datetime.datetime.now().day)

    postfix = '_'+year+month+day+'_'+str(datetime.datetime.now().hour)+str(datetime.datetime.now().minute)

    filename = prefix +'_'+ model_name + postfix+'.h5'

    filepath = datafolder+filename

    # Create target folder if it doesn't exist
    if not os.path.exists(datafolder):
        os.mkdir(datafolder)

    print("Saving model to {}".format(filepath))
    model.save(filepath)
 
    return (filename, filepath)





def mlp_exp(datafolder, prefix, postfix, 
            trainfile='_traindata', testfile='_testdata',
               hidden_layers_no=1,
               hidden_nodes=[5],
               hl_activations=[tf.nn.relu],
               optimizer=Adam(),
               loss_func=tf.keras.losses.BinaryCrossentropy(),
               kernel_initializer = tf.keras.initializers.lecun_uniform(seed=42),
               bias_initializer = tf.keras.initializers.Zeros(),
               metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
               dropout=[0.45],
               to_monitor=('accuracy', 0.9),
               early_stopping=False,
               batch_size=512,
               epochs=5,
               class_1_weight=50,
               validation_size=10000,
               pred_threshold = 0.55,
               kernel_regularizers=[],
               shuffle=False,
               use_batch_and_steps=False,
               save_model=False,
               plot_diagnostics = True,
               models_path='../data/models/', mlf_tracking=False, experiment_name='experiment',
               save_results_for_viz=False, viz_output_path='../data/viz_data/'):

    #loading preprocessed data
    trainfiles = datafolder + prefix + trainfile + postfix+'.pkl'
    testfiles = datafolder + prefix + testfile + postfix+'.pkl'

    print("-Loading preprocessed data...")
    print("training files: {}".format(trainfiles))
    print("testing files: {}".format(testfiles))
    [X_train, y_train, feature_labels] = pd.read_pickle(trainfiles) 
    [X_test, y_test, feature_labels] = pd.read_pickle(testfiles) 
    
    input_shape = len(feature_labels)

    #create mlp
    mlp = create_mlp_model(input_shape=input_shape, 
                       hidden_layers_no=hidden_layers_no,
                       hidden_nodes=hidden_nodes, 
                       hl_activations=hl_activations,                                        
                       optimizer = optimizer,
                       loss_func=loss_func,
                      kernel_regularizers = kernel_regularizers,
                      kernel_initializer = kernel_initializer,
                      bias_initializer = bias_initializer,
                       dropout = dropout,
                      metrics = metrics)

    modeltype = mlp.get_config()['name']

    #taking validation dataset from the training dataset
    midpoint = X_train.shape[0]//2 #in order to have meaningful data for validation, val data are taken from the midpoint of the training set (data are sorted by time and not shuffled)
    X_val = X_train[midpoint-validation_size//2:midpoint+validation_size//2]
    partial_X_train = np.array(list(X_train[:midpoint-validation_size//2])+list(X_train[midpoint+validation_size//2:]))
    y_val = y_train[midpoint-validation_size//2:midpoint+validation_size//2]
    partial_y_train = np.array(list(y_train[:midpoint-validation_size//2])+list(y_train[midpoint+validation_size//2:]))

    #fitting
    if early_stopping: #including early stopping callback

        print("Early stopping active: training will be stopped when {0} overcomes the value {1}".format(to_monitor[0], to_monitor[1]))

        es_callback = TerminateOnBaseline(to_monitor[0], to_monitor[1])

        if use_batch_and_steps: #case where both steps per epochs and batch size are used
            history = mlp.fit(partial_X_train, partial_y_train, epochs=epochs,  batch_size = batch_size, verbose=1, 
                callbacks=[es_callback], steps_per_epoch = math.ceil(X_train.shape[0]/batch_size),
                         validation_data=(X_val, y_val), class_weight={0:1, 1:class_1_weight}, 
                          shuffle=shuffle)
        else:
            history = mlp.fit(partial_X_train, partial_y_train, epochs=epochs,  batch_size = batch_size, verbose=1, 
                callbacks=[es_callback],  
                         validation_data=(X_val, y_val), class_weight={0:1, 1:class_1_weight}, 
                          shuffle=shuffle)

    else: #case without early stopping callback
        if use_batch_and_steps: #case where both steps per epochs and batch size are used
            history = mlp.fit(partial_X_train, partial_y_train, epochs=epochs,  batch_size = batch_size, verbose=1, 
                             steps_per_epoch = math.ceil(X_train.shape[0]/batch_size),
                             validation_data=(X_val, y_val), class_weight={0:1, 1:class_1_weight}, 
                              shuffle=shuffle)
        else:
            history = mlp.fit(partial_X_train, partial_y_train, epochs=epochs,  batch_size = batch_size, verbose=1, 
                             validation_data=(X_val, y_val), class_weight={0:1, 1:class_1_weight}, 
                              shuffle=shuffle)


    #epochs history data to store
    history_dict = history.history
    history_dict['experiment'] = experiment_name
    history_dict['prefix'] = prefix
    history_dict['postfix'] = postfix

    if plot_diagnostics:
        plot_epochs_graph(history_dict, 'loss')
        plot_epochs_graph(history_dict, 'accuracy')
        plot_epochs_graph(history_dict,  'auc') 

    if save_model:
        output_path = models_path+experiment_name+'/'
        print('- Saving the model to {}...'.format(output_path))
        filename, filepath = save_tf_model(mlp, output_path, modeltype, prefix)
        
    #predictions on validation-set
    predictions_val = mlp.predict(X_val)

    #validation AUC
    print("Prediction performance on {} observations from validation set using holdout".format(X_val.shape[0]))
    vfpr, vtpr, vthresholds = roc_curve(y_val, predictions_val) 
    vauc = roc_auc_score(y_val, predictions_val)
    print('AUC: {}'.format(vauc))

    history_dict['validation_results'] = {'fpr':vfpr, 'tpr':vtpr, 'auc':vauc}

    #predictions on test-set
    predictions = mlp.predict(X_test)
    preds = (predictions>pred_threshold)

    #test AUC
    print("Prediction performance on {} observations from test set".format(X_test.shape[0]))
    fpr, tpr, thresholds = roc_curve(y_test, predictions) 
    auc = roc_auc_score(y_test, predictions)
    print('AUC: {}'.format(auc))

    history_dict['test_results'] = {'fpr':fpr, 'tpr':tpr, 'auc':auc}

    #test cm
    print("Confusion matrix:")
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    print(cm)

    rcm = cm_ratio(cm)
    tnr, fpr, fnr, tpr = rcm.ravel()
    print(rcm)

    history_dict['test_confusion_matrix'] = {'tn':tn, 'fp':fp, 'fn':fn, 'tp':tp,
                                             'tnr':tnr, 'fpr':fpr, 'fnr':fnr, 'tpr':tpr}

    return history_dict

def main():
    print("ann_utils.py executed/loaded..")

if __name__ == "__main__":
    main()