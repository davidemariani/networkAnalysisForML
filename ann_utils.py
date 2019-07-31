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
import time

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


def plot_epochs_graph(history_dict, metric, plot_validation):
    """
    This function takes as inputs a dictionary containing mpl history data
    and the name of the metric to monitor, and plot a graph using matplotlib 
    for model diagnostics
    """

    values = history_dict[metric]
    epochs = range(1, len(values)+1)

    if plot_validation:
        val_values = history_dict['val_'+metric]
        plt.plot(epochs, values, 'bo', label='Train {:}'.format(metric))
        plt.plot(epochs, val_values, 'r', label='Validation {:}'.format(metric))
        plt.title("Training and validation {:}".format(metric))
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend()
        plt.show()

    else:
        plt.plot(epochs, values, 'bo', label='Train {:}'.format(metric))
        plt.title("Training {:}".format(metric))
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend()
        plt.show()


def plot_diag(history_dict, validation_ep):
    """
    This function call the plot_epochs_graph function handling the exceptions
    due to dynamic names assigned during training
    """

    plot_epochs_graph(history_dict, 'loss', validation_ep)
    plot_epochs_graph(history_dict, 'accuracy', validation_ep)

    count=0
    not_found=True
    while not_found:
        try: #handling some inconsistency in naming the metrics during training
            if count!=0:
                try_this = history_dict['auc_'+str(count)]
                plot_epochs_graph(history_dict,  'auc_'+str(count), validation_ep) 
                not_found = False
            else:
                try_this = history_dict['auc']
                plot_epochs_graph(history_dict, 'auc', validation_ep) 
                not_found = False
        except KeyError:
            count+=1
            pass


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


def validation_split(X_train, y_train, validation_size, validation_mode='midpoint'):
    """
    This function will split training set into two parts in order to take one of it for validation depending on the validation modes
    (midpoint, startpoint, endpoint)
    """

    if validation_mode=='midpoint':
        midpoint = X_train.shape[0]//2 #in order to have meaningful data for validation, val data are taken from the midpoint of the training set (data are sorted by time and not shuffled)
        X_val = X_train[midpoint-validation_size//2:midpoint+validation_size//2]
        partial_X_train = np.array(list(X_train[:midpoint-validation_size//2])+list(X_train[midpoint+validation_size//2:]))
        y_val = y_train[midpoint-validation_size//2:midpoint+validation_size//2]
        partial_y_train = np.array(list(y_train[:midpoint-validation_size//2])+list(y_train[midpoint+validation_size//2:]))
    
    elif validation_mode=='startpoint':
        X_val = X_train[:validation_size]
        partial_X_train = X_train[validation_size:]
        y_val = y_train[:validation_size]
        partial_y_train = y_train[validation_size:]

    elif validation_mode=='endpoint':
        X_val = X_train[len(X_train)-validation_size:]
        partial_X_train = X_train[:len(X_train)-validation_size]
        y_val = y_train[len(X_train)-validation_size:]
        partial_y_train = y_train[:len(X_train)-validation_size]

    return partial_X_train, X_val, partial_y_train, y_val



def tr_time(training_start, training_end):
    """
    This function calculates the training time having a time start and a time end to compute it.
    It also prints the results on screen and return the message to be stored for tracking.
    """

    training_time = training_end-training_start

    if training_time>3600:
        hrs = int(training_time//3600)
        mins = int((training_time%3600)//60)
        secs = round((training_time%3600)%60, 2)
        time_message = "Training completed in {} hrs, {} mins and {} secs".format(hrs, mins, secs)
        print("-------------TRAINING TIME--------------")
        print(time_message)
    else:
        mins = int(training_time//60)
        secs = round(training_time%60,2)
        print("-------------TRAINING TIME--------------")
        time_message = "Training completed in {} mins and {} secs".format(mins, secs)
        print(time_message)

    return training_time, time_message



def mlp_training(mlp, X_train, y_train, epochs=50, batch_size=512, use_batch_and_steps=False, class_1_weight=25, shuffle=False,
                 validation_ep=True, validation_size=1000, validation_mode='midpoint', early_stopping=True, to_monitor=('accuracy',0.9)):
    """
    This function is an auxiliar function for mlp_exp and performs a multi-layer perceptron training considering the following parameters and attributes:
    mlp (the mlp to train), X_train (train set observations), y_train (train set labels), epochs (int), batch_size (int), use_batch_and_steps (True/False),
    class_1_weight (int), shuffle (True/False), validation_ep (True/False - it indicates if a portion of train set is used for validation), validation_size (int),
    validation_mode (type of validation portion in regard to the train set: 'midpoint', 'endpoint', 'startpoint'),
    early_stopping (True/False), to_monitor (metric to monitor and value to stop for early stopping)
    """

    #fitting
    if early_stopping: #including early stopping callback

        print("Early stopping active: training will be stopped when {0} overcomes the value {1}".format(to_monitor[0], to_monitor[1]))

        es_callback = TerminateOnBaseline(to_monitor[0], to_monitor[1])

        if validation_ep: #training with epochs validation
            partial_X_train, X_val, partial_y_train, y_val = validation_split(X_train, y_train, validation_size, validation_mode)

            if use_batch_and_steps: #case where both steps per epochs and batch size are used
                history = mlp.fit(partial_X_train, partial_y_train, epochs=epochs,  batch_size = batch_size, verbose=1, 
                    callbacks=[es_callback], steps_per_epoch = math.ceil(partial_X_train.shape[0]/batch_size),
                                validation_data=(X_val, y_val), class_weight={0:1, 1:class_1_weight}, 
                                shuffle=shuffle)
            else:
                history = mlp.fit(partial_X_train, partial_y_train, epochs=epochs,  batch_size = batch_size, verbose=1, 
                    callbacks=[es_callback],  
                                validation_data=(X_val, y_val), class_weight={0:1, 1:class_1_weight}, 
                                shuffle=shuffle)

        else: #case without validation split
            if use_batch_and_steps: #case where both steps per epochs and batch size are used
                history = mlp.fit(X_train, y_train, epochs=epochs,  batch_size = batch_size, verbose=1, 
                    callbacks=[es_callback], steps_per_epoch = math.ceil(X_train.shape[0]/batch_size),
                                class_weight={0:1, 1:class_1_weight}, shuffle=shuffle)
            else:
                history = mlp.fit(X_train, y_train, epochs=epochs,  batch_size = batch_size, verbose=1, 
                    callbacks=[es_callback], class_weight={0:1, 1:class_1_weight}, shuffle=shuffle)

    else: #case without early stopping callback

        if validation_ep: #training with epochs validation
            partial_X_train, X_val, partial_y_train, y_val = validation_split(X_train, y_train, validation_size, validation_mode)

            if use_batch_and_steps: #case where both steps per epochs and batch size are used
                history = mlp.fit(partial_X_train, partial_y_train, epochs=epochs,  batch_size = batch_size, verbose=1, 
                                    steps_per_epoch = math.ceil(partial_X_train.shape[0]/batch_size),
                                    validation_data=(X_val, y_val), class_weight={0:1, 1:class_1_weight}, 
                                    shuffle=shuffle)
            else:
                history = mlp.fit(partial_X_train, partial_y_train, epochs=epochs,  batch_size = batch_size, verbose=1, 
                                    validation_data=(X_val, y_val), class_weight={0:1, 1:class_1_weight}, 
                                    shuffle=shuffle)
        else:#case without validation split
            if use_batch_and_steps: #case where both steps per epochs and batch size are used
                history = mlp.fit(X_train, y_train, epochs=epochs,  batch_size = batch_size, verbose=1, 
                                    steps_per_epoch = math.ceil(X_train.shape[0]/batch_size),
                                    class_weight={0:1, 1:class_1_weight}, 
                                    shuffle=shuffle)
            else:
                history = mlp.fit(X_train, y_train, epochs=epochs,  batch_size = batch_size, verbose=1, 
                                    class_weight={0:1, 1:class_1_weight}, 
                                    shuffle=shuffle)
    if validation_ep:
        return history, partial_X_train, partial_y_train, X_val, y_val
    else:
        return history

def mlp_oostest(model, X_test, y_test, pred_thr):
    """
    This function tests the model performance on out of sample data
    """

    #cm
    y_score_prob = model.predict(X_test)
    y_score = y_score_prob>pred_thr
    cm = confusion_matrix(y_test, y_score)
    rcm = cm_ratio(cm)

    #metrics
    fpr, tpr, thresholds = roc_curve(y_test, y_score_prob)
    auc = roc_auc_score(y_test, y_score_prob)

    print("AUC {:.3f}".format(auc))

    return {'fpr':fpr, 'tpr':tpr, 'auc':auc, 'rcm':rcm, 'cm':cm}



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
               validation_ep = True,
               validation_size=10000,
               validation_mode = 'midpoint',
               pred_threshold = 0.55,
               kernel_regularizers=[],
               shuffle=False,
               use_batch_and_steps=False,
               save_model=False,
               retrain_for_testing = False,
               plot_diagnostics = True,
               models_path='../data/models/', mlf_tracking=False, experiment_name='experiment',
               save_results_for_viz=False, viz_output_path='../data/viz_data/'):

    """
    This function performs an end-to-end experiment (training, validation, testing) using a multi-layer perceptron.
    It returns a dictionary with the experiment data and a list with the predictions.
    It requires the following inputs:
    - datfolder: the folder from where taking the input data files
    - prefix: prefix id of the input datafile
    - postfix: postfix id of the input datafile
    - trainfile and testfile name
    - hidden_layers_no: number of hidden layers for the mlp architecture (int)
    - hidden_nodes: list containing the number of nodes for each hidden layer in sequence
    - hl_activations: list containing the activation functions for each hidden layer in sequence
    - optimizer: learning algorithm to use
    - loss_func: the function used to calculate the loss
    - kernel_initializer: kernel initializer (lecun_uniform is the default)
    - bias_initializer: weights bias initializer (zeros is the default)
    - metrics: metrics to monitor during training
    - dropout: list of dropout values. No dropout is used if the list is empty
    - to_monitor: this is an input used with "early_stopping", which indicates for which metric and for which value the early stopping stops the process
    - early_stopping: Boolean indicating if early_stopping will be active or not
    - batch_size: batch size
    - epochs: number of epochs for training
    - class_1_weight: the weight to give to the event to detect
    - validation_ep: Boolean indicating if during training a portion of the training set is used for validation at each epoch comparing training and validation
    - validation_size: the size of the validation portion to take from the train set
    - validation_mode: the way the validation portion will be taken from the train set: if 'midpoint' from the middle, if 'endpoint' from the end, if 'startpoint' from the start
    - kernel_regularizers: list of regularizers for each layer
    - shuffle: if train and validation data are shuffled
    - use_batch_and_steps: if True, the train set is used train.shape[0]//batch_size times for each epoch, otherwise a batch of the selected size is taken from it and used once for each epoch
    - save_model: if True, the model is saved at model_path
    - retrain_for_testing: if True, the model is first trained without a portion of size 'validation_size' for validation, and then retrained on the full training set for the predictions on test set
    - plot_diagnostics: if True, plots showing main training info are displayed
    - mlf_tracking: if True, mlflow is activated for tracking the experiment
    - experiment_name: experiment name
    - save_results_for_viz: if True, the dictionary containing experiment data is saved at viz_output_path
    """

    #loading preprocessed data
    trainfiles = datafolder + prefix + trainfile + postfix+'.pkl'
    testfiles = datafolder + prefix + testfile + postfix+'.pkl'

    print("-Loading preprocessed data...")
    print("training files: {}".format(trainfiles))
    print("testing files: {}".format(testfiles))
    [X_train, y_train, feature_labels] = pd.read_pickle(trainfiles) 
    [X_test, y_test, feature_labels] = pd.read_pickle(testfiles) 
    
    input_shape = len(feature_labels) #shape of experiment input

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

    training_start = time.time() #tracking training time

    #training
    if validation_ep: #if a validation split is done, X_val and y_val are returned for last evaluations on validation set before testing
        history, partial_X_train, partial_y_train, X_val, y_val = mlp_training(mlp, X_train, y_train, epochs=epochs, batch_size=batch_size, use_batch_and_steps=use_batch_and_steps, class_1_weight=class_1_weight, shuffle=shuffle,
                    validation_ep=validation_ep, validation_size=validation_size, validation_mode=validation_mode, early_stopping=early_stopping, to_monitor=to_monitor)
    else:
        history = mlp_training(mlp, X_train, y_train, epochs=epochs, batch_size=batch_size, use_batch_and_steps=use_batch_and_steps, class_1_weight=class_1_weight, shuffle=shuffle,
                    validation_ep=validation_ep, validation_size=validation_size, validation_mode=validation_mode, early_stopping=early_stopping, to_monitor=to_monitor)

    training_end = time.time()
    
    training_time, time_message = tr_time(training_start, training_end)

    #epochs history data to store
    history_dict = history.history
    history_dict['experiment'] = experiment_name
    history_dict['prefix'] = prefix
    history_dict['postfix'] = postfix

    if plot_diagnostics: #plotting model data
        plot_diag(history_dict, validation_ep)
        
    #predictions on validation-set if validation_ep is active
    if validation_ep:
        #validation AUC
        print("Prediction performance on {} observations from validation set using holdout".format(X_val.shape[0]))
        history_dict['validation_results'] = mlp_oostest(mlp, X_val, y_val, pred_threshold)

    if retrain_for_testing:
        #retraining the model from scratch on the full dataset
        #in case of retraining, the retrained model will be saved and its training time will be recorded
        #while the model trained with a validation split won't be stored
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

        training_start = time.time() #tracking training time

        history_test = mlp_training(mlp, X_train, y_train, epochs=epochs, batch_size=batch_size, use_batch_and_steps=use_batch_and_steps, class_1_weight=class_1_weight, shuffle=shuffle,
                        validation_ep=False, early_stopping=early_stopping, to_monitor=to_monitor)

        training_end = time.time()
    
        training_time, time_message = tr_time(training_start, training_end)

        if plot_diagnostics: #plotting test model data
            history_test_dict = history_test.history
            plot_diag(history_test_dict, False)


    #saving the model
    if save_model:
        output_path = models_path+experiment_name+'/'
        print('- Saving the model to {}...'.format(output_path))
        filename, filepath = save_tf_model(mlp, output_path, modeltype, prefix)

    #test AUC
    print("Prediction performance on {} observations from test set".format(X_test.shape[0]))
    history_dict['test_results'] = mlp_oostest(mlp, X_test, y_test, pred_threshold) 

    #test cm
    print("Confusion matrix:")
    cm = history_dict['test_results']['cm']  #confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    print(cm)

    rcm = history_dict['test_results']['rcm']  #cm_ratio(cm)
    tnr, fpr, fnr, tpr = rcm.ravel()

    history_dict['test_confusion_matrix'] = {'tn':tn, 'fp':fp, 'fn':fn, 'tp':tp,
                                             'tnr':tnr, 'fpr':fpr, 'fnr':fnr, 'tpr':tpr}

    #actual predictions with threshold
    preds = mlp.predict(X_test)>pred_threshold

    #saving results dictionary for viz
    if save_results_for_viz:
        dict_name = save_dictionary(history_dict, viz_output_path+experiment_name+'/', filename.split('.')[0],'_viz')

    if mlf_tracking: #mlflow tracking

        #checking the name of existing experiments
        expnames = set([exp.name for exp in mlflow.tracking.MlflowClient().list_experiments()])

        #creating a new experiment if its name is not among the existing ones
        if experiment_name not in expnames:
            print("- Creating the new experiment '{}',  the following results will be saved in it...".format(experiment_name))
            exp_id = mlflow.create_experiment(experiment_name)
        else: #adding new results to the existing one otherwise
            print("- Activating existing experiment '{}', the following results will be saved in it...".format(experiment_name))
            mlflow.set_experiment(experiment_name)
            exp_id = mlflow.tracking.MlflowClient().get_experiment_by_name(experiment_name).experiment_id
               
        with mlflow.start_run(experiment_id=exp_id, run_name=prefix + modeltype): #create and initialize experiment

            print("- Tracking the experiment on mlflow...")

            #experiment type tracking
            mlflow.log_param("experiment_type", prefix)

            #source file tracking
            mlflow.log_param("train_file_path", trainfiles)
            mlflow.log_param("train_file_name", prefix + trainfile + postfix+'.pkl')
            mlflow.log_param("test_file_path", testfiles)
            mlflow.log_param("test_file_name", prefix + testfile + postfix+'.pkl')
            mlflow.log_param("train_size", X_train.shape[0])
            mlflow.log_param("test_size", X_test.shape[0])

            if validation_ep:
                mlflow.log_param("partial_train_size", partial_X_train.shape[0])
                mlflow.log_param("validation_size", X_val.shape[0])
                mlflow.log_param("validation_size", validation_size)

            #model info and hyperparameters tracking
            mlflow.log_param("model_type", modeltype)

            if save_model:
                mlflow.log_param("model_filename", filename)
                mlflow.log_param("model_filepath", filepath)
            else:
                mlflow.log_param("model_filename", None)
                mlflow.log_param("model_filepath", None)

            #mlp hyperparameters:
            mlflow.log_param("hidden_layers_no", hidden_layers_no)
            mlflow.log_param("hidden_nodes", str(hidden_nodes))
            mlflow.log_param("hl_out_activations", str([l['config']['activation'] for l in mlp.get_config()['layers'] if l['class_name']=='Dense']))
            mlflow.log_param("optimizer", str(optimizer).split('tensorflow.python.keras.optimizer_v2.')[1].split(' ')[0])
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("loss_func", str(mlp.loss).split('<tensorflow.python.keras.losses.')[1].split(' ')[0])
            mlflow.log_param("epochs_settings", epochs)
            mlflow.log_param("kernel_init", str(kernel_initializer.get_config()))
            mlflow.log_param("kernel_regularizers", str(kernel_regularizers))
            mlflow.log_param("bias_init", str([l['config']['bias_initializer']['class_name'] for l in mlp.get_config()['layers'] if l['class_name']=='Dense']))
            mlflow.log_param("dropout", dropout)
            mlflow.log_param("early_stopping", early_stopping)
            if early_stopping:
                mlflow.log_param("early_stopping_metric", str(to_monitor))
            mlflow.log_param("class_1_weight", class_1_weight)
            mlflow.log_param("tr_val_shuffle", shuffle)
            mlflow.log_param("batch_and_steps", use_batch_and_steps)
            mlflow.log_param("pred_threshold", pred_threshold)
            mlflow.log_param("tr_time_str", time_message)
          
            #mlp metrics
            mlflow.log_metric("epochs_actual", len(history_dict['loss']))
            mlflow.log_metric("tr_time", training_time)

            for metric in mlp.metrics_names:
                mlflow.log_metric("tr_"+metric, history_dict[metric][-1])
                if validation_ep:
                    mlflow.log_metric("val_"+metric, history_dict['val_'+metric][-1])

            mlflow.log_metric("test_auc", history_dict['test_results']['auc'])
            mlflow.log_metric("test_tp", history_dict["test_confusion_matrix"]["tp"])
            mlflow.log_metric("test_tn", history_dict["test_confusion_matrix"]["tn"])
            mlflow.log_metric("test_fp", history_dict["test_confusion_matrix"]["fp"])
            mlflow.log_metric("test_fn", history_dict["test_confusion_matrix"]["fn"])
            mlflow.log_metric("test_tpr", history_dict["test_confusion_matrix"]["tpr"])
            mlflow.log_metric("test_tnr", history_dict["test_confusion_matrix"]["tnr"])
            mlflow.log_metric("test_fpr", history_dict["test_confusion_matrix"]["fpr"])
            mlflow.log_metric("test_fnr", history_dict["test_confusion_matrix"]["fnr"])

            #storing the model file as pickle
            mlflow.keras.log_model(mlp, "model")

            #storing pipeline-processed trainset and testset
            mlflow.log_artifact(trainfiles, "train_file")
            mlflow.log_artifact(testfiles, "test_file")
            mlflow.log_artifact(dict_name, "results_dict_for_viz")

            print("- Experiment tracked.")

    return history_dict, preds

def main():
    print("ann_utils.py executed/loaded..")

if __name__ == "__main__":
    main()