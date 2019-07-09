#==================================================================================#
# Author       : Davide Mariani                                                    #  
# University   : Birkbeck College, University of London                            # 
# Programme    : Msc Data Science                                                  #
# Script Name  : models_utils.py                                                   #
# Description  : utils for modelling implementation and prototipation              #
# Version      : 0.1                                                               #
#==================================================================================#
# This file contains functions do implement and prototype ML models using          #
# scikit learn and tensorflow                                                      #
#==================================================================================#

#importing main modules
import pandas as pd
import numpy as np
import pickle
import datetime 

from sklearn.model_selection import cross_val_predict, RandomizedSearchCV
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, confusion_matrix
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from visualization_utils import *


def save_sk_model(model, datafolder, model_name, prefix):
    """
    This function saves a scikit learn model in pickle format
    """

    #creating reference for output file
    year = str(datetime.datetime.now().year)[2:]
    month = str(datetime.datetime.now().month)
    if len(month)==1:
        month = '0'+month
    day = str(datetime.datetime.now().day)

    postfix = '_'+year+month+day+'_'+str(datetime.datetime.now().hour)+str(datetime.datetime.now().minute)

    filename = datafolder+prefix +'_'+ model_name + postfix+'.pkl'

    print("Saving model to {}".format(filename))
    with open(filename, "wb") as pickle_file:
            pickle.dump(model, pickle_file)


def models_loop(models, datapath, prefixes, postfixes, trainfile='_traindata', testfile='_testdata',
                CrossValFolds=5, save_sk_model=False):
    """
    This function performs training, validation and testing of one or more models on one or more credit events,
    requiring as inputs:
    - a list of models
    - the path of the folder containing train and test files generated by the preprocessing pipeline
    - the prefix for each of them (list)
    - the postfixes for each of them (list)
    - traindata and testdata name are default
    - number of folds for cross validation phase

    It returns a dictionary containing bokeh plots of the AUC curve for both validation and testing performances of each experiment,
    and another dictionary containing the results from validation and testing phase.
    This function' main purpose is the comparison between validation and testing in order to tune the model during calibration.
    """

    #results storage dictionary
    results = {}

    #visualizations storage dictionary
    vizs = {}

    #check that the lists have consistent length
    if len(prefix) == len(postfix):
        pass
    else:
        print("Inputs length is inconsistent! Please check that number of prefixes and postfixes is the same.")
        return
    
    for p in range(len(prefixes)): #loop by credit event (imp, p90, p180)
        
        prefix = prefixes[p]
        postfix = postfixes[p]

        for loop in range(len(models)):
            #loading transformed train and test sets
            postfix = postfixes[loop]

            print("Training, validation, testing and performance visualization of experiment with prefix {} and postfix {}".format(prefix, postfix))

            [X_train, y_train, feature_labels] = pd.read_pickle(datafolder + prefix + trainfile + postfix[loop+'.pkl') 
            [X_test, y_test, feature_labels] = pd.read_pickle(datafolder + prefix + testfile + postfix+'.pkl') 

            #fitting model
            model = models[loop]
            model.fit(X_train, y_train)

            #validation performance
            model_kfold = model_diag(model, X_train, y_train, run_confusion_matrix=True, CrossValFolds=CrossValFolds)

            #testing on out of sample observations
            model_oos = model_ootest(model, X_test, y_test)

            #visualize AUC curves
            model_auc_vizs = plot_rocs([model_kfold, model_oos], p_width=600, p_height=600, model_appendix=['SGD - 5folds','SGD - test'])

    #PLEASE CONTINUE!!!




def main():
    print("models_utils.py executed..")


if __name__ == "__main__":
    main()