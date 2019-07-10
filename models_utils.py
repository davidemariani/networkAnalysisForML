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


def model_diag(model, X_train, y_train, CrossValFolds=5, run_confusion_matrix=False):
    """
    This function returns as output false positive rate, true positive rate and auc score in the form of a dictionary.
    It needs model, training x and training y as inputs.
    """
    y_pred = cross_val_predict(model, X_train, y_train, cv=CrossValFolds)
    
    if hasattr(model, "decision_function"):
        y_scores = cross_val_predict(model, X_train, y_train, cv=CrossValFolds, method="decision_function")
    else:
        y_proba = cross_val_predict(model, X_train, y_train, cv=CrossValFolds, method="predict_proba")
        y_scores = y_proba[:,1]
    fpr, tpr, thresholds = roc_curve(y_train, y_scores) #false positive rate, true positive rate and thresholds
    auc = roc_auc_score(y_train, y_scores)
    
    print("AUC {:.3f}".format(auc))
    
    if run_confusion_matrix:
        cm = confusion_matrix(y_train, y_pred)
        #rescale the confusion matrix
        rcm = np.empty([2,2])
        rcm[0, :] = cm[0, :] / float(sum(cm[0, :]))
        rcm[1, :] = cm[1, :] / float(sum(cm[0, :]))
        
        print("Confusion matrix: \n" + np.array_str(rcm, precision=5, suppress_small=True))
    
    return {'fpr':fpr, 'tpr':tpr, 'auc':auc}


def model_oostest(model, X_test, y_test):
    """
    This function tests the model performance on out of sample data
    """

    #cm count
    y_score = model.predict(X_test)
    cm = confusion_matrix(y_test, y_score)
    print(cm)

    #cm ratio and auc
    y_scores = model.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    auc = roc_auc_score(y_test, y_scores)

    print("AUC {:.3f}".format(auc))

    return {'fpr':fpr, 'tpr':tpr, 'auc':auc}



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
                CrossValFolds=5, save_model=False, output_path='', model_name='', prefix=''):
    """
    This function' main purpose is the comparison between validation and testing in order to tune the model during calibration.
    It performs training, validation and testing of one or more models on one or more credit events, requiring as inputs:
    - a list of models
    - the path of the folder containing train and test files generated by the preprocessing pipeline
    - the prefix for each of them (list)
    - the postfixes for each of them (list)
    - traindata and testdata name are default
    - number of folds for cross validation phase

    It returns a dictionary containing the results from validation and testing phase, useful to be plugged in the plot_rocs function
    to visualize AUCs.
    """

    #results storage dictionary
    results = {}

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

            [X_train, y_train, feature_labels] = pd.read_pickle(datafolder + prefix + trainfile + postfix+'.pkl') 
            [X_test, y_test, feature_labels] = pd.read_pickle(datafolder + prefix + testfile + postfix+'.pkl') 

            #loading model and retrieving its data
            model = models[loop]
            model_name = str(model).split('(')[0]

            #fitting model
            model.fit(X_train, y_train)

            #validation performance
            print('Validation...')
            model_kfold = model_diag(model, X_train, y_train, run_confusion_matrix=True, CrossValFolds=CrossValFolds)
            results[model_name+'_'+prefix+'validation'] = model_kfold
            print()

            #testing on out of sample observations
            print('Testing...')
            model_oos = model_ootest(model, X_test, y_test)
            results[model_name+'_'+prefix+'testing'] = model_oos

            #saving the model
            if save_model:
                save_sk_model(model, output_path, model_name, prefix)

    return results


def main():
    print("models_utils.py executed..")

if __name__ == "__main__":
    main()