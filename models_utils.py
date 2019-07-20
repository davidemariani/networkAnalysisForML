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
import os

from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

import mlflow
import mlflow.sklearn


def cm_ratio(cm):
    """
    This function takes as an input a scikit learn confusion matrix and returns it with ratio values.
    """

    rcm = np.empty([2,2])
    rcm[0, :] = cm[0, :] / float(sum(cm[0, :]))
    rcm[1, :] = cm[1, :] / float(sum(cm[0, :]))
        
    print("Confusion matrix: \n" + np.array_str(rcm, precision=5, suppress_small=True))

    return rcm


def model_diag(model, X_train, y_train, CrossValFolds=5, scoring = {'AUC':'roc_auc'}):
    """
    This function returns as output false positive rates, true positive rates and auc score for stratified kfold and each
    cross validation forlds in the form of a dictionary.
    It needs model, training x and training y as inputs.
    """
    
    validation = cross_validate(model, X_train, y_train, cv=CrossValFolds, scoring=scoring) #cross_validate is used to evaluate each fold separately
    
    if hasattr(model, "decision_function"): #cross_val_predict is used to make predictions on each data point using stratified folds (evaluation of the whole set)
        y_scores = cross_val_predict(model, X_train, y_train, cv=CrossValFolds, method="decision_function")
    else:
        y_proba = cross_val_predict(model, X_train, y_train, cv=CrossValFolds, method="predict_proba")
        y_scores = y_proba[:,1]
    fpr, tpr, thresholds = roc_curve(y_train, y_scores) #false positive rates, true positive rates and thresholds
    auc = roc_auc_score(y_train, y_scores)
    
    print("AUC {:.3f}".format(auc))

    results = {'fpr':fpr, 'tpr':tpr, 'auc':auc}

    for score in list(scoring.keys()):
        for fold in range(1, len(validation['test_'+score])+1): #saving auc score at each fold of cross validation
            results[score+'_fold_'+str(fold)] = validation['test_'+score][fold-1]

    return results


def model_oostest(model, X_test, y_test):
    """
    This function tests the model performance on out of sample data
    """

    #cm
    y_score = model.predict(X_test)
    cm = confusion_matrix(y_test, y_score)
    rcm = cm_ratio(cm)

    #metrics
    y_scores = model.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    auc = roc_auc_score(y_test, y_scores)

    print("AUC {:.3f}".format(auc))

    return {'fpr':fpr, 'tpr':tpr, 'auc':auc, 'rcm':rcm}



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

    filename = prefix +'_'+ model_name + postfix+'.pkl'

    filepath = datafolder+filename

    # Create target folder if it doesn't exist
    if not os.path.exists(datafolder):
        os.mkdir(datafolder)

    print("Saving model to {}".format(filepath))
    with open(filepath, "wb") as pickle_file:
            pickle.dump(model, pickle_file)

    return (filename, filepath)


def save_dictionary(dict, datafolder, dict_name, postfix):
    """
    This function saves a dictionary to a pickle file
    """
    filepath = datafolder+dict_name+postfix+'.pkl'

    # Create target folder if it doesn't exist
    if not os.path.exists(datafolder):
        os.mkdir(datafolder)

    print("Saving dictionary to {}".format(filepath))
    with open(filepath, "wb") as pickle_file:
            pickle.dump(dict, pickle_file)
    return filepath




def models_loop(models, datafolder, prefixes, postfixes, trainfile='_traindata', testfile='_testdata', scoring = {'AUC':'roc_auc'},
                CrossValFolds=5, save_model=False, models_path='../data/models/', mlf_tracking=False, experiment_name='experiment',
                save_results_for_viz=False, viz_output_path='../data/viz_data/'):
    """
    This function's main purpose is the comparison between validation and testing in order to tune the model during calibration.
    It performs training, validation and testing of one or more models on one or more credit events, requiring as inputs:
    - a list of models
    - the path of the folder containing train and test files generated by the preprocessing pipeline
    - the prefix for each of them (list)
    - the postfixes for each of them (list)
    - traindata and testdata name are default
    - dictionary with the scoring methods for cross validation
    - number of folds for cross validation phase
    - option to save the model, give it an output path and model name
    - option to save AUC visualization data as pickles in specific folder
    - option of tracking the experiments in mlflow

    It returns a dictionary containing the results from validation and testing phase, useful to be plugged in the plot_rocs function
    to visualize AUCs.

    It also allow results and hyperparameters tracking in MLflow setting mlf_tracking to True. 
    """

    #results storage dictionary
    results = {}

    #check that the lists have consistent length
    if len(prefixes) == len(postfixes):
        pass
    else:
        print("Inputs length is inconsistent! Please check that number of prefixes and postfixes is the same.")
        return
    
    for p in range(len(prefixes)): #loop by credit event (imp, p90, p180)
        
        prefix = prefixes[p]
        postfix = postfixes[p]

        for loop in range(len(models)):
            #selecting model and transformed train and test sets
            model = models[loop]
            postfix = postfixes[loop]

            modeltype = str(model).split('(')[0]

            print("Training, validation and testing of experiment with prefix {} and postfix {} using {}".format(prefix, postfix, modeltype))

            #loading preprocessed data
            trainfiles = datafolder + prefix + trainfile + postfix+'.pkl'
            testfiles = datafolder + prefix + testfile + postfix+'.pkl'

            print("-Loading preprocessed data...")
            print("training files: {}".format(trainfiles))
            print("testing files: {}".format(testfiles))
            [X_train, y_train, feature_labels] = pd.read_pickle(trainfiles) 
            [X_test, y_test, feature_labels] = pd.read_pickle(testfiles) 

            #fitting model
            print('- Training...')
            model.fit(X_train, y_train)

            #validation performance
            print('- Validation...')
            model_kfold = model_diag(model, X_train, y_train, CrossValFolds=CrossValFolds, scoring=scoring)
            results[modeltype+'_'+prefix+'validation'] = model_kfold

            #testing on out of sample observations
            print('- Testing...')
            model_oos = model_oostest(model, X_test, y_test)
            results[modeltype+'_'+prefix+'testing'] = model_oos

            #saving the model
            if save_model:
                output_path = models_path+experiment_name+'/'
                print('- Saving the model to {}...'.format(output_path))
                filename, filepath = save_sk_model(model, output_path, modeltype, prefix)

            #saving results dictionary for viz
            if save_results_for_viz:
                dict_name = save_dictionary(results, viz_output_path+experiment_name+'/', filename.split('.')[0],'_viz')

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

                    #model info and hyperparameters tracking
                    mlflow.log_param("model_type", modeltype)

                    if save_model:
                        mlflow.log_param("model_filename", filename)
                        mlflow.log_param("model_filepath", filepath)
                    else:
                        mlflow.log_param("model_filename", None)
                        mlflow.log_param("model_filepath", None)

                    hpar = model.get_params()
                    for par_name in hpar.keys():
                        mlflow.log_param(par_name, hpar[par_name])

                    #kfold validation metrics tracking
                    auc_kf_general = round(model_kfold['auc'],3)
      
                    mlflow.log_metric("validation_nfolds", CrossValFolds)
                    mlflow.log_metric("val_auc", auc_kf_general)

                    for fold in range(1,CrossValFolds+1):
                        auc_kf_fold = round(model_kfold['AUC_fold_'+str(fold)],3)
                        mlflow.log_metric("val_auc_fold_"+str(fold), auc_kf_fold)

                    #test metrics tracking
                    auc = round(model_oos['auc'],3)
                    cm = model_oos['rcm']

                    mlflow.log_metric("test_auc", auc)
                    mlflow.log_metirc("test_tpr", round(cm[0,0],2)) #true positive rate
                    mlflow.log_metirc("test_fpr", round(cm[0,1],2)) #false positive rate
                    mlflow.log_metirc("test_fnr", round(cm[1,0],2)) #false negative ratee
                    mlflow.log_metirc("test_tnr", round(cm[1,1],2)) #true negative rate

                    #storing the model file as pickle
                    mlflow.sklearn.log_model(model, "model")

                    #storing pipeline-processed trainset and testset
                    mlflow.log_artifact(trainfiles, "train_file")
                    mlflow.log_artifact(testfiles, "test_file")
                    mlflow.log_artifact(dict_name, "results_dict_for_viz")
            print()
    return results


def main():
    print("models_utils.py executed/loaded..")

if __name__ == "__main__":
    main()