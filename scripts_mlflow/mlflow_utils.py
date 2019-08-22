#==================================================================================#
# Author       : Davide Mariani                                                    #  
# University   : Birkbeck College, University of London                            # 
# Programme    : Msc Data Science                                                  #
# Script Name  : mlflow_utils.py                                                   #
# Description  : utils for ML experiments tracking using mlflow                    #
# Version      : 0.3                                                               #
#==================================================================================#
# This file contains functions to track the ML experiments using mlflow            #
#==================================================================================#

#importing main modules
import pandas as pd
import numpy as np
import pickle
import datetime 
import os

import mlflow
import mlflow.sklearn

from scripts_ml.models_utils import *

from os import listdir
from os.path import isdir, join


#-----------------------------------------
# UTILS
#-----------------------------------------

def list_to_string(list_):
            """
            This function is an auxiliar function for the mlflow tracking system.
            It transform a list of values into a single string with comma separated values that can be reproduced for viz
            """
            string_=''

            count=0
            for i in list_:
                if count!=len(list_)-1:
                    string_+=str(i)
                    string_+=","
                else:
                    string_+=str(i)
            return string_


def set_clf_cols(viz, abbr_dict={"RandomForestClassifier":"RF", "SGDClassifier":"SGD"}):
    """
    This is an auxiliar function for create_exp_df.
    It gets a visualization dataframe as inputs and it returns the same dataframe adapted for gridplot viz.
    """
    
    viz = viz.copy()
    names = viz['model_filename']
    
    mod_names = []
    
    count=0
    postfix = ''
    for name in names:
        try:
            postfix = name.split(list(viz['model_type'])[count])[1].replace(postfix+'__', '').replace('.pkl','')

            types = [abbr_dict[i] if i in abbr_dict.keys() else i for i in list(viz['model_type'])]
            newname = types[count]+postfix
            mod_names.append(newname)
        except:
            mod_names.append(str(name))
        count+=1
    
    viz.index = mod_names
    
    return viz.transpose()


#-----------------------------------------
# EXPERIMENT TRACKING
#-----------------------------------------

def mlf_sk_tracking(experiment_name, prefix, postfix, modeltype, trainfile, testfile, datafolder,
                    train_size, test_size, model, model_kfold, model_oos, timeSeqValid, feat_imp_dict,
                    CrossValFolds=None, save_model=False, filename=None, filepath=None,
                    save_results_for_viz=False, dict_name=None):
    """
    This function is an auxiliar function of models_loop which activates the mlflow tracking of an experiment
    using RandomForestClassifier and/or SGDClassifier implemented in scikit-learn.
    Inputs required are:
    - experiment_name: the name of the experiment
    - prefix: the prefix used for the experiment for files sourcing and saving
    - postfix: the postfix used for the experiment for files sourcing and saving
    - modeltype: the type of model used
    - trainfile: the name of the training files
    - testfile: the name of the testing files
    - datafolder: the folder where preprocessed data of the experiment have been saved
    - train_size: the size of the training set
    - test_size: the size of the test set
    - model: the model used during the experiment
    - model_kfold: the output of the model_diag or model_diag_seq used during the experiment (dict)
    - model_oos: the output of out of sample testing used during the experiment (dict)
    - timeSeqValid: boolean value indicating if sequential validation has been used
    - feat_imp_dict: a dictionary containing the feature importances
    - save_model: boolean indicating if the model has been stored somewhere
    - filename: the name of the model
    - filepath: the path where the model has been saved
    """

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

        trainfilepath = datafolder + prefix + trainfile + postfix+'.pkl'
        testfilepath = datafolder + prefix + testfile + postfix+'.pkl'

        #source file tracking
        mlflow.log_param("train_file_path", trainfilepath)
        mlflow.log_param("train_file_name", prefix + trainfile + postfix+'.pkl')
        mlflow.log_param("test_file_path", testfilepath)
        mlflow.log_param("test_file_name", prefix + testfile + postfix+'.pkl')
        mlflow.log_param("train_size", train_size)
        mlflow.log_param("test_size", test_size)

        #model info and hyperparameters tracking
        mlflow.log_param("model_type", modeltype)

        if save_model:
            if filename!=None or filepath!=None:
                mlflow.log_param("model_filename", filename)
                mlflow.log_param("model_filepath", filepath)
            else:
                print("WARNING! - filename and filepath set to 'None'! Please change them to a meaningful output path. The models output path have not been saved in mlflow!")
        else:
            mlflow.log_param("model_filepath", None)
            mlflow.log_param("model_filename", None)

        hpar = model.get_params()
        for par_name in hpar.keys():
            mlflow.log_param(par_name, hpar[par_name])

        #kfold validation metrics tracking
        auc_kf_general = round(model_kfold['auc'],3)

        if not timeSeqValid:
            mlflow.log_metric("validation_nfolds", CrossValFolds)

            for fold in range(1,CrossValFolds+1):
                auc_kf_fold = round(model_kfold['AUC_fold_'+str(fold)],3)
                mlflow.log_metric("val_auc_fold_"+str(fold), auc_kf_fold)
        else:
            count=0
            for fold in model_kfold.keys():
                if 'AUC_fold_' in fold:
                    auc_seq_fold = round(model_kfold[fold],3)
                    mlflow.log_metric("val_auc_fold_"+str(count+1), auc_seq_fold)
                    count+=1
            mlflow.log_metric("validation_nfolds", count)


        mlflow.log_param("roc_val_fpr", list_to_string(model_kfold['fpr']))
        mlflow.log_param("roc_val_tpr", list_to_string(model_kfold['tpr']))
                    
        mlflow.log_metric("val_auc", auc_kf_general)

        #test metrics tracking
        auc = round(model_oos['auc'],3)
        rcm = model_oos['rcm']

        tnr, fpr, fnr, tpr = rcm.ravel()

        mlflow.log_metric("test_auc", auc)
        mlflow.log_metric("test_tpr", round(tpr,4)) #true positive rate
        mlflow.log_metric("test_fpr", round(fpr,4)) #false positive rate
        mlflow.log_metric("test_fnr", round(fnr,4)) #false negative ratee
        mlflow.log_metric("test_tnr", round(tnr,4)) #true negative rate

        cm = model_oos['cm']
        tn, fp, fn, tp = cm.ravel()
        mlflow.log_metric("test_tp", tp) #true positive rate
        mlflow.log_metric("test_fp", fp) #false positive rate
        mlflow.log_metric("test_fn", fn) #false negative ratee
        mlflow.log_metric("test_tn", tn) #true negative rate

        mlflow.log_param("roc_test_fpr", list_to_string(model_oos['fpr']))
        mlflow.log_param("roc_test_tpr", list_to_string(model_oos['tpr']))
        mlflow.log_param("test_predictions", list_to_string(model_oos['predictions']))


        #features importances tracking
        for f in feat_imp_dict.keys():
            mlflow.log_metric("f_"+f, feat_imp_dict[f])

        #MODEL STORAGE
        #storing the model file as pickle
        mlflow.sklearn.log_model(model, "model")

        #ARTIFACTS STORAGE
        #storing pipeline-processed trainset and testset
        mlflow.log_artifact(trainfilepath, "train_file")
        mlflow.log_artifact(testfilepath, "test_file")

        if save_results_for_viz:
            if dict_name!=None:
                mlflow.log_artifact(dict_name, "results_dict_for_viz")
            else:
                print("WARNING! - dictiname set to 'None'! Please change it to a meaningful output path. The dict viz output has not been saved in mlflow as artifact!")

        print("- Experiment tracked.")

#-----------------------------------------
# EXPERIMENT RETRIEVAL
#-----------------------------------------

to_transform_to_float = ['eta0', 'n_iter_no_change', 'max_iter', 'alpha', 'learning_rate', 'loss', 'val_auc', 'test_auc', 
                         'n_estimators', 'max_depth', 'max_features', 'max_leaf_nodes', 'min_samples_leaf', 'min_samples_split', 
                         'batch_size', 'epochs_actual', 'class_1_weight', 'hidden_layers_no', 
                         'hidden_nodes', 'tr_accuracy']



def create_exp_df(experiment, to_float_list = to_transform_to_float):
    """
    This function, given the experiment name, retrieves experiment data from mlflow and include them in a pandas dataframe
    """
    
    #retrieve experiment and runs
    e = mlflow.tracking.MlflowClient().get_experiment_by_name(experiment)
    filepath =e.artifact_location
    mypath = filepath.split('file:///')[1].replace("%20", " ")
    runids = [f for f in listdir(mypath) if isdir(join(mypath, f))]
    
    #retrieve metrics and parameters
    all_runs_params = []
    all_runs_metrics = []
    for run in runids:
        rundata = mlflow.tracking.MlflowClient().get_run(run).to_dictionary()['data']
        params = pd.DataFrame(rundata['params'].values(), index=rundata['params'].keys())
        metrics = pd.DataFrame(rundata['metrics'].values(), index=rundata['metrics'].keys())
        all_runs_params.append(params)
        all_runs_metrics.append(metrics)
    
    #create base df
    all_rows_params = set()
    for r in all_runs_params:
        all_rows_params = all_rows_params.union(set(r.index))

    all_rows_metrics = set()
    for r in all_runs_metrics:
        all_rows_metrics = all_rows_metrics.union(set(r.index))
        
    df_base_params = pd.DataFrame([np.NaN]*len(all_rows_params), index=all_rows_params)
    df_base_metrics = pd.DataFrame([np.NaN]*len(all_rows_metrics), index=all_rows_metrics)
    df_base = pd.concat([df_base_params, df_base_metrics])
    
    #add information to df
    runs_df_list = []

    runinfo = mlflow.tracking.MlflowClient().get_run(run).to_dictionary()['info']

    for run in runids:
        df_base_c = df_base.copy() 
        rundata = mlflow.tracking.MlflowClient().get_run(run).to_dictionary()['data']
        for param in rundata['params']:
            df_base_c.loc[param] = rundata['params'][param]
        for metric in rundata['metrics']:
            df_base_c.loc[metric] = rundata['metrics'][metric]
        for info in runinfo:
            if info in ('start_time', 'end_time'):
                df_base_c.loc[info] = [pd.to_datetime(runinfo[info], unit='ms')]
            else:
                df_base_c.loc[info] = [runinfo[info]]
        runs_df_list.append(df_base_c)
    viz = pd.concat(runs_df_list, axis=1).transpose().dropna(how='all') 
    viz = set_clf_cols(viz)

    for field in to_transform_to_float:
        if field in viz.index:
            try:
                viz.loc[field] = viz.loc[field].apply(pd.to_numeric, errors='coerce')
            except:
                pass

    viz.loc['tp_rate'] = viz.loc['test_tp']/(viz.loc['test_tp'] + viz.loc['test_fn'])
    viz.loc['tn_rate'] = viz.loc['test_tn']/(viz.loc['test_tn'] + viz.loc['test_fp'])
    viz.loc['fp_rate'] = viz.loc['test_fp']/(viz.loc['test_tp'] + viz.loc['test_tn'])
    viz.loc['fn_rate'] = viz.loc['test_fn']/(viz.loc['test_tn'] + viz.loc['test_tp'])

    return viz

#-----------------------------------------
# MAIN
#-----------------------------------------

def main():
    print("mlflow_utils.py executed/loaded..")

if __name__ == "__main__":
    main()