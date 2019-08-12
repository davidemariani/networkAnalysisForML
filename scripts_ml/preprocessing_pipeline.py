#==================================================================================#
# Author       : Davide Mariani                                                    #  
# University   : Birkbeck College, University of London                            # 
# Programme    : Msc Data Science                                                  #
# Script Name  : preprocessing_pipeline.py                                         #
# Description  : utils and scripts for data preprocessing                          #
# Version      : 0.2                                                               #
#==================================================================================#
# This file contains several functions to preprocess data for model feeding        #
# using scikit-learn                                                               #
#==================================================================================#

import numpy as np
import pandas as pd
import pickle
import datetime
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelBinarizer, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn_pandas import DataFrameMapper, gen_features

from scripts_ml.models_utils import *
from scripts_preproc.features_bond_graph_utils import *
#import itertools
#import sys
import os


#utils
class Date2Num(BaseEstimator, TransformerMixin):
    """
    Auxiliar class for scikit learn preprocessing module for converting datetimes to ordinal
    """
    def __init__(self):
        return
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        #print(X[0])
        nanidx = pd.isnull(X)
        X1 = np.zeros(X.shape)*np.nan
        X1[~nanidx] = [float(pd.Timestamp(x).toordinal()) for x in X[~nanidx]]
        return X1

class ReplaceImputer(BaseEstimator, TransformerMixin):
    """
    Auxiliar class for scikit learn preprocessing module for replacing nans
    """
    def __init__(self, replacewith=999):
        self.replacewith = replacewith
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X1 = X.copy()
        X1[np.isnan(X1)] = self.replacewith
        return X1


class LogScaler(BaseEstimator, TransformerMixin):
    """
    Auxiliar class for scikit learn preprocessing module for log-scaling quant features
    """
    def __init__(self, ZeroNegReplace=1e-5):
        self.ZeroNegReplace = ZeroNegReplace
        return
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X1=np.float32(X.copy())
        nanidx = np.isnan(X1)
        X2 = X1[~nanidx]
        X2[X2 < self.ZeroNegReplace] = self.ZeroNegReplace

        X1[~nanidx] = np.log(X2)
        return X1

class CapOutliers(BaseEstimator, TransformerMixin):
    """
    Auxiliar class for scikit learn preprocessing module for capping outliers greater than M standard dev
    """
    def __init__(self, Maxstd=3.):
        self.Maxstd = Maxstd
    def fit(self, X, y=None):
        self.mean = np.nanmean(X)
        self.std = np.nanstd(X)
        return self
    def transform(self, X):
        X1 = np.float32(X.copy())
        nanidx = np.isnan(X1)
        X2 = X1[~nanidx]
        #print("CapOutliers: {:} nans, {:} mean, {:} std".format(sum(nanidx), self.mean, self.std))
        bigvals =  (np.abs(X2 - self.mean) > self.Maxstd * self.std)
        X2[bigvals] = self.mean + self.Maxstd * self.std * np.sign(X2[bigvals] - self.mean)
        X1[~nanidx] = X2
        return X1



class SignLogScaler(BaseEstimator, TransformerMixin):
    """
    Auxiliar class for scikit learn preprocessing module that calculates the log for both positive and negative
    numbers, returning sign(x)*log(abs(x)+1) for the latter case.
    """

    def __init__(self):
        return
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X1=np.float32(X.copy())
        nanidx = np.isnan(X1)
        X1[~nanidx] = np.sign(X1[~nanidx]) * np.log(np.abs(X1[~nanidx])+1.)
        return X1


def features_pipeline(feat_str, feat_quant, feat_exp, feat_date):
    """
    This function, given the lists of different type of features, returns a scikit learn preprocessing pipeline
    ready to be used with the modeule 'transform'
    """
    #preparing groups of features for the pipeline
    feat_str = [[i] for i in feat_str]
    feat_quant = [[j] for j in feat_quant]
    feat_exp = [[k] for k in feat_exp]
    feat_date = [[l] for l in feat_date]

    #pipelines
    trans_date = gen_features(columns = feat_date,
                              classes = [{'class': Date2Num},
                                         {'class': CapOutliers, 'Maxstd': 4},
                                         {'class': SimpleImputer, 'strategy': "mean"},
                                         {'class': StandardScaler}])

    trans_quant = gen_features(columns =  feat_quant, 
                                   classes = [{'class': SimpleImputer, 'strategy': "mean"},
                                              {'class': CapOutliers, 'Maxstd': 4},
                                              {'class': StandardScaler}])

    trans_exp = gen_features(columns = feat_exp, 
                                   classes = [{'class': LogScaler, 'ZeroNegReplace': 1e-3},
                                              {'class': CapOutliers, 'Maxstd': 4},
                                              {'class': SimpleImputer, 'strategy': "mean"}, 
                                              {'class': StandardScaler}])

    trans_str = gen_features(columns = feat_str, 
                                 classes = [LabelBinarizer])

    preproc_pipeline = DataFrameMapper(trans_quant + trans_exp + trans_str + trans_date)

    return preproc_pipeline


def shuffle_train_test(df, trainsize, testsize, testset_control_feature):
    """
    This function, given a dataset, train and test size and a time control feature, will return train set and test set
    based on the initial dataset processed by scikitlearn StratifiedShuffleSplit module.
    Train set and test set will be created in shuffle mode, without any clear timewise 'cut'.
    """

    df = df.copy()

    print("Dropping {} instruments that are not due...".format(len(df[~df['is_due']])))
    df = df.copy()
    df = df[df.is_due]

    print("Sampling {:} for train and {:} for test sets by shuffling...".format(trainsize, testsize))

    df[testset_control_feature + "_year"] = df[testset_control_feature].apply(lambda x: x.year)

    split = StratifiedShuffleSplit(n_splits=1, 
                                    train_size = trainsize, 
                                    test_size = testsize, 
                                    random_state=42)

    df = df.reset_index(drop=True)
    
    #constructing oversampled class y=1 train and test sets:
    for train_index, test_index in split.split(df, df[testset_control_feature+"_year"]):
        train_all = df.loc[train_index]
        test_all = df.loc[test_index]

    return train_all, test_all



def time_train_test(df, testset_control_feature, testdate):
    """
    This function, given a datasetand a time control feature, will return train set and test set
    based on the initial dataset split by a date for which all the instruments before that date will form the train set,
    while all the instrument past that date will form the test set.
    """

    df = df.copy()

    print("Dropping {} instruments that are not due...".format(len(df[~df['is_due']])))
    df = df.copy()
    df = df[df.is_due]

    print("Splitting train and test sets by time, test cutoff: {:}...".format(testdate))

    test_all  = df.loc[df[testset_control_feature] >= testdate]
    train_all = df.loc[df[testset_control_feature] <  testdate]
    print("  {:}({:.1f}%) train, {:}({:.1f}%) test".format(train_all.shape[0], 100*train_all.shape[0]/df.shape[0],
                                                            test_all.shape[0],   100*test_all.shape[0]/df.shape[0]))

    return train_all, test_all



def idx_train_test(df, testset_control_feature, train_idx, test_idx):
    """
    This function, given a dataset, train indexes and test indexes, will return train set and test set
    """

    df = df.copy()

    print("Dropping {} instruments that are not due...".format(len(df[~df['is_due']])))
    df = df.copy()
    df = df[df.is_due]

    print("Splitting train and test sets by indexes...")

    test_set  = df.iloc[test_idx]
    train_set = df.iloc[train_idx]
    print("train from {} to {}, test from {} to {}".format(train_idx[0], train_idx[-1],
                                                            test_idx[0], test_idx[-1]))

    return train_set, test_set


def transform_train_test(train_all, test_all, preproc_pipeline, target_feature):
    """
    This function, given the train and test split generated with shuffle_train_test or time_train_test,
    will use a preproc_pipeline to transform the data returning train and test set ready for the models
    """

    print("Running the pipeline, target feature is {:}...".format(target_feature))

    #prepare and save train sets
    #separate features and labels
    y_train = train_all[target_feature].copy().values
    print("Train y: {:} total, {:} class_1 observations ({:.2f}%) > 0".format(y_train.shape[0], sum(y_train>0), sum(y_train>0)/y_train.shape[0]*100))
    #apply the pipeline to the training set
    print("pipeline fit_transform for train set...")
    X_train = preproc_pipeline.fit_transform(train_all)

    #prepare and save test sets
    #separate features and labels
    y_test = test_all[target_feature].copy().values
    print("Test y: {:} total, {:} class_1 observations ({:.2f}%) > 0".format(y_test.shape[0], sum(y_test>0), sum(y_test>0)/y_test.shape[0]*100))
    #apply the pipeline to the training set
    print("pipeline transform only for test set...")
    X_test = preproc_pipeline.transform(test_all) 

    #group the category labels together for charting
    feature_labels = preproc_pipeline.transformed_names_

    return y_train, X_train, y_test, X_test, feature_labels



def save_preproc_files(outputfolder, prefix, preproc_pipeline, y_train, X_train, y_test, X_test, feature_labels, fold_indexes=None):
    """
    This function will save all the preprocessing files into a given datafolder in pickle format
    """

    #creating reference for output file
    year = str(datetime.datetime.now().year)[2:]
    month = str(datetime.datetime.now().month)
    if len(month)==1:
        month = '0'+month
    day = str(datetime.datetime.now().day)

    postfix = '_'+year+month+day+'_'+str(datetime.datetime.now().hour)+str(datetime.datetime.now().minute)

    #saving training and test sets
    print("Saving with file name prefix {:} and postfix {:}...".format(prefix, postfix))
    pickle.dump([X_train, y_train, feature_labels], open(outputfolder+prefix+"_traindata" + postfix+'.pkl', "wb"), protocol=4)
    pickle.dump([X_test, y_test, feature_labels], open(outputfolder+prefix+"_testdata" + postfix+'.pkl', "wb"), protocol=4)
    pickle.dump(preproc_pipeline, open(outputfolder+prefix+"_preproc_pipeline" + postfix+'.pkl', "wb"))
    pickle.dump(feature_labels, open(outputfolder+prefix+"_feature_labels" + postfix+'.pkl', "wb"))
    if fold_indexes!=None:
        pickle.dump(fold_indexes, open(outputfolder+prefix+"_fold_indexes" + postfix+'.pkl', "wb"))
    print("...done.")



def preprocessing_pipeline(df, feat_str, feat_quant, feat_exp, feat_date, target_feature, testset_control_feature, experimentname, timewise = False,  
                           trainsize=None, testsize=None, testdate=datetime.datetime(2018, 4, 30), save_to_file=False, outputpath="../data/", prefix=''):
    """
    This function execute the whole preprocessing pipeline on a given dataframe, allowing the choice between timewise splitting and 
    shuffle splitting of the dataset between train and test with the boolean parameter 'timewise'.
    The function will save the files as pickle and return y train and test, x train and test and feature labels list.
    """
    preproc_pipeline = features_pipeline(feat_str, feat_quant, feat_exp, feat_date)

    prefix1 = '' #placeholder for dynamic prefix

    if timewise:
        prefix1 = 'time_'+str(testdate).split(' ')[0]+'_'
        train_all, test_all = time_train_test(df, testset_control_feature, testdate)

    else:
        prefix1 = 'shuffle_'
        train_all, test_all = shuffle_train_test(df, trainsize, testsize, testset_control_feature)

    y_train, X_train, y_test, X_test, feature_labels = transform_train_test(train_all, test_all, preproc_pipeline, target_feature)

    if save_to_file:
        outputfolder = outputpath+experimentname+'/'

        # Create target folder if it doesn't exist
        if not os.path.exists(outputfolder):
            os.mkdir(outputfolder)

        save_preproc_files(outputfolder, prefix1+prefix, preproc_pipeline, y_train, X_train, y_test, X_test, feature_labels)

    return y_train, X_train, y_test, X_test, feature_labels


def preproc_pipeline_timeseq(df, feat_str, feat_quant, feat_exp, feat_date, target_feature, testset_control_feature, experimentname, 
                             bg_settings_dicts, testdate=datetime.datetime(2018, 4, 30), train_window=12000, test_window=3000, #indexWise = False, 
                             whole_network_with_bg_file_path = "../data/04_instrumentsdf_bondgraph.pkl",
                             save_to_file=False, outputpath="../data/", prefix=''):
    """
    """
    prefix1 = 'time_'+str(testdate).split(' ')[0]+'_'
    preproc_pipeline = features_pipeline(feat_str, feat_quant, feat_exp, feat_date)

    print("---------Macro train split-----------")
    #macro-train split
    train_all = time_train_test(df, testset_control_feature, testdate)[0]
    train_all_bg = train_all.copy()

    count_1=0
    for set_dict in bg_settings_dicts:
        count_1+=1
        print("---------Adding bond graph features {} of {}-----------".format(count_1, len(bg_settings_dicts)))
        train_all_bg = add_bg_features(**{**{'df':train_all_bg}, **set_dict}) #adding bg features

    #test split with all bg features
    print("---------Macro test split-----------")
    full_df = pd.read_pickle(whole_network_with_bg_file_path)
    test_all = time_train_test(full_df, testset_control_feature, testdate)[1]

    print("---------Pipeline application-----------")
    y_train, X_train, y_test, X_test, feature_labels = transform_train_test(train_all_bg, test_all, preproc_pipeline, target_feature)

    print("---------Macro train-test saving-----------")
    if save_to_file:
        outputfolder = outputpath+experimentname+'/'

        #Create target folder if it doesn't exist
        if not os.path.exists(outputfolder):
            os.mkdir(outputfolder)

        save_preproc_files(outputfolder, prefix1+prefix, preproc_pipeline, y_train, X_train, y_test, X_test, feature_labels)

    print("---------Sequential validation splits-----------")
    #creating validation dataset
    T = train_all.shape[0]
    if train_window<1: #given as share of T
        train_window = np.floor(train_window*T)
    if test_window<1:
        test_window = np.floor(test_window*T)

    fold_generator = rolling_window(T, train_window, test_window)

    folds_idx = []

    valid_train = np.array([])

    #placeholders
    X_valid_train = []
    y_valid_train = []
    X_valid_test = []
    y_valid_test = []

    for count, train_idx, test_idx, Nsteps in fold_generator: #rolling window applied to training dataset for calibration
        print("---------Train test for validation fold {}-----------".format(count))
        train_idx = train_idx.astype(int)
        test_idx = test_idx.astype(int)

        #for creating test df, the whole network before the last test idx needs to be generated and used as input for bg extraction
        full_test_window_idx = np.concatenate([train_idx, test_idx], axis=None) #index of the train window + test window
        full_test_window_df = train_all.iloc[:test_idx[-1]+1]
        full_test_window_df_bg = full_test_window_df.copy()

        #for df train the last train index is considered
        df_train = train_all.iloc[train_idx]
        full_train_window_df = train_all.iloc[:train_idx[-1]+1]
        full_train_window_df_bg = full_train_window_df.copy()

        count_2 = 0
        for set_dict in bg_settings_dicts:
            count_2+=1

            print("---------Adding bond graph features {} of {} to TRAIN SET for fold {}-----------".format(count_2, len(bg_settings_dicts), count))
            full_train_window_df_bg = add_bg_features(**{**{'df':full_train_window_df_bg}, **set_dict})

            print("---------Adding bond graph features {} of {} to TEST SET for fold {}-----------".format(count_2, len(bg_settings_dicts), count))
            full_test_window_df_bg = add_bg_features(**{**{'df':full_test_window_df_bg}, **set_dict})

        df_train = full_train_window_df_bg.iloc[train_idx]
        print("Training dataset shape before preprocessing pipeline: {}".format(df_train.shape))
        df_test = full_test_window_df_bg.iloc[test_idx] 
        print("Test dataset shape before preprocessing pipeline: {}".format(df_test.shape))

        y_train_fold, X_train_fold, y_test_fold, X_test_fold, feature_labels = transform_train_test(df_train, df_test, preproc_pipeline, target_feature)

        #sanity check on feature columns consistency among folds
        if count==0:
            feature_labels_check = set(feature_labels)
        else:
            feature_labels_current_check = set(feature_labels)
            if feature_labels_check != feature_labels_current_check:
                if len(feature_labels_current_check)>len(feature_labels_check):
                    missing = set(feature_labels_check).difference(feature_labels_current_check)
                else:
                    missing = set(feature_labels_current_check).difference(feature_labels_check)

                print("WARNING! {} column seems to be missing!".format(missing))
            feature_labels_check = feature_labels_current_check

        #saving folds indexes
        folds_idx.append((train_idx, test_idx))

        if count==0:
            print("Creating first fold with X_train of shape {}, y_train of shape {}, X_test of shape {} and y_test of shape {}...".format(X_train_fold.shape, y_train_fold.shape, 
                                                                                                                                           X_test_fold.shape, y_test_fold.shape))
            X_valid_train = X_train_fold
            y_valid_train = y_train_fold
            X_valid_test = X_test_fold
            y_valid_test = y_test_fold
            print()

        else:
            print("Creating fold {} with X_train of shape {}, y_train of shape {}, X_test of shape {} and y_test of shape {}...".format(count, X_train_fold.shape, y_train_fold.shape, 
                                                                                                                                           X_test_fold.shape, y_test_fold.shape))
            X_valid_train = np.concatenate([X_valid_train, X_train_fold], axis=0)
            y_valid_train = np.concatenate([y_valid_train, y_train_fold], axis=0)
            X_valid_test = np.concatenate([X_valid_test, X_test_fold], axis=0)
            y_valid_test = np.concatenate([y_valid_test, y_test_fold], axis=0)
            print()

    if save_to_file:
        print("---------Saving sequential validation train and test-----------")
        outputfolder = outputpath+experimentname+'/'
        prefix_2 = '_val_'+str(train_window)+'_'+str(test_window)+'_'

        #Create target folder if it doesn't exist
        if not os.path.exists(outputfolder):
            os.mkdir(outputfolder)

        save_preproc_files(outputfolder, prefix1+prefix+prefix_2, preproc_pipeline, 
                           y_valid_train, X_valid_train, y_valid_test, X_valid_test, 
                           feature_labels, folds_idx)
        print()

    return y_train, X_train, y_test, X_test, feature_labels, y_valid_train, X_valid_train, y_valid_test, X_valid_test, folds_idx  


