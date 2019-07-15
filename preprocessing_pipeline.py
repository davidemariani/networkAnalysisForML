#==================================================================================#
# Author       : Davide Mariani                                                    #  
# University   : Birkbeck College, University of London                            # 
# Programme    : Msc Data Science                                                  #
# Script Name  : preprocessing_pipeline.py                                         #
# Description  : utils and scripts for data preprocessing                          #
# Version      : 0.2                                                               #
#==================================================================================#
# This file contains several functions to preprocess data for model feeding        #
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
import itertools
import sys


#utils
class Date2Num(BaseEstimator, TransformerMixin):
    """
    Auxiliar class for scikit learn preprocessing module for converting datetimes to float
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
    This function, given a dataset, train and test size and a time control feature, will return train set and test set
    based on the initial dataset split by a date for which all the instruments before that date will form the train set,
    while all the instrument past that date will form the test set.
    """

    print("Splitting train and test sets by time, test cutoff: {:}...".format(testdate))

    test_all  = df.loc[df[testset_control_feature] >= testdate]
    train_all = df.loc[df[testset_control_feature] <  testdate]
    print("  {:}({:.1f}%) train, {:}({:.1f}%) test".format(train_all.shape[0], 100*train_all.shape[0]/df.shape[0],
                                                            test_all.shape[0],   100*test_all.shape[0]/df.shape[0]))

    return train_all, test_all


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



def save_preproc_files(outputfolder, prefix, preproc_pipeline, y_train, X_train, y_test, X_test, feature_labels):
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
    print("...done.")



def preprocessing_pipeline(df, feat_str, feat_quant, feat_exp, feat_date, target_feature, testset_control_feature, timewise = False,  
                           trainsize=None, testsize=None, testdate=None, save_to_file=False, outputfolder='', prefix=''):
    """
    This function execute the whole preprocessing pipeline on a given dataframe, allowing the choice between timewise splitting and 
    shuffle splitting of the dataset between train and test with the boolean parameter 'timewise'.
    The function will save the files as pickle and return y train and test, x train and test and feature labels list.
    """
    preproc_pipeline = features_pipeline(feat_str, feat_quant, feat_exp, feat_date)

    prefix1 = ''

    if timewise:
        prefix1 = 'time_'+str(testdate).split(' ')[0]+'_'
        train_all, test_all = time_train_test(df, testset_control_feature, testdate)
    else:
        prefix1 = 'shuffle_'
        train_all, test_all = shuffle_train_test(df, trainsize, testsize, testset_control_feature)

    y_train, X_train, y_test, X_test, feature_labels = transform_train_test(train_all, test_all, preproc_pipeline, target_feature)

    if save_to_file:
        save_preproc_files(outputfolder, prefix1+prefix, preproc_pipeline, y_train, X_train, y_test, X_test, feature_labels)

    return y_train, X_train, y_test, X_test, feature_labels


