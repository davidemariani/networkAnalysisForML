#==================================================================================#
# Author       : Davide Mariani                                                    #  
# University   : Birkbeck College, University of London                            # 
# Programme    : Msc Data Science                                                  #
# Script Name  : models_utils.py                                                   #
# Description  : utils for modelling implementation and prototipation              #
# Version      : 0.1                                                               #
#==================================================================================#
# This file contains functions do implement and prototype ML models                #
#==================================================================================#

#importing main modules
import pandas as pd
import numpy as np
import pickle
import datetime as dt


def save_sk_model(datafolder, postfix, model_name):
    """
    This function saves a scikit learn model to pickle
    """

#saving sgd
#if (int(cfg["modelsave"])):
#        logpostfix = str(int(dt.datetime.now().timestamp()))[4:] #uniqueish name postfix
#        filename = datafolder + cfg["modelfolder"] + "linear" + logpostfix + ".pickle"
#        log.info("Saving model to " + filename)
#        with open(filename, "wb") as pickle_file:
#            pickle.dump(sgd_clf, pickle_file)


#saving rf
#if (int(cfg["modelsave"])):
#        logpostfix = str(int(dt.datetime.now().timestamp()))[4:] #uniqueish name postfix
#        filename = datafolder + cfg["modelfolder"] + "forest" + logpostfix + ".pickle"
#        log.info("Saving model to " + filename)
#        with open(filename, "wb") as pickle_file:
#            pickle.dump(forest_clf, pickle_file)