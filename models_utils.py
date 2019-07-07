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
import datetime 


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
