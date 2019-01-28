#==================================================================================#
# Author       : Davide Mariani                                                    #  
# University   : Birkbeck College, University of London                            # 
# Programme    : Msc Data SCience                                                  #
# Script Name  : features_utils.py                                                 #
# Description  : utils for feature engineering                                     #
# Version      : 0.1                                                               #
#==================================================================================#
# This file contains several functions to add, process and enhance features        #
#==================================================================================#

import pandas as pd
import numpy as np

def _xor0(x):
    """
    This function replaces nans with 0
    """
    return 0. if np.isnan(x) else x
xor0 = np.vectorize(_xor0)


def we_share(lst):
    """
    This function return the ratio of weekend payments for an instrument. nan if there's no weekend payment.
    """
    res = np.nan
    wec = 0
    datec = 0
    for x in lst:
        if not pd.isnull(x):
            datec+=1
            if x.weekday()>4:
                wec+=1
    if datec>0:
        res=wec/datec
    return res


def add_main_features(inst, ReportDate, impthr=0.009, imp2thr=0.04, purthr=0.009, dedthr=0.009, prefix=''):
    """
    This function add the main features to an input instruments dataframe
    inst: instruments dataframe
    impthr: threshold for impairment1
    imp2thr: threshold for impairment2
    prefix: prefix to add to new columns (and to read the correct columns from different snapshots depending on their name)
    """

    xor0 = np.vectorize(_xor0)

    #define the discharge loss as difference between invoice_amount and discharge amount...
    inst[prefix+"discharge_loss"] = xor0(inst[prefix+"invoice_amount"] - inst[prefix+"discharge_amount"])
    inst.loc[pd.isnull(inst[prefix+"discharge_amount"]), prefix+"discharge_loss"] = 0. #...but it is 0 for NaN discharge_amount

    #define the presence of impairment1 as deduction_amount>0.009
    inst[prefix+"has_impairment1"] =  inst[prefix+"deduction_amount"]>impthr

    #define the presence of impairment2 as discharge_loss>0.009
    inst[prefix+"has_impairment2"] =  inst.discharge_loss>impthr

    #define the presence of impairment3 as discharge_loss>proportion of invoice amount or deduction_amount>proportion of invoice amount
    inst[prefix+"has_impairment3"] =  (inst[prefix+"discharge_loss"]>imp2thr*inst[prefix+"invoice_amount"]) | (inst[prefix+"deduction_amount"]>imp2thr*inst[prefix+"invoice_amount"])

    #instrument which are open and more than 90 days past the due date 
    inst[prefix+"is_pastdue90"] =  inst[prefix+"due_date"].apply(lambda x: (ReportDate - x).days > 90) & (inst[prefix+"document_status"]=="offen")

    #instrument which are open and more than 180 days past the due date
    inst[prefix+"is_pastdue180"] =  inst[prefix+"due_date"].apply(lambda x: (ReportDate - x).days > 180) & (inst[prefix+"document_status"]=="offen")

    #instrument with prosecution
    inst[prefix+"has_prosecution"] = inst[prefix+"prosecution"].apply(lambda x: x=="Ja")

    #amount of the last payment for a certain instrument
    inst[prefix+"last_payment_amount"] = xor0(inst[prefix+"payment_amount"].apply(lambda x: x[-1]))

    #sum of all the distinct entries for a single instrument
    inst[prefix+"total_repayment"] = xor0(inst[prefix+"payment_amount"].apply(lambda x: sum(list(set(x))))) #sum of distinct entries

    #sum of discharge_loss and deduction_amount
    inst[prefix+"total_impairment"] = xor0(inst[prefix+"discharge_loss"]) + xor0(inst[prefix+"deduction_amount"])

    #field indicating if an instrument is open or not
    inst[prefix+"is_open"] = inst[prefix+"document_status"].apply(lambda x: x=="offen")

    #sort instruments dataset by invoice date and debtor id
    inst = inst.sort_values(by=[prefix+"invoice_date", prefix+"debtor_id"], ascending=[True, True])

    #weekend payment ratio
    inst[prefix+"we_payment_share"] = inst[prefix+"payment_date"].apply(lambda x: we_share(x))

    #this indicates if an instrument has a purchase amount (if not, the client is not involved in repayment)
    inst[prefix+"has_purchase"] = inst[prefix+"purchase_amount"].apply(lambda x: x>purthr)

    #this indicates if an instrument has a deduction amount
    inst[prefix+"has_deduction"] = inst[prefix+"deduction_amount"].apply(lambda x: x>dedthr)

    #this field indicates if an instrument is due
    inst[prefix+"is_due"] = inst[prefix+"due_date"].apply(lambda x: x < ReportDate)

    #discharge amount
    inst[prefix+"has_discharge"] = inst[prefix+"discharge_amount"]>0.001