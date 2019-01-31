#==================================================================================#
# Author       : Davide Mariani                                                    #  
# University   : Birkbeck College, University of London                            # 
# Programme    : Msc Data Science                                                  #
# Script Name  : features_utils.py                                                 #
# Description  : utils for feature engineering                                     #
# Version      : 0.1                                                               #
#==================================================================================#
# This file contains several functions to add, process and enhance features        #
#==================================================================================#

#importing main modules
import pandas as pd
import numpy as np
import datetime
from scipy import stats


#utility functions
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
    if len(lst)==0:
        return res

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


def select_payment(x):
    """
    This function is used to select the payments in a time snapshot, where we need to exlude payments executed after the reportdate of that particular snapshot.
    It is meant to be used inside an 'apply' with axis=1 to be rowwise.
    """
    payments = x.payment_amount
    last_item = int(x.tmp_dates_to_count)
    return payments[:last_item]

def select_date(x):
    """
    This function is used to select the payment dates in a time snapshot, where we need to exlude dates happening after the reportdate of that particular snapshot.
    It is meant to be used inside an 'apply' with axis=1 to be rowwise.
    """
    dates = x.payment_date
    last_item = int(x.tmp_dates_to_count)
    return dates[:last_item]

def add_main_features(inst, ReportDate, impthr=0.009, imp2thr=0.04, purthr=0.009, dedthr=0.009, prefix='', date_debtor_sort=False):
    """
    This function add the main features to an input instruments dataframe, both in the general case and the snapshots creation systems.
    inst: instruments dataframe
    impthr: threshold for impairment1
    imp2thr: threshold for impairment2
    prefix: prefix to add to new columns (and to read the correct columns from different snapshots depending on their name)

    if prefix=='' it assumes that the input dataframe is the main one (not for snapshots purpose)
    """

    print('Addding main network features for snapshot with date < {}'.format(ReportDate))

    xor0 = np.vectorize(_xor0)

    #-----------------------------------------------------------------
    # Fields not affected (or not 'affectable') from snapshots systems
    #-----------------------------------------------------------------
    if prefix=='':
        #define the discharge loss as difference between invoice_amount and discharge amount...
        inst[prefix+"discharge_loss"] = xor0(inst["invoice_amount"] - inst["discharge_amount"])
        inst.loc[pd.isnull(inst["discharge_amount"]), "discharge_loss"] = 0. #...but it is 0 for NaN discharge_amount

        #define the presence of impairment1 as deduction_amount>0.009
        inst[prefix+"has_impairment1"] =  (inst["deduction_amount"]>impthr) & (inst["invoice_date"]<ReportDate)

        #define the presence of impairment2 as discharge_loss>0.009
        inst[prefix+"has_impairment2"] =  (inst["discharge_loss"]>impthr) & (inst["invoice_date"]<ReportDate)

        #sum of discharge_loss and deduction_amount
        inst[prefix+"total_impairment"] = xor0(inst["discharge_loss"]) + xor0(inst["deduction_amount"])

        #instrument with prosecution
        inst[prefix+"has_prosecution"] = inst["prosecution"].apply(lambda x: x=="Ja")

        #this indicates if an instrument has a purchase amount (if not, the client is not involved in repayment)
        inst[prefix+"has_purchase"] = inst["purchase_amount"].apply(lambda x: x>purthr)

        #this indicates if an instrument has a deduction amount
        inst[prefix+"has_deduction"] = inst["deduction_amount"].apply(lambda x: x>dedthr)

        #discharge amount
        inst[prefix+"has_discharge"] = inst["discharge_amount"]>0.001

    #-----------------------------------------------------------------
    # Fields affected from snapshots systems
    #-----------------------------------------------------------------

    #snapshot marker for selection
    if prefix!='':
        inst[prefix]=inst["invoice_date"]<=ReportDate

    #amount of the last payment for a certain instrument
    if prefix=='':
        inst["last_payment_amount"] = xor0(inst["payment_amount"].apply(lambda x: x[-1]))
    else:
        inst.loc[inst[prefix],'tmp_dates_to_count'] = inst.loc[inst[prefix],"payment_date"].apply(lambda x:sum(pd.Series(x)<ReportDate)) #this retrieve the index of the last payment snapshot to snapshot (it is a temp column)
        inst.loc[inst[prefix],prefix+"payment_date"] = inst.loc[inst[prefix],["payment_date", "tmp_dates_to_count"]].apply(select_date, axis=1)
        inst.loc[inst[prefix],prefix+"payment_amount"] = inst.loc[inst[prefix],["payment_amount", "tmp_dates_to_count"]].apply(select_payment, axis=1)
        inst.loc[inst[prefix],prefix+"last_payment_amount"] = xor0(inst.loc[inst[prefix],prefix+"payment_amount"].apply(lambda x: x[-1] if len(x)>0 else np.nan)) #last payment amount in this particular snapshot
        inst.loc[inst[prefix],prefix+"last_payment_date"] = inst.loc[inst[prefix],prefix+"payment_date"].apply(lambda x:x[-1] if len(x)>0 else pd.NaT)

    #sum of all the distinct entries for a single instrument
    if prefix=='':
        inst["total_repayment"] = xor0(inst["payment_amount"].apply(lambda x: sum(list(set(x))))) #sum of distinct entries
    else:
        inst.loc[inst[prefix],prefix+"total_repayment"] = xor0(inst.loc[inst[prefix],prefix+"payment_amount"].apply(lambda x: sum(list(set(x))))) #sum of distinct entries

    #instrument which are open and more than 90 days past the due date 
    if prefix=='': #base case without snapshots
        inst[prefix+"is_pastdue90"] =  inst["due_date"].apply(lambda x: (ReportDate - x).days > 90) & (inst["document_status"]=="offen")
    else:
        inst.loc[inst[prefix],prefix+"is_pastdue90"] =  inst.loc[inst[prefix],"due_date"].apply(lambda x: (ReportDate - x).days > 90) & (inst.loc[inst[prefix],prefix+"total_repayment"]<inst.loc[inst[prefix],"purchase_amount"]) #in this way fully repaid transactions won't be counted among the pastdues (kind of healing)

    #instrument which are open and more than 180 days past the due date
    if prefix=='':
        inst[prefix+"is_pastdue180"] =  inst["due_date"].apply(lambda x: (ReportDate - x).days > 180) & (inst["document_status"]=="offen")
    else:
        inst.loc[inst[prefix],prefix+"is_pastdue180"] =  inst.loc[inst[prefix],"due_date"].apply(lambda x: (ReportDate - x).days > 180) & (inst.loc[inst[prefix],prefix+"total_repayment"]<inst.loc[inst[prefix],"purchase_amount"])

    #mismatch between last payment date and due date
    if prefix=='':
        inst["payment_date_mismatch"] = (inst.last_payment_date - inst.due_date).dt.days
    else:
        inst.loc[inst[prefix],prefix+"payment_date_mismatch"] = (inst.loc[inst[prefix],prefix+"last_payment_date"] - inst.loc[inst[prefix],"due_date"]).dt.days

    #field indicating if an instrument is open or not
    if prefix=='':
        inst[prefix+"is_open"] = (inst["document_status"].apply(lambda x: x=="offen"))
    else:
        inst.loc[inst[prefix],prefix+"is_open"] = (inst.loc[inst[prefix],'invoice_date']<ReportDate) & (inst.loc[inst[prefix],prefix+"total_repayment"]<inst.loc[inst[prefix],"purchase_amount"])

    #weekend payment ratio
    if prefix=='':
        inst["we_payment_share"] = inst["payment_date"].apply(lambda x: we_share(x))
    else:
        inst.loc[inst[prefix],prefix+"we_payment_share"] = inst.loc[inst[prefix],prefix+"payment_date"].apply(we_share)

    #this field indicates if an instrument is due
    inst.loc[inst[prefix],prefix+"is_due"] = inst.loc[inst[prefix],"due_date"].apply(lambda x: x < ReportDate) & (inst.loc[inst[prefix],prefix+"total_repayment"]<inst.loc[inst[prefix],"purchase_amount"])

    if date_debtor_sort:
        #sort instruments dataset by invoice date and debtor id (time consuming - since the main df is already sorted, for snapshots this is not necessary)
        inst = inst.sort_values(by=["invoice_date", "debtor_id"], ascending=[True, True])

    


def series_trend(s, applylog=True):
    """
    This function defines a trend for a particular given series using linear regression.
    To be used with invoice_amount for the current dataset, in order to establish the entity of the transactions.
    """
    x=np.arange(s.shape[0])
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,s)
    #print(slope)
    if applylog:
        res = 0 if np.abs(slope)<1e-8 else np.sign(slope) * np.log(np.abs(slope))
    else:
        res = slope
    return res


def add_node_stats(inst, igroup, idx, id, ii, decision_date_col, prefix, prefix_read=''):
    """
    This function adds stats to each node, both in the general case and the snapshots creation systems.
    inst: instruments dataframe sorted by invoice_date
    igroup: group of instruments between a certain buyer and a certain seller
    idx: instrument index in the igroup 
    id: instrument id
    ii: instrument features (literally the dataset sliced in correspondence of that instrument)
    prefix: the prefix to add when creating a new df column
    prefix_read: the prefix to read when reading a new slice

    if prefix_read=='' it assumes that the input dataframe is the main one (not for snapshots purpose)
    """
    #adding counter of previously lent in this customer/debtor pair (inst is sorted by invoice date)
    inst.loc[id, prefix_read+prefix+"lent_c"] = idx 
        
    #adding counter of previously repaid instruments in this customer/debtor pair
    #to be repaid, the last payment date needs to be smaller than all the instrument date and the instrument needs to not be open
    repaid = (igroup.loc[:, prefix_read+"last_payment_date"] < ii[decision_date_col]) & (~ igroup.loc[:, prefix_read+"is_open"]) #filter for repaid instruments in this customer/debtor pair
    inst.loc[id, prefix_read+prefix+"repaid_c"] = sum(repaid) 
    
    if prefix_read=='':
        #adding counter of previously impaired in this customer/debtor pair
        inst.loc[id, prefix_read+prefix+"impaired1_c"] = sum(igroup.loc[repaid,prefix_read+"has_impairment1"])
        inst.loc[id, prefix_read+prefix+"impaired2_c"] = sum(igroup.loc[repaid,prefix_read+"has_impairment2"])
        
    #counter of overdue in this customer/debtor pair (considering previous instruments)
    previous = igroup.index[:idx] #previous instruments selector
    inst.loc[id, prefix_read+prefix+"pastdue90_c"] = sum((igroup.loc[previous,"due_date"] < ii[decision_date_col] - datetime.timedelta(90)) &
                                                        igroup.loc[previous,prefix_read+"is_pastdue90"])
    inst.loc[id, prefix_read+prefix+"pastdue180_c"] = sum((igroup.loc[previous,"due_date"] < ii[decision_date_col] - datetime.timedelta(180)) & 
                                                          igroup.loc[previous,prefix_read+"is_pastdue180"])
        
    #adding trend in amount lent in this customer/debtor pair
    inst.loc[id, prefix_read+prefix+"trend_a"] = 0 if idx<2 else series_trend(igroup.loc[previous,"invoice_amount"])
        
    #adding counter of weekend payments in this pair
    inst.loc[id, prefix_read+prefix+"we_payment_share"] = igroup.loc[repaid, prefix_read+"we_payment_share"].agg("mean")
        
    #adding payment_date_mismatch stats
    inst.loc[id, prefix_read+prefix+"pd_mismatch_mean"] = igroup.loc[repaid, prefix_read+"payment_date_mismatch"].agg("mean")
    inst.loc[id, prefix_read+prefix+"pd_mismatch_std"] = igroup.loc[repaid, prefix_read+"payment_date_mismatch"].agg("std") 