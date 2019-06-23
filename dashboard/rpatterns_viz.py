#===================================================================================
# Author       : Davide Mariani                                                    #
# Company      : Tradeteq                                                          #
# Script Name  : rpatterns_viz.py                                                #
# Description  : Functions for repayment patterns visualization                    #
# Version      : 0.2                                                               #
#==================================================================================#
#==================================================================================# 
# It has been developed using bokeh 0.12.16                                        #
#==================================================================================#

#base modules
import pandas as pd
import numpy as np
import datetime
from datetime import date,datetime
import math
from matplotlib.dates import date2num, num2date


def startdate(trades, idate_col='invoice_date'):
    """
    This function returns the inception date of a given instrument
    """
    return trades[idate_col].toordinal()


def expdate(trades):
    """
    This function returns the due date of a given instrument
    """
    due_date = trades.due_date.toordinal()
    start_date = startdate(trades)
    
    if due_date<start_date:
        due_date = start_date
    
    return due_date



def firstrepdate(trades):
    """
    This function returns the first repayment date in a given instrument (if existing, nan otherwise)
    """
    try:
        res = min(trades.payment_date).toordinal()
    except ValueError:
        res= np.nan
    
    if res<startdate(trades) and res!=np.nan:
        res = startdate(trades)
    
    return res
  

def dueamount(trades):
    """
    This function returns the due amount in a given instrument
    """
    due = trades.invoice_amount
    return due

def enddate(trades, tolerance = .02):
    """
    This function returns the enddate of a given instrument, considering the last payment date if it has been paid fully (considering the tolerance),
    the last posting date otherwise.
    """
    Due = dueamount(trades)
    Rec = sum(trades.payment_amount)
    if (Rec > (1-tolerance)* Due): #deeming repaid
        enddate = trades.last_payment_date.toordinal()
    else: #not repaid
        if trades.is_open:
            enddate=np.nan #it was nan  
        else:
            if 'posting_date' in trades.index: #it was columns: investigate!
                enddate=max(trades.posting_date).toordinal()
            else:
                enddate=max(trades.posting_date).toordinal()
    return enddate



def set_the_patterns(tr, reportdate):
    """
    This function sets up the patterns fixing the limits of each bar before it is visualized in bokeh
    """

    df_tmp = tr.copy()

    #bar dimensions columns
    df_tmp['right1']=0.0
    df_tmp['left1']=0.0
    df_tmp['right2']=0.0
    df_tmp['left2']=0.0
    df_tmp['right3']=0.0
    df_tmp['left3']=0.0
    df_tmp['ed'] = 0.0
    df_tmp['fd'] =0.0
    
    
    y = range(len(df_tmp))#vertical bar position

    #trade info columns
    df_tmp.left1 = df_tmp.apply(startdate, axis=1)
    df_tmp.right1 = df_tmp.apply(expdate, axis=1)
    df_tmp.ed = df_tmp.apply(enddate, axis=1)
    df_tmp.fd = df_tmp.apply(firstrepdate, axis=1)

    df_tmp['repaid']=False
    df_tmp.repaid = df_tmp.right1<reportdate #due in the past

    #not overdue
    df_tmp.loc[df_tmp.repaid & ~(df_tmp.ed-df_tmp.right1>0) & ~(pd.isnull(df_tmp.ed)), 'left2'] = df_tmp.loc[df_tmp.repaid & ~(df_tmp.ed-df_tmp.right1>0) & ~(pd.isnull(df_tmp.ed)), 'right1']
    df_tmp.loc[df_tmp.repaid & ~(df_tmp.ed-df_tmp.right1>0) & ~(pd.isnull(df_tmp.ed)), 'right2'] = df_tmp.loc[df_tmp.repaid & ~(df_tmp.ed-df_tmp.right1>0) & ~(pd.isnull(df_tmp.ed)), 'left2']

    #repaid but later than due
    df_tmp.loc[df_tmp.repaid & (df_tmp.ed-df_tmp.right1>0), 'left2'] = df_tmp.loc[df_tmp.repaid & (df_tmp.ed-df_tmp.right1>0), 'right1']
    df_tmp.loc[df_tmp.repaid & (df_tmp.ed-df_tmp.right1>0), 'right2'] = df_tmp.loc[df_tmp.repaid & (df_tmp.ed-df_tmp.right1>0), 'ed']

    #still not repaid
    df_tmp.loc[df_tmp.repaid & (pd.isnull(df_tmp.ed)), 'left2']=df_tmp.loc[df_tmp.repaid & (pd.isnull(df_tmp.ed)), 'right1']
    df_tmp.loc[df_tmp.repaid & (pd.isnull(df_tmp.ed)), 'right2']=reportdate

    #not due as of reportdate
    df_tmp.loc[~df_tmp.repaid, ['left2', 'right2']] = df_tmp.loc[~df_tmp.repaid, 'right1'] 
    
    #early repayment
    df_tmp.loc[df_tmp.fd<df_tmp.right1, 'left3'] = df_tmp.loc[df_tmp.fd<df_tmp.right1, 'fd']
    df_tmp.loc[df_tmp.fd<df_tmp.right1, 'right3'] = df_tmp.loc[df_tmp.fd<df_tmp.right1, 'right1']

    right1 = df_tmp.right1.tolist()
    right2 = df_tmp.right2.tolist()
    right3 = df_tmp.right3.tolist()
    left1 = df_tmp.left1.tolist()
    left2 = df_tmp.left2.tolist()
    left3 = df_tmp.left3.tolist()

    return y, right1, right2, right3, left1, left2, left3

