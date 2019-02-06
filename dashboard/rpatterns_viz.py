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
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from datetime import date,datetime
import math
from matplotlib.dates import date2num, num2date

#bokeh
from bokeh.io import show, output_notebook, output_file, curdoc
from bokeh.plotting import figure
from bokeh.layouts import gridplot, layout, widgetbox
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    Range1d,
    Plot)
from bokeh.models.glyphs import MultiLine
from bokeh.models.widgets import Select, MultiSelect


#TTQ colors dictionary
TTQcolor = {
    'font': '#353535',
    'lightGrey':'#fafafa',
    'borderColour' : '#707070',
    'cream' : '#edece8',
    'lightCream' : '#F8F7F7',
    'background' : '#ffffff',
    'link' : '#3B5A95',
    'brightLink':'#FE4045',
    'marketplaceOrange' : '#F6601E',
    'warningRed' : '#ce130b',
    'Salmon' : '#F1B096',
    'victorian' : '#CEDFDD',
    'cream' : '#EFEFDC',
    'whiteGrey' : '#EDECE9',
    'eightyGrey' : '#D1D0CF',
    'blueGrey' : '#939EA9',
    'sixtyGrey' : '#8E9494',
    'navy' : '#143154',
    'darkPurple' : '#323651',
    'darkNavy' : '#0E2335',
    'darkCyan' : '#0D2B2C',
    'redBrown' : '#6B191E',
    'richBrown': '#85372B',
    'algae' : '#0E6930',
    'mutedBlue' : '#496592',
    'azureBlue' : '#155C8A',
    'electric' : '#67BDCE',
    'sky' : '#76B9D0',
    'turq' : '#1A9E78',
    'pea' : '#94CB91',
    'ocean' : '#55B7BB',
    'richOrange' : '#FB8C36',
    'richPeach' : '#F66B53',
    'yell' : '#F9EE16',
    'yellowOrange' : '#F6BD4F',
    'peach' : '#F2B097',
    'bloodRed' : '#CD130B',
    "PPTbg": '#19242F'
  }


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
    Ni = len(tr) #no of different instruments in tr
    
    #empty arrays for coords
    y=range(Ni)#vertical bar position
    right1=np.zeros(Ni) #first bar: expected span
    left1= np.zeros(Ni)
    right2=np.zeros(Ni) #second bar: overdue period
    left2= np.zeros(Ni)
    right3=np.zeros(Ni) #third bar: paid early period
    left3= np.zeros(Ni)


    #filling the values
    for idx in range(len(tr)): #for each instrument, retrieve all the coordinates for the bars
        trades = tr.iloc[idx] #selection of each instrument
        
        #main dates
        left1[idx] = startdate(trades)
        right1[idx] = expdate(trades)
        ed = enddate(trades)
        fd = firstrepdate(trades)
        
        #conditions:
        if right1[idx] < reportdate: #is due in the past
            if ed - right1[idx] > 0: #repaid but later than due
                left2[idx] = right1[idx]
                right2[idx] = ed
            elif np.isnan(ed): #still not repaid
                left2[idx] = right1[idx]
                right2[idx] = reportdate
            else: #not overdue
                left2[idx]=right1[idx]
                right2[idx]=left2[idx]
        else: #not due as of reportdate
                left2[idx]=right1[idx]
                right2[idx]=left2[idx]
        if fd < right1[idx]: #first repayment earlier than due
            left3[idx] = fd
            right3[idx] = right1[idx]

    return y, right1, right2, right3, left1, left2, left3


