#==================================================================================#
# Author       : Davide Mariani                                                    #  
# University   : Birkbeck College, University of London                            # 
# Programme    : Msc Data Science                                                  #
# Script Name  : visualization_rep_patterns.py                                     #
# Description  : repayment patterns visualization                                  #
# Version      : 0.2                                                               #
#==================================================================================#
# This file contains bokeh scripts for visualizing repayment patterns between      #
# buyers and sellers                                                               #
#==================================================================================#

#base modules
import pandas as pd
import numpy as np
import datetime
from datetime import date,datetime
import math
from matplotlib.dates import date2num, num2date

#config
from locale import setlocale, getlocale, LC_ALL, atof
import locale
import sys
from myutils import *

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


def rep_patterns(tr, reportdate, plot_height=220, plot_width=220, datesnum = 10, title=''):
    """
    This function will visualize the repayment patterns using bokeh server
    """
    
    
    #setting the patterns
    y, right1, right2, right3, left1, left2, left3 = set_the_patterns(tr, reportdate)
    

    #columndatasource
    source= ColumnDataSource(data = {'y':y, 'right1':right1, 'left1':left1, 'right2':right2, 'left2':left2, 
                              'right3':right3, 'left3':left3,
                              'customers_name': tr['customer_name_1'],
                                     'customer_id':tr['customer_id'],
                             'debtor_name': tr['debtor_name_1'],
                                     'debtor_id':tr['debtors_id'],
                             'is_open': tr['is_open'],
                                'is_pastdue90': tr['is_pastdue90'],
                                'is_pastdue180': tr['is_pastdue180'],
                                'invoice_date':[i[0] for i in tr['invoice_date'].map(str).str.split(' ')],
                                'last_payment_date': [i[0] for i in tr['last_payment_date'].map(str).str.split(' ')],
                                'invoice_amount':tr['invoice_amount'].map(str),
                                'purchase_amount':tr.groupby(tr.index).apply(lambda x:str(np.nansum(x['purchase_amount']))),
                                'due_date':[i[0] for i in tr['due_date'].map(str).str.split(' ')],
                                'paid_amount':tr.groupby(tr.index).apply(lambda x:str(round(np.nansum([i for i in x['payment_amount']]),2))),
                                    'discharge_amount':tr.groupby(tr.index).apply(lambda x:str(np.nansum(x['discharge_amount']))),
                                    'purchase_amount_open':tr.groupby(tr.index).apply(lambda x:str(np.nansum(x['purchase_amount_open']))),
                                    'deduction_amount':tr.groupby(tr.index).apply(lambda x:str(np.nansum(x['deduction_amount']))),
                                    'has_impairment1':tr['has_impairment1'],
                                     'has_prosecution':tr['has_prosecution']
                                    })

    #axis ranges
    x_range = (np.min(source.data['left1']), max(reportdate+15, np.nanmax(source.data['right2'])))
    y_range = (-.5, max(source.data['y'])+.5)
    
    #tools
    _tools_to_show = 'box_zoom,pan,save,hover,tap,wheel_zoom'   #reset,
    
    #main fig
    fig = figure(x_range=x_range, y_range=y_range, 
                 plot_width=plot_width, plot_height=plot_height,
                tools = _tools_to_show, title=title)

    
    #hover tool
    TOOLTIPS = [('customer_name', '@customers_name'),
                      ('customer_id', '@customer_id'),
                     ('debtor_name', '@debtor_name'),
                      ('debtor_id', '@debtor_id'),
                      ('invoice_date', '@invoice_date'),
                      ('due_date', '@due_date'),
                      ('last_payment_date', '@last_payment_date'),
                      ('invoice_amount', '@invoice_amount'),
                      ('purchase_amount', '@purchase_amount'),
                      ('paid_amount', '@paid_amount'),
                      ('discharge_amount', '@discharge_amount'),
                      ('deduction_amount', '@deduction_amount'),
                      ('purchase_amount_open', '@purchase_amount_open'),
                     ('is_open', '@is_open'),
                     ('is_pastdue90', '@is_pastdue90'),
                      ('is_pastdue180', '@is_pastdue180'),
                     ('has_impairment1', '@has_impairment1'),
                     ('has_prosecution', '@has_prosecution')]

    
    hover = fig.select(dict(type=HoverTool))
    hover.tooltips = TOOLTIPS
    
    #customise figure attributes
    fig.background_fill_color = TTQcolor["PPTbg"]
    fig.background_fill_alpha = 1.
    fig.yaxis.visible = False

    
    linedict = {}
    fig.ygrid.grid_line_color=None
    fig.xgrid.grid_line_color=None
    multiline_count = 0

    #customer selection widget
    customer_name_vals = [n for n in tr['customer_name_1'].unique()]

    customer_name_options = [(i,i) for i in customer_name_vals]

    customer_select = Select(title='Customer', value=customer_name_vals[0], options = customer_name_options) 

    #debtors selection widget
    debtor_name_vals = [nd for nd in tr['debtor_name_1'].unique()]
    debtor_name_options = [(k,k) for k in debtor_name_vals]

    debtor_select = MultiSelect(title='Debtors', value=debtor_name_vals, options = debtor_name_options) 

    def select_df():
        cust_val = customer_select.value
        debt_val = debtor_select.value

        selected = tr[(tr['customer_name_1']==cust_val) & (tr['debtor_name_1'].isin(debt_val))]

        return selected

    def update():
        df_cust = select_df()

        new_debtor_name_vals = [nd for nd in tr.loc[tr['customer_name_1']==customer_select.value, 'debtor_name_1'].unique()]
        new_debtor_name_options = [(k,k) for k in new_debtor_name_vals]

        debtor_select.options = new_debtor_name_options

        new_y, new_right1, new_right2, new_right3, new_left1, new_left2, new_left3 = set_the_patterns(df_cust, reportdate)

        colors_ = [TTQcolor["warningRed"], TTQcolor["richOrange"]]
        rules = [df_cust['has_impairment1'], ~df_cust['has_impairment1']]
        imp_colors = np.select(rules, colors_) 

        source.data = {'y':new_y, 'right1':new_right1, 'left1':new_left1, 'right2':new_right2, 'left2':new_left2, 
                              'right3':new_right3, 'left3':new_left3,
                              'customers_name': df_cust['customer_name_1'],
                                     'customer_id':df_cust['customer_id'],
                                     'imp_colors':imp_colors,
                             'debtor_name': df_cust['debtor_name_1'],
                                     'debtor_id':df_cust['debtors_id'],
                             'is_open': df_cust['is_open'],
                                'is_pastdue90': df_cust['is_pastdue90'],
                                'is_pastdue180': df_cust['is_pastdue180'],
                                'invoice_date':[i[0] for i in df_cust['invoice_date'].map(str).str.split(' ')],
                                'last_payment_date': [i[0] for i in df_cust['last_payment_date'].map(str).str.split(' ')],
                                'invoice_amount':df_cust['invoice_amount'].map(str),
                                'purchase_amount':df_cust.groupby(df_cust.index).apply(lambda x:str(np.nansum(x['purchase_amount']))),
                                'due_date':[i[0] for i in df_cust['due_date'].map(str).str.split(' ')],
                                'paid_amount':df_cust.groupby(df_cust.index).apply(lambda x:str(round(np.nansum([i for i in x['payment_amount']]),2))),
                                    'discharge_amount':df_cust.groupby(df_cust.index).apply(lambda x:str(np.nansum(x['discharge_amount']))),
                                    'purchase_amount_open':df_cust.groupby(df_cust.index).apply(lambda x:str(np.nansum(x['purchase_amount_open']))),
                                    'deduction_amount':df_cust.groupby(df_cust.index).apply(lambda x:str(np.nansum(x['deduction_amount']))),
                                    'has_impairment1':df_cust['has_impairment1'],
                                     'has_prosecution':df_cust['has_prosecution']
                                    }

        #axis ranges
        x_range_new = (np.min(source.data['left1']), max(reportdate+15, np.nanmax(source.data['right2'])))
        y_range_new = (-.5, max(source.data['y'])+.5)

        #main fig ranges
        fig.x_range.start = x_range_new[0] 
        fig.x_range.end = x_range_new[1]
        fig.y_range.start = y_range_new[0]
        fig.y_range.end = y_range_new[1]

        #leftbar
        fig.hbar(y = 'y', left = 'left1', right='right1', height = .8,
                 fill_color=TTQcolor["background"], alpha=.95, line_alpha=0., source=source)
        #late repayment bar
        fig.hbar(y = 'y', left = 'left2', right='right2', height = .8, 
                 fill_color='imp_colors', alpha=.75, line_alpha=0., source=source)
        #early repayment bar
        fig.hbar(y = 'y', left = 'left3', right='right3', height = .8, 
                 fill_color=TTQcolor["algae"], alpha=.75, line_alpha=0., source=source)
        #report date line
        fig.line(x=[reportdate, reportdate], y=[min(y)-1, max(y)+1], 
                 line_color=TTQcolor["yellowOrange"], line_dash='dashed')

        #hover tool
        hover = fig.select(dict(type=HoverTool))
        hover.tooltips = TOOLTIPS


        #xaxis dates
        daystmp = np.arange(x_range_new[0], x_range_new[1]).tolist() #list of dates into the range
        no_of_dates = int(len(daystmp)/datesnum)-1 #step to use when selecting the daynums to display
        if no_of_dates<=1: #in case there are too few dates in the range:
            no_of_dates = len(daystmp) #just set the step to the current number of dates
        
        days_margin = 15 #to avoid overlaps with report date line
        if len(daystmp)>2:
            days_margin = datetime.fromordinal(int(daystmp[2])) - datetime.fromordinal(int(daystmp[1])) 
        else:
            days_margin = datetime.fromordinal(int(daystmp[1])) - datetime.fromordinal(int(daystmp[0]))

        if len(linedict.keys())>0:
            for k in linedict.keys():
                linedict[k].visible = False
        
        coords = {}
        coords['xs'] = [[d,d] for d in daystmp[::no_of_dates] if (datetime.fromordinal(int(d))+days_margin < datetime.fromordinal(int(reportdate)) or 
                                                                  datetime.fromordinal(int(d))-days_margin> datetime.fromordinal(int(reportdate)))]
        coords['ys'] = [[min(y)-1, max(y)+1] for d in daystmp[::no_of_dates] if (datetime.fromordinal(int(d))+days_margin < datetime.fromordinal(int(reportdate)) or 
                                                                                 datetime.fromordinal(int(d))-days_margin> datetime.fromordinal(int(reportdate)))]
        tmp_source = ColumnDataSource(data=coords)

        linedict['gridlines']= fig.multi_line(xs = 'xs', ys = 'ys', line_color=TTQcolor["yellowOrange"], line_alpha=0.5, source = tmp_source)

        daynums =  sorted([int(o) for o in daystmp[::no_of_dates] if datetime.fromordinal(int(o))+days_margin < datetime.fromordinal(reportdate)]+[int(reportdate)]) #conversion to int to make major label overrides work
        fig.xaxis.ticker = daynums
        fig.xaxis.major_label_overrides = dict(zip(daynums, [datetime.fromordinal(int(s)).strftime("%Y-%m-%d") for s in daynums]))
        fig.xaxis.major_label_orientation = -math.pi/4
        fig.ygrid.grid_line_color=None
        fig.xgrid.grid_line_color=None
 
    controls = [customer_select, debtor_select]
    for control in controls:
        control.on_change('value', lambda attrname, old, new: update())

    widgets = widgetbox(*controls)
    l = layout([widgets],
               [fig])

    return l






def rep_patterns_bs(tr, reportdate, plot_height=220, plot_width=220, datesnum = 10, title=''):
    """
    This function will visualize the repayment patterns using bokeh server
    """
    
    
    #setting the patterns
    y, right1, right2, right3, left1, left2, left3 = set_the_patterns(tr, reportdate)
    

    #columndatasource
    source= ColumnDataSource(data = {'y':y, 'right1':right1, 'left1':left1, 'right2':right2, 'left2':left2, 
                              'right3':right3, 'left3':left3,
                              'customers_name': tr['customer_name_1'],
                                     'customer_id':tr['customer_id'],
                             'debtor_name': tr['debtor_name_1'],
                                     'debtor_id':tr['debtors_id'],
                             'is_open': tr['is_open'],
                                'is_pastdue90': tr['is_pastdue90'],
                                'is_pastdue180': tr['is_pastdue180'],
                                'invoice_date':[i[0] for i in tr['invoice_date'].map(str).str.split(' ')],
                                'last_payment_date': [i[0] for i in tr['last_payment_date'].map(str).str.split(' ')],
                                'invoice_amount':tr['invoice_amount'].map(str),
                                'purchase_amount':tr.groupby(tr.index).apply(lambda x:str(np.nansum(x['purchase_amount']))),
                                'due_date':[i[0] for i in tr['due_date'].map(str).str.split(' ')],
                                'paid_amount':tr.groupby(tr.index).apply(lambda x:str(round(np.nansum([i for i in x['payment_amount']]),2))),
                                    'discharge_amount':tr.groupby(tr.index).apply(lambda x:str(np.nansum(x['discharge_amount']))),
                                    'purchase_amount_open':tr.groupby(tr.index).apply(lambda x:str(np.nansum(x['purchase_amount_open']))),
                                    'deduction_amount':tr.groupby(tr.index).apply(lambda x:str(np.nansum(x['deduction_amount']))),
                                    'has_impairment1':tr['has_impairment1'],
                                     'has_prosecution':tr['has_prosecution']
                                    })

    #axis ranges
    x_range = (np.min(source.data['left1']), max(reportdate+15, np.nanmax(source.data['right2'])))
    y_range = (-.5, max(source.data['y'])+.5)
    
    #tools
    _tools_to_show = 'box_zoom,pan,save,hover,tap,wheel_zoom'   #reset,
    
    #main fig
    fig = figure(x_range=x_range, y_range=y_range, 
                 plot_width=plot_width, plot_height=plot_height,
                tools = _tools_to_show, title=title)

    
    #hover tool
    TOOLTIPS = [('customer_name', '@customers_name'),
                      ('customer_id', '@customer_id'),
                     ('debtor_name', '@debtor_name'),
                      ('debtor_id', '@debtor_id'),
                      ('invoice_date', '@invoice_date'),
                      ('due_date', '@due_date'),
                      ('last_payment_date', '@last_payment_date'),
                      ('invoice_amount', '@invoice_amount'),
                      ('purchase_amount', '@purchase_amount'),
                      ('paid_amount', '@paid_amount'),
                      ('discharge_amount', '@discharge_amount'),
                      ('deduction_amount', '@deduction_amount'),
                      ('purchase_amount_open', '@purchase_amount_open'),
                     ('is_open', '@is_open'),
                     ('is_pastdue90', '@is_pastdue90'),
                      ('is_pastdue180', '@is_pastdue180'),
                     ('has_impairment1', '@has_impairment1'),
                     ('has_prosecution', '@has_prosecution')]

    
    hover = fig.select(dict(type=HoverTool))
    hover.tooltips = TOOLTIPS
    
    #customise figure attributes
    fig.background_fill_color = TTQcolor["PPTbg"]
    fig.background_fill_alpha = 1.
    fig.yaxis.visible = False

    
    linedict = {}
    fig.ygrid.grid_line_color=None
    fig.xgrid.grid_line_color=None
    multiline_count = 0

    #customer selection widget
    customer_name_vals = [n for n in tr['customer_name_1'].unique()]

    customer_name_options = [(i,i) for i in customer_name_vals]

    customer_select = Select(title='Customer', value=customer_name_vals[0], options = customer_name_options) 

    #debtors selection widget
    debtor_name_vals = [nd for nd in tr['debtor_name_1'].unique()]
    debtor_name_options = [(k,k) for k in debtor_name_vals]

    debtor_select = MultiSelect(title='Debtors', value=debtor_name_vals, options = debtor_name_options) 

    def select_df():
        cust_val = customer_select.value
        debt_val = debtor_select.value

        selected = tr[(tr['customer_name_1']==cust_val) & (tr['debtor_name_1'].isin(debt_val))]

        return selected

    def update():
        df_cust = select_df()

        new_debtor_name_vals = [nd for nd in tr.loc[tr['customer_name_1']==customer_select.value, 'debtor_name_1'].unique()]
        new_debtor_name_options = [(k,k) for k in new_debtor_name_vals]

        debtor_select.options = new_debtor_name_options

        new_y, new_right1, new_right2, new_right3, new_left1, new_left2, new_left3 = set_the_patterns(df_cust, reportdate)

        colors_ = [TTQcolor["warningRed"], TTQcolor["richOrange"]]
        rules = [df_cust['has_impairment1'], ~df_cust['has_impairment1']]
        imp_colors = np.select(rules, colors_) 

        source.data = {'y':new_y, 'right1':new_right1, 'left1':new_left1, 'right2':new_right2, 'left2':new_left2, 
                              'right3':new_right3, 'left3':new_left3,
                              'customers_name': df_cust['customer_name_1'],
                                     'customer_id':df_cust['customer_id'],
                                     'imp_colors':imp_colors,
                             'debtor_name': df_cust['debtor_name_1'],
                                     'debtor_id':df_cust['debtors_id'],
                             'is_open': df_cust['is_open'],
                                'is_pastdue90': df_cust['is_pastdue90'],
                                'is_pastdue180': df_cust['is_pastdue180'],
                                'invoice_date':[i[0] for i in df_cust['invoice_date'].map(str).str.split(' ')],
                                'last_payment_date': [i[0] for i in df_cust['last_payment_date'].map(str).str.split(' ')],
                                'invoice_amount':df_cust['invoice_amount'].map(str),
                                'purchase_amount':df_cust.groupby(df_cust.index).apply(lambda x:str(np.nansum(x['purchase_amount']))),
                                'due_date':[i[0] for i in df_cust['due_date'].map(str).str.split(' ')],
                                'paid_amount':df_cust.groupby(df_cust.index).apply(lambda x:str(round(np.nansum([i for i in x['payment_amount']]),2))),
                                    'discharge_amount':df_cust.groupby(df_cust.index).apply(lambda x:str(np.nansum(x['discharge_amount']))),
                                    'purchase_amount_open':df_cust.groupby(df_cust.index).apply(lambda x:str(np.nansum(x['purchase_amount_open']))),
                                    'deduction_amount':df_cust.groupby(df_cust.index).apply(lambda x:str(np.nansum(x['deduction_amount']))),
                                    'has_impairment1':df_cust['has_impairment1'],
                                     'has_prosecution':df_cust['has_prosecution']
                                    }

        #axis ranges
        x_range_new = (np.min(source.data['left1']), max(reportdate+15, np.nanmax(source.data['right2'])))
        y_range_new = (-.5, max(source.data['y'])+.5)

        #main fig ranges
        fig.x_range.start = x_range_new[0] 
        fig.x_range.end = x_range_new[1]
        fig.y_range.start = y_range_new[0]
        fig.y_range.end = y_range_new[1]

        #leftbar
        fig.hbar(y = 'y', left = 'left1', right='right1', height = .8,
                 fill_color=TTQcolor["background"], alpha=.95, line_alpha=0., source=source)
        #late repayment bar
        fig.hbar(y = 'y', left = 'left2', right='right2', height = .8, 
                 fill_color='imp_colors', alpha=.75, line_alpha=0., source=source)
        #early repayment bar
        fig.hbar(y = 'y', left = 'left3', right='right3', height = .8, 
                 fill_color=TTQcolor["algae"], alpha=.75, line_alpha=0., source=source)
        #report date line
        fig.line(x=[reportdate, reportdate], y=[min(y)-1, max(y)+1], 
                 line_color=TTQcolor["yellowOrange"], line_dash='dashed')

        #hover tool
        hover = fig.select(dict(type=HoverTool))
        hover.tooltips = TOOLTIPS


        #xaxis dates
        daystmp = np.arange(x_range_new[0], x_range_new[1]).tolist() #list of dates into the range
        no_of_dates = int(len(daystmp)/datesnum)-1 #step to use when selecting the daynums to display
        if no_of_dates<=1: #in case there are too few dates in the range:
            no_of_dates = len(daystmp) #just set the step to the current number of dates
        
        days_margin = 15 #to avoid overlaps with report date line
        if len(daystmp)>2:
            days_margin = datetime.fromordinal(int(daystmp[2])) - datetime.fromordinal(int(daystmp[1])) 
        else:
            days_margin = datetime.fromordinal(int(daystmp[1])) - datetime.fromordinal(int(daystmp[0]))

        if len(linedict.keys())>0:
            for k in linedict.keys():
                linedict[k].visible = False
        
        coords = {}
        coords['xs'] = [[d,d] for d in daystmp[::no_of_dates] if (datetime.fromordinal(int(d))+days_margin < datetime.fromordinal(int(reportdate)) or 
                                                                  datetime.fromordinal(int(d))-days_margin> datetime.fromordinal(int(reportdate)))]
        coords['ys'] = [[min(y)-1, max(y)+1] for d in daystmp[::no_of_dates] if (datetime.fromordinal(int(d))+days_margin < datetime.fromordinal(int(reportdate)) or 
                                                                                 datetime.fromordinal(int(d))-days_margin> datetime.fromordinal(int(reportdate)))]
        tmp_source = ColumnDataSource(data=coords)

        linedict['gridlines']= fig.multi_line(xs = 'xs', ys = 'ys', line_color=TTQcolor["yellowOrange"], line_alpha=0.5, source = tmp_source)

        daynums =  sorted([int(o) for o in daystmp[::no_of_dates] if datetime.fromordinal(int(o))+days_margin < datetime.fromordinal(reportdate)]+[int(reportdate)]) #conversion to int to make major label overrides work
        fig.xaxis.ticker = daynums
        fig.xaxis.major_label_overrides = dict(zip(daynums, [datetime.fromordinal(int(s)).strftime("%Y-%m-%d") for s in daynums]))
        fig.xaxis.major_label_orientation = -math.pi/4
        fig.ygrid.grid_line_color=None
        fig.xgrid.grid_line_color=None
 
    controls = [customer_select, debtor_select]
    for control in controls:
        control.on_change('value', lambda attrname, old, new: update())

    widgets = widgetbox(*controls)
    l = layout([widgets],
               [fig])

    update()
    
    curdoc().add_root(l)
    curdoc().title = 'Repayment Patterns'


def main():
    # --------------------------------------------------------------------------------
    # Log initialisation
    # --------------------------------------------------------------------------------

    _modulename = 'repayment_patterns_interactive'

    # Process command line arguments
    if len(sys.argv)>1:
        cfgname = sys.argv[1]
        print(cfgname)
    else:
        cfgname="DEFAULT"

    LogPath = getLogPath(_modulename)
    CfgPath = getCfgPath(_modulename)

    #sanity check
    if fileExists(CfgPath):
        cfg = getconfig(_modulename, cfgname)
    else:
        print("Config file " + CfgPath + " not found. Exiting ...")
        quit(2)

    log = setuplog(_modulename, cfg)
    datafolder = cfg["datafolder"] 
    repdate = cfg["reportdate"]
    filename = cfg["filename"]
    # --------------------------------------------------------------------------------
    # Import data
    # --------------------------------------------------------------------------------

    print("Reading the input dataset...")
    df = pd.read_pickle(datafolder + filename)

    #date we received the data
    ReportDate = pd.to_datetime(repdate, yearfirst=True)
    ReportDateOrd = ReportDate.toordinal()

    #columns names fixing
    col_names = []

    #some modification of the input columns 
    for j in df.columns:
        if j not in ['customer_name_1.1', 'debtor_id']:
            col_names.append(j)
        else:
            if j=='debtor_id':
                col_names.append('debtors_id')
            elif j=='customer_name_1.1':
                col_names.append('debtor_name_1')

    df.columns = col_names

    # --------------------------------------------------------------------------------
    # Visualization execution
    # --------------------------------------------------------------------------------
    rep_patterns_bs(df, ReportDateOrd, 800, 800)

if __name__ == '__main__':
    main()

