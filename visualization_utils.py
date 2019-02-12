#==================================================================================#
# Author       : Davide Mariani                                                    #  
# University   : Birkbeck College, University of London                            # 
# Programme    : Msc Data SCience                                                  #
# Script Name  : visualization_utils.py                                            #
# Description  : utils for data visualizations                                     #
# Version      : 0.2                                                               #
#==================================================================================#
# This file contains general purpose visualization functions initially based on    #
# bokeh (v.0.12.16). Recently updated to bokeh v.1.0.4                             #
#==================================================================================#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math


#bokeh
from bokeh.plotting import figure
from bokeh.models import LinearAxis, Range1d, SingleIntervalTicker, AdaptiveTicker, ColumnDataSource, LabelSet, HoverTool, Label
from bokeh.models.formatters import BasicTickFormatter
from bokeh.models.glyphs import Text
from numpy import histogram, linspace
from scipy.stats.kde import gaussian_kde
from bokeh.layouts import gridplot, column


#colors library
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

##UTILS

def convert_to_eur(val, curr):
    """
    This function, given a certain numeric amount and its currency from the thesis dataset,
    converts it to EURO.
    """

    #please note the conversion rate are hard coded, check 
    #conversion to dollar first
    if curr == 'Schweizer Franken':
        to_usd_rate = round(1.0/0.9916,4)
    elif curr == 'Euro':
        return val #in case the amount is in euro already, return it
    elif curr == 'US-Dollar':
        to_usd_rate = 1.0
    elif curr == 'Britisches Pfund':
        to_usd_rate = 1.3154
    
    #conversion from dollar to euro
    from_usd_to_eur_rate = round(1.0/1.1559,4)
    
    return round(val*to_usd_rate*from_usd_to_eur_rate,2)

def count_and_amount(df):
    """
    This function, given the desired dataframe slice, will return the count of instruments and the invoice amount
    for each month.
    """
    count = {}
    amount = {}
    
    years = range(2013,2019)
    months = range(1,13)
    
    for y in years:
        for m in months:
            if (y!=2013 or m not in range(1,8)): #conditions for GoFactoring dataset
                if (y!=2018 or m<10):
                    count[str(y)+'_'+str(m)] = len(df[df['value_date'].apply(lambda x:(x.year==y) & (x.month==m))])
                    amount[str(y)+'_'+str(m)] = df[df['value_date'].apply(lambda x:(x.year==y) & (x.month==m))]['invoice_amount'].sum()
    
    #dataframes creation
    count_df = pd.DataFrame(count, index = ['count'])
    amount_df = pd.DataFrame(amount, index = ['amount'])
    
    count_amount_df = pd.concat([count_df, amount_df])
    
    return count_df, amount_df, count_amount_df


def show_stats(inst, buyer, seller):
    """
    It shows some basic statistics about the provided instruments dataframe and its respective buyers and sellers dataframe.
    It is useful for a first look when comparing similar datasets.
    """
    Ni = inst.shape[0]
    print("{:} instruments,".format(Ni))
    fc = ['has_purchase', 'is_open', 'has_impairment1', 'has_impairment2', 'has_impairment3', "is_due",
           'is_pastdue90', 'is_pastdue180', 'has_prosecution', 'has_deduction']
    for c in fc:
          print("  {:}: {:} ({:2.1f}%)".format(c, sum(inst[c]), sum(100*inst[c])/Ni))

    fc = ['has_impairment1',
           'has_impairment2', 'is_pastdue90', 'is_pastdue180', 'has_prosecution',
           'is_open']
    Nb = buyer.shape[0]
    print("{:} buyers,".format(Nb))
    print("  {:.2f} sellers per buyer".format(buyer.customer_name_1.mean()))
    print("  {:.2f} instruments per buyer".format(buyer.uid.mean()))
    for c in fc:
        print("    {:}: {:.1f} average instruments, {:}({:.1f}%) with some".format(
              c, buyer[c].mean(), sum(buyer[c] > 0), 100*sum(buyer[c]>0)/Nb))

    Ns = seller.shape[0]
    print("{:} sellers,".format(Ns))
    print("  {:.2f} buyers per seller".format(seller.debtor_name_1.mean()))
    print("  {:.2f} instruments per seller".format(seller.uid.mean()))
    for c in fc:
        print("    {:}: {:.1f} average instruments, {:}({:.1f}%) with some".format(
              c, seller[c].mean(), sum(seller[c] > 0), 100*sum(seller[c]>0)/Ns))

#DATASETS COMPARISON

#base empty dataframe structure
report = pd.DataFrame(columns=["label"]).set_index("label")

def addreprow(df, column, label, value):
    """
    Auxiliar function for save_stats
    """
    df.at[label, column] = value

def save_stats(column, inst, buyer, seller, df = report):
    """
    This function creates a dataframe of basic stats for the thesis project dataset
    """
    Ni = inst.shape[0]
    addreprow(df, column,  "no of instruments", str(Ni))
    fc = ['has_purchase', 'is_open', 'has_impairment1', 'has_impairment2', 'has_impairment3', "is_due",
           'is_pastdue90', 'is_pastdue180', 'has_prosecution', 'has_deduction']
    for c in fc:
        if Ni!=0:
            f = sum(100*inst[c])/Ni
        else:
            f = 0.0
        addreprow(df, column,  c, "{:} ({:2.1f}%)".format(sum(inst[c]), f))

    fc = ['has_impairment1',
           'has_impairment2', 'is_pastdue90', 'is_pastdue180', 'has_prosecution',
           'is_open']
    Nb = buyer.shape[0]
    addreprow(df, column, "no of buyers", str(Nb))
    addreprow(df, column, "sellers per buyer", "{:.2f}".format(buyer.customer_name_1.mean()))
    addreprow(df, column, "instruments per buyer",   "{:.2f}".format(buyer.uid.mean()))
    for c in fc:
        if Nb!=0:
            g = 100*sum(buyer[c]>0)/Nb
        else:
            g = 0.0
        addreprow(df, column,  c+" average instruments per buyer", "{:.1f}".format(buyer[c].mean()))
        addreprow(df, column,  c+" buyers with some", "{:} ({:.1f}%)".format(sum(buyer[c] > 0), 
                                                                        g) )        
    Ns = seller.shape[0]
    addreprow(df, column, "no of sellers", str(Ns))
    addreprow(df, column, "buyers per seller", "{:.2f}".format(seller.debtor_name_1.mean()))
    addreprow(df, column, "instruments per seller",   "{:.2f}".format(seller.uid.mean()))
    for c in fc:
        if Ns!=0:
            h = 100*sum(seller[c]>0)/Ns
        else:
            h = 0.0
        addreprow(df, column,  c+" average instruments per seller", "{:.1f}".format(seller[c].mean()))
        addreprow(df, column,  c+" sellers with some", "{:} ({:.1f}%)".format(sum(seller[c] > 0), h))
    
    return df



##VISUALS

def distplot(series,title = '', yaxisname = '', xaxisname = '', density = False, bins=50,
             range = None, color = TTQcolor['azureBlue'], plot_w = 1000, plot_h=750, boxplot=False,
             boxtext = True, boxtextsize = '8pt', box_outliers=True, outliers_size = 2):
    """
    This function creates a distribution plot given a series.
    If range is specified, it will affect how the bins are used.
    If boxplot = True, it will create a boxplot for the series, using the function boxplot_hor
    """

    f = figure(plot_width=plot_w, plot_height=plot_h, title = title)
    if range!=None:
        hist, edges = histogram(series, density=density, bins=bins, range=(range[0], range[1]))
    else:
        hist, edges = histogram(series, density=density, bins=bins)
    
    f.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], alpha=0.4, color = color)
    f.yaxis.axis_label = yaxisname
    f.xaxis.axis_label = xaxisname

    if boxplot:
        b = boxplot_hor(series, xrange=f.x_range, fill_color = color, 
                        btext = boxtext, textsize = boxtextsize, outliers = box_outliers, outliers_size=outliers_size)
        l = column(b,f)
        return l

    return f

def stacked_distplot(*args, title = '', yaxisname = '', xaxisname = '', density = False, bins=50
                     , colors = [TTQcolor['azureBlue'], TTQcolor['algae']], plot_w = 1000, plot_h=750,
                     alphacolor = 0.5, legendnames =  ['',''], legendlocation = 'top_left',
                     logscale = False, boxplot = False, boxtext = True, boxtextsize = '8pt', box_outliers=True,
                     outliers_size = 2):
    """
    Given an unspecified number of series as main arguments,
    this function generates a distribution plot, comparing them with a stack plot.
    If boxplot = True, it will create a boxplot for each series, using the function boxplot_hor
    """
    f = figure(plot_width=plot_w, plot_height=plot_h, title = title)

    if logscale:
        f = figure(plot_width=plot_w, plot_height=plot_h, title = title, x_axis_type = 'log')
    
    min_range=min([min(i) for i in args])
    max_range=max([max(j) for j in args])

    hists = []
    edges = []

    for k in range(len(args)):
        hist_, edges_ = histogram(args[k], density=density, bins=bins, range=(min_range, max_range))
        hists.append(hist_)
        edges.append(edges_)

        if k==0:
            f.quad(top=hist_, bottom=0, left=edges_[:-1], right=edges_[1:], alpha=alphacolor, color=colors[k],
                   legend = legendnames[k])
        else:
            f.quad(top=hists[k]+hists[k-1], bottom=hists[k-1], left=edges[k][:-1], right=edges[k][1:],
                   alpha = alphacolor, color=colors[k], legend=legendnames[k])
        
    f.yaxis.axis_label = yaxisname
    f.xaxis.axis_label = xaxisname
    f.legend.location = legendlocation
    f.legend.click_policy = 'hide'

    if boxplot:
        grid = []
        for s in range(len(args)):
            b = boxplot_hor(args[s], xrange=f.x_range, fill_color = colors[s], logscale = logscale, btext = boxtext, 
                            textsize = boxtextsize, outliers = box_outliers, outliers_size=outliers_size)
            grid.append([b])
        grid.append([f])
        l = gridplot(grid)
        return l
    return f


def timeSeries(df, colors, title, plot_w=1800, plot_h =450, showY=True, rightY=False, Ylabel='', Xlabel='',
              custom_Y_left = None, custom_Y_right = None, Ylabel_2='', output_file_name='',
              showXGrid = True, showYGrid=True, legend_names = [],custom_Y_interval_right = 2,
              custom_Y_precision_right = 0,  custom_Y_scientific_right = False, custom_Y_interval_left = 500,
              custom_Y_precision_left = 0,  custom_Y_scientific_left = False,
              legend_font_size = '10pt', axis_label_font_size = '10pt', x_axis_ticks_font_size='8pt',
              y_axis_ticks_font_size = '8pt', boxtextplot = '8pt'): 
    """
    df: input dataframe from which taking the data. 
    colors: a list of colors
    title: String
    plot_w and plot_h: Integers indicating the plot size
    showY: Boolean (is the y axis shown or not)
    rightY: Boolean (right y axis or not)
    Ylabel: String for the left y axis label
    Xlabel: String for the x axis label
    custom_Y_left: if a tuple, it will set a custom range for the left y axis
    custom_Y_right: if a tuple, it will set a custom range for the right y axis
    Ylabel_2: String for the right y axis label
    """
    
    #timeline view invoice amount
    labels = list(df.columns)
    rows = list(df.index)
    
    range_min = min(df.iloc[0])
    range_max = max(df.iloc[0]) + max(df.iloc[0])/10
    
    #main figure settings
    ts = figure(plot_width =plot_w, plot_height =plot_h, x_range = labels, 
                y_range=(range_min, range_max), title = title)
    ts.xaxis.major_label_orientation = np.pi/4
    ts.xaxis.major_label_text_font_size = x_axis_ticks_font_size
    
    #legend settings
    ts.legend.location = "top_left"
    ts.legend.click_policy="hide"
    
    #axis settings
    ts.yaxis.visible = showY
    ts.yaxis.axis_label_text_font_size = axis_label_font_size
    ts.yaxis.major_label_text_font_size = y_axis_ticks_font_size
    
    if custom_Y_left!=None: #if a custom left y axis range is set
        if showY:
            ts.yaxis.visible = False
            ts.extra_y_ranges = {"new_range": Range1d(custom_Y_left[0], custom_Y_left[1]+custom_Y_left[1]/10)}
            ts.add_layout(LinearAxis(y_range_name="new_range", axis_label = Ylabel,
                                     axis_label_text_font_size = axis_label_font_size,
                                     major_label_text_font_size = y_axis_ticks_font_size,
                                    ticker = SingleIntervalTicker(interval = custom_Y_interval_left),
                                    formatter = BasicTickFormatter(precision=custom_Y_precision_left , 
                                                               use_scientific=custom_Y_scientific_left)), 'left')
    
    if rightY: #if an additional y axis on the right is set
        if custom_Y_right!= None: #if a custom right y axis range is set
            range_right_min = custom_Y_right[0]
            range_right_max = custom_Y_right[1]
        else:
            range_right_min = range_min
            range_right_max = range_max
            
        ts.extra_y_ranges["new_range_2"] = Range1d(range_right_min, range_right_max+range_right_max/10)
        
        ts.add_layout(LinearAxis(y_range_name="new_range_2", axis_label = Ylabel_2, 
                                 axis_label_text_font_size = axis_label_font_size,
                                 major_label_text_font_size = y_axis_ticks_font_size,
                                 ticker = SingleIntervalTicker(interval = custom_Y_interval_right),
                                formatter = BasicTickFormatter(precision=custom_Y_precision_right , 
                                                               use_scientific=custom_Y_scientific_right)), 'right')
                                    
    #creating each timeseries line
    count=0
    for n in range(len(rows)):
        if len(legend_names)==0:
            legendname = rows[n]
        else:
            legendname = legend_names[n]
        ts.line(x = labels, y=df.loc[rows[n]], line_width=3.5, alpha=0.6, 
                legend=legendname, color=colors[count])
        count+=1
    
    if Ylabel_2=='': #if there's no additional label for right axis
        ts.yaxis.axis_label = Ylabel
        
    ts.xaxis.axis_label = Xlabel #x axis label
    #ts.ygrid.ticker = SingleIntervalTicker(interval = 1.1/11.3)
    if not showXGrid:
        ts.xgrid.grid_line_color = None
    
    if not showYGrid:
        ts.ygrid.grid_line_color = None
    
    
    ts.legend.label_text_font_size= legend_font_size
    
    if output_file_name!='':
        output_file(output_file_name+".html")
    
    return ts

def stacked(df):
    """
    Auxiliar stacking function for stackedChart
    """
    df_top = df.cumsum(axis=1)
    df_bottom = df_top.shift(axis=1).fillna({'count': 0})[::-1]
    df_stack = pd.concat([df_bottom, df_top], ignore_index=True)
    return df_stack

def stackedChart(df, colors, plot_w=1800, plot_h=450, showY=True):
    """
    It produces a timeseries as stacked area charts, using the df rows as fields to visualize
    """
    stack_df = df.transpose()
    areas = stacked(stack_df)
    x2 = np.hstack((stack_df.index[::-1], stack_df.index))
    
    p = figure(plot_width=plot_w, plot_height=plot_h, x_range=list(stack_df.index), 
               y_range=(0, sum([max(stack_df[i]) for i in list(stack_df.columns)])))
    
    for idx,value in enumerate(areas):
        p.patch(x2, areas[value].values,
              color=colors[idx], alpha=0.8, line_color=None, legend = list(stack_df.columns)[idx])
    
    p.legend.location = "top_left"
    p.legend.click_policy="hide"
    
    p.xaxis.axis_label = 'Year_month'
    
    p.yaxis.visible = showY
    p.xaxis.major_label_orientation = np.pi/4
    
    return p

def boxplot_hor(series, xrange, plot_h = 200, plot_w = 1000, margins = 50,
                box_width = 100, fill_color = TTQcolor['azureBlue'],
                center = 0, endline_stroke = 10, endline_length = 30, outliers = True, outliers_size = 0.5,
                logscale=False, btext = True, textsize = '8pt'):
    """
    This function will draw an horizontal boxplot chart, based on an input series
    """
    series = pd.Series(series)

    q1 = series.quantile(q=0.25) #1st quantile
    q2 = series.quantile(q=0.5)  #2nd quantile
    q3 = series.quantile(q=0.75) #3rd quantile
    iqr=q3 - q1
    upper = q3 + 1.5*iqr 
    lower = q1 - 1.5*iqr
    outl = series[(series>upper) | (series<lower)] #outliers
    

    outx = []
    outy = 0
    qmin = series.quantile(q=0.00)
    qmax = series.quantile(q=1.00)
    if len(outl)>0:
        outx = [v for v in outl]
        outy = [0]*len(outx)

    #main figure
    p = figure(title="", x_range = xrange,
           y_range = (-100,100), plot_height = plot_h, plot_width= plot_w)
    if logscale:
        p = figure(title="", x_range = xrange,
           y_range = (-100,100), plot_height = plot_h, plot_width= plot_w, x_axis_type = 'log')

    # if no outliers, shrink lengths of stems to be no longer than the minimums or maximum
    upper_s = min(qmax, upper)
    lower_s = max(qmin, lower)

    # stems
    p.segment(upper_s, center, q3, center, line_color="black")
    p.segment(lower_s, center, q1, center, line_color="black")

    # boxes
    p.vbar(q2+(q3-q2)/2,q3-q2, -box_width//2,  box_width//2, fill_color=fill_color, line_color="black", fill_alpha = 0.45) #x, top, bottom, width
    p.vbar(q1+(q2-q1)/2, q2-q1, -box_width//2, box_width//2, fill_color=fill_color, line_color="black", fill_alpha = 0.85)

    # whiskers (almost-0 height rects simpler than segments)
    p.rect(lower_s, center, endline_stroke, endline_length, line_color="black") #x, y, width, height
    p.rect(upper_s, center, endline_stroke, endline_length, line_color="black")

    #mean mark
    mean = round(series.mean(),2)
    p.rect(mean, -box_width, endline_stroke*2, endline_length*2, line_color=fill_color)
    p.circle(mean,-box_width*0.75, color = fill_color, size = outliers_size*5)

    if btext:
        #text
        text_1 = Label(x=q1, y=box_width*0.55, text='Lower quartile: '+str(round(q1,2)), text_font_size = textsize, text_align='center')
        p.add_layout(text_1)

        text_2 = Label(x=q2, y=box_width*0.67, text='Median: '+str(round(q2,2)), text_font_size = textsize, text_align='center')
        p.add_layout(text_2)

        text_3 = Label(x=q3, y=box_width*0.79, text='Upper quartile: '+str(round(q3,2)), text_font_size = textsize, text_align='center')
        p.add_layout(text_3)

        text_4 = Label(x=series.mean(), y=-box_width*0.68, text='Mean: '+str(mean), text_font_size = textsize, text_align='center')
        p.add_layout(text_4)

    #outliers
    if outliers:
        p.circle(outx, outy, size = outliers_size, fill_alpha=0.6)

    p.yaxis.visible = False
    p.xaxis.visible = False

    return p

def unit_poly_verts(theta, centre, radius=0.5):
    """Return vertices of polygon for subplot axes.
    This polygon is circumscribed by a unit circle centered at (0.5, 0.5).
    Changing the radius it is possible to change its size
    """
    x0, y0 = [centre ] * 2
    r = radius
    verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
    return verts

def radar_patch(r, theta, centre ):
    """ Returns the x and y coordinates corresponding to the magnitudes of 
    each variable displayed in the radar plot
    """
    # offset from centre of circle
    offset = 0.001
    yt = (r*centre + offset) * np.sin(theta) + centre 
    xt = (r*centre + offset) * np.cos(theta) + centre 
    return xt, yt

def spiderWebChart(legend_names, names, vals, colors, chart_name = '', 
                   subvids = 4, main_size = 0.5, normalize = False,
                  perc_scale = False, fill_alpha=0.3): #,plot_height=220, plot_width=220):
    """
    This function plots a single spider web chart that overlap different sets of values (if more than one is provided).
    names: the name of each field to compare in the graph
    vals: this is a list of lists. Each list into the main array represents a set of values for a particular dataset.
    colors: list of colors to use for each dataset plugged in
    char_name: the title of the chart
    subvids: the number of subdivision on the 'web'
    main_size: this is the radius of the web chart normally fixed to 0.5 to stay in the 1 to 1 circle
    normalize: if True, the values will be normalized between 0 and 1 considering the whole range of values provided.
                If normalize is set to False, the absolute values will be used.
    perc_scale:if True the normalization will be done between 0 and 100
    """
    #Vertex basis
    num_vars = len(names) #number of vertex of the polygon
    centre = 0.5 #centre of the polygon
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False) #theta of the circle
    theta += np.pi/2 # rotate theta such that the first axis is at the top
    theta_mod = list(theta)+[theta[-1]] #adding the first point to the end of the list to close the polygon
    verts = unit_poly_verts(theta, centre, main_size) #vertex coordinates
    #verts+=[verts[0]] #adding the first point to the end of the list to close the polygon 
    
    #ColumnDataSource
    x = [v[0] for v in verts] #x coordinates
    y = [v[1] for v in verts] #y coordinates
    text = names + [''] #labels
    source = ColumnDataSource({'x':x + [centre ],'y':y + [1],'text':text, 'angle':theta_mod})
    
    #tools
    #_tools_to_show = 'box_zoom,pan,save,hover,reset,tap,wheel_zoom'
    
    #figure
    p = figure(title=chart_name, x_range=(-0.35,1.35), y_range=(-0.35,1.35)) #, tools= _tools_to_show)
    
    #main polygon line
    p.line(x="x", y="y", source=source)

    #radial lines
    for w in verts:
        p.line(x=[centre,w[0]], y=[centre,w[1]])
    
    #secondary poligon lines
    for i in np.linspace(main_size/subvids,main_size/subvids*(subvids-1),subvids-1):
        verts_tmp = unit_poly_verts(theta, centre, i)
        x_tmp = [v[0] for v in verts_tmp] 
        x_tmp.append(x_tmp[0]) #addition to close the polygon
        y_tmp = [v[1] for v in verts_tmp] 
        y_tmp.append(y_tmp[0]) #addition to close the polygon
        p.line(x=x_tmp, y=y_tmp)

    #adding labels
    labels = LabelSet(x="x",y="y",text="text",
                      source=source, text_font_size="8pt", 
                      #x_offset = -40, 
                      #y_offset = -10, 
                      angle = 'angle'
                     )
    p.add_layout(labels)
    
    #values normalization
    vals_copy = [i for j in vals for i in j]
    vals = [np.array(o) for o in vals] #note the manually typed division by 10
    
    all_vals = np.array([i for j in vals for i in j])
    
    if normalize:
        if not perc_scale:
            flist = []
            for u in vals:
                xmax, xmin = all_vals.max(), min(0,all_vals.min())
                flist.append((u-xmin)/(xmax - xmin))
        else:
            flist = []
            for u in vals:
                xmax, xmin = 100, 0
                flist.append((u-xmin)/(xmax - xmin))
    else:
        flist = vals
    
    x_patch =[]
    y_patch =[]
    
    for f in range(len(flist)):
        xt, yt = radar_patch(flist[f], theta, centre)
        source_tmp = ColumnDataSource({'x':xt, 'y':yt, 'values':flist[f]})
        p.patch(x='x', y='y', fill_alpha=fill_alpha, fill_color=colors[f], source=source_tmp, legend = legend_names[f])

    #hover = p.select(dict(type=HoverTool))
    #hover.tooltips = [('%', '@values')]
    #p.add_tools(hover)
    
    p.legend.location = "top_left"
    p.legend.click_policy="hide"
    
    p.xaxis.visible = False
    p.yaxis.visible = False
    
    return p



#MODEL PERFORMANCE VIZ

def plot_rocs(metrics, label=None, 
              select_point = 0.80, 
              p_width = 1024, 
              p_height = 1024, 
              title_lab = '',
              file_output = False,
              exportpng = False,
              model_appendix = None,
              dark_background = False,
              deepFprOnly = True): 
    """
    This function will create a ROC curve for each metric plugged in. The argument 'metrics' needs to be a list.
    Each metric must be in the form {'fpr': array, 'tpr': array, 'auc': array} as per 'model_analysis_viz.py' modelling.
    """

    #basic settings
    
    colors = [TTQcolor['azureBlue'], TTQcolor['richOrange'], TTQcolor['bloodRed'], 
              TTQcolor['peach'], TTQcolor['richBrown'], TTQcolor['yell'], TTQcolor['darkPurple'],
             TTQcolor['font'], TTQcolor['marketplaceOrange'], TTQcolor['blueGrey']]

    p = figure(plot_width = p_width, 
               plot_height = p_height, 
               title = title_lab, 
               toolbar_location='above',
               x_axis_label = 'False Positive Rate',
               y_axis_label = 'True Positive Rate',
               tools = [BoxZoomTool(), ResetTool(), HoverTool()],
               output_backend="webgl"
               )
    
    dashed_line_color = 'grey'
    fpr80_pt_color = 'black'
    fpr80_text_color = 'black'
    legend_background_color = 'white'
    legend_text_color = 'black'

    #darksettings
    if dark_background:
        p.background_fill_color = TTQcolor["PPTbg"]
        dashed_line_color = TTQcolor['whiteGrey']
        fpr80_pt_color = 'white'
        fpr80_text_color = TTQcolor['whiteGrey']
        legend_background_color = TTQcolor["PPTbg"]
        legend_text_color = 'white'
    

    #diagonal    
    p.line([0,1],[0,1], line_dash = [4,4], line_color = dashed_line_color, legend = "AUC = 0.5")

    #colors
    col = [colors[c] for c in range(len(metrics))]

    #false positive rate 80
    fpr80=[]

    if label==None:
        label=[' ']*len(metrics)
    if model_appendix==None:
        model_appendix=[' ']*len(metrics)


    for idx, metric in enumerate(metrics):
        #roc curves
        p.line(metric["fpr"], metric["tpr"], line_color = col[idx], line_width = 3, legend = label[idx]+model_appendix[idx]+' '+str(round(metric['auc'],2)))
        fpr80.append(np.interp([select_point], metric["tpr"], metric["fpr"])[0])

    #labels
    select_point_list = []
    for j in fpr80:
        select_point_list.append(select_point)
    
    txtfpr = [str(round(num,4)) for num in fpr80]

    if deepFprOnly:
        fpr80 = [fpr80[-1]]
        select_point_list = [select_point_list[-1]]
        txtfpr = ['('+ str(round(select_point,2))+' , '+str(round(fpr80[0],3))+ ')']


    label_source = ColumnDataSource(data=dict(fpr80_lab = fpr80,
                                       select_point_lab = select_point_list,
                                       text_lab = txtfpr))

    labels = LabelSet(x = 'fpr80_lab' , 
                      y = 'select_point_lab', 
                      text = 'text_lab',
                      level = 'glyph',
                      source = label_source,
                      text_font_size = '12pt',
                      x_offset = -20,
                      y_offset = -28,
                      render_mode = 'css',
                      text_color = fpr80_text_color)

    p.add_layout(labels)

    #fpr80 dots
    p.circle(fpr80, select_point_list, size = 8, color = fpr80_pt_color)

    #legend
    p.legend.location = 'bottom_right'
    p.legend.click_policy = 'hide'
    p.legend.background_fill_color = legend_background_color
    p.legend.label_text_color = legend_text_color
    p.legend.label_text_font_size = '12pt'

    #output file
    if file_output:
         output_file(title_lab +'.html')
    
    if exportpng:
        export_png(p, filename = 'ROC.png')

    return p