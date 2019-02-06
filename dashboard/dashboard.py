#===================================================================================
# Author       : Davide Mariani                                                    #
# Company      : Tradeteq                                                          #
# Script Name  : dashboard.py                                                      #
# Description  : Dashboard for network visualization and relations details         #
# Version      : 0.2                                                               #
# Required bokeh version : 1.0.4                                                   #
#==================================================================================#
#==================================================================================# 
#                                                                                  #
#==================================================================================#

#base modules
import pandas as pd
import numpy as np
import datetime
from datetime import date,datetime
import math
from matplotlib.dates import date2num, num2date
from os import environ

#network analysis
import networkx as nx

#bokeh 
from bokeh.io import show, output_notebook, output_file, curdoc
from bokeh.plotting import figure
from bokeh.layouts import gridplot, widgetbox, layout
from bokeh.models import (
    ColumnDataSource, 
    CustomJS,
    Circle,
    HoverTool,
    TapTool,
    Range1d,
    Plot,
    MultiLine,
    Button,
    LinearColorMapper)
from bokeh.transform import factor_cmap, linear_cmap
from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.models.widgets import Select, MultiSelect, Toggle
from bokeh.models.glyphs import MultiLine
from bokeh.events import ButtonClick

from bokeh.palettes import Spectral

#config
from locale import setlocale, getlocale, LC_ALL, atof
import locale
import sys
from myutils import *


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

#repayment patterns
from rpatterns_viz import * 

#network visualization
from network_viz import *


# --------------------------------------------------------------------------------
# Log initialisation
# --------------------------------------------------------------------------------

_modulename = 'dashboard'

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


username = environ["USERNAME"]
datafolder = cfg["datafolder"].format(username) # user specific path
loadprefix = cfg["loadprefix"]
repdate = cfg["reportdate"]
filename = cfg["filename"]
circlayout = cfg["circlayout"]=='True'



def run_dashboard(tr, graph_, edges_, nodes, reportdate, log=log, datesnum = 10, 
                  highlight_parameter_vals  = ['is_pastdue90', 'is_pastdue180', 'has_impairment1', 'has_impairment2', 'is_open'],
                  background_color = TTQcolor["PPTbg"],
                  nodes_colors =[TTQcolor['sky'], TTQcolor['Salmon'], TTQcolor['marketplaceOrange']],
                  edges_colors = [TTQcolor['whiteGrey'], TTQcolor['warningRed']],
                  patterns_latecol = TTQcolor["background"], #TTQcolor["richOrange"], 
                  patterns_maincol = TTQcolor["background"], #used for boundary only atm
                  patterns_earlycol = TTQcolor["algae"], 
                  warningcol = TTQcolor['warningRed'],
                  replinecol = TTQcolor["yellowOrange"],
                  alphanode_choices = [1.0, 0.25],
                  widthnode_choices = [0.7, 0.05],
                  widthedge_choices = [0.9, 0.1],
                  alphaedge_choices = [0.8, 0.25],
                  edges_selected_col = TTQcolor['whiteGrey'],
                  _tools_to_show = 'box_zoom,pan,save,hover,tap,wheel_zoom',
                  NET_TOOLTIPS = [('Company Name', '@index'), ('Company Type', '@type')],
                  RP_TOOLTIPS = [('customer_name', '@customers_name'),
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
                     ('has_impairment2', '@has_impairment2'),
                     ('has_prosecution', '@has_prosecution'),
                     ('impairment1_score', '@score_0'),
                     #('pastdue90_score', '@score_1'),
                     #('pastdue180_score', '@score_2')
                     ],
                  circularLayout = circlayout):
    """
    This function executes the dashboard.
    It needs to run on a bokeh server to be visualized - standard procedure is to run 'bokeh serve --show nameofthefilecontainingthisfunction.py'
    tr :transactions dataframe to visualize
    edges_:edges dataframe
    nodes: nodes dataframe
    reportdate: date used as a reference in order to estimate a current date/date limit
    log: log config file
    datesnum: number of date to display on the x axis of the repayment pattern plot
    highlight_parameter_vals: parameters to include in the dropdown highlight selection
    nodes_colors: colors for the different types of nodes
    edges_colors: colors for the edges
    patterns_latecol: color for late repayment bar
    patterns_earlycol: color for early repayment bar
    patterns_maincol: color for standard repayment bar
    warningcol: color for bars with highlighted features (usually warnings)
    replinecol: color of the report date line
    alphanode_choices: alpha values for nodes if selected or not
    widthnode_choices: width values for nodes if selected or not
    widthedge_choices: width values for edges if selected or not
    alphaedge_choices: alpha values for edges if selected or not
    edges_selected_col: color for edges connected to the selected node
    _tools_to_show: tools to include in the dashboard like zoom, pan etc
    NET_TOOLTIPS contains the instructions for the parameter to be visualized on the network
    RP_TOOLTIPS contains the instructions for the parameter to be visualized on the repayment patterns graph
    """

    # --------------------------------------------------------------------------------
    # Network
    # --------------------------------------------------------------------------------
    graph_, edges_, nodes = network_info(graph_, edges_, nodes, log, circularLayout=circularLayout)

    #visualization with bokeh
    plot = figure(title = '', plot_width = 800, plot_height=800, tools="box_select,box_zoom,tap,wheel_zoom,reset,pan,save", x_range=(-1.1,1.1), y_range=(-1.1,1.1),
                  background_fill_color = background_color)

    #highlight field selection widget for network
    highlight_name_options = [('None', 'None')]+[(i,i) for i in highlight_parameter_vals]
    highlight_select = Select(title='Highlight edges with: ', value='None', options = highlight_name_options) 

    #visualize the base graph
    graph = visualize_graph(graph_, edges_, nodes, log, nodes_colors=nodes_colors, edges_colors=edges_colors, circularLayout=circularLayout)
    
    if circularLayout:
        edges = edges_
    else:
        log.info("Re-sorting graph edges...")
        #resorting edges dataset in order to match edge info and their visualization on the graph
        edge_start = graph.edge_renderer.data_source.data['start']
        edge_end = graph.edge_renderer.data_source.data['end']
        sorter = []
        for j in edges_['edges_couple']:
            for k in range(len(edge_start)):
                if set(j) == set((edge_start[k], edge_end[k])):
                    sorter.append(k)
                    break
        edges_['sorter'] = sorter

        edges = edges_.sort_values(by=['sorter'])
        edges = edges.drop_duplicates(subset = 'sorter')

    #plot settings
     #hover tooltips

    plot.add_tools(HoverTool(tooltips=NET_TOOLTIPS))
    plot.xgrid.grid_line_color = None
    plot.ygrid.grid_line_color = None
    plot.axis.visible = False

    # --------------------------------------------------------------------------------
    # Patterns
    # --------------------------------------------------------------------------------
    y, right1, right2, right3, left1, left2, left3 = set_the_patterns(tr, reportdate)

    #columndatasource (this needs to be hardcoded from case to case)
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
                                    'has_impairment2':tr['has_impairment2'],
                                     'has_prosecution':tr['has_prosecution'],
                                     'score_0':tr['score_f0'],
                                     #'score_1':tr['score_f1'],
                                     #'score_2':tr['score_f2']
                                    })

    #axis ranges
    x_range = (np.min(source.data['left1']), max(reportdate+15, np.nanmax(source.data['right2'])))
    y_range = (-.5, max(source.data['y'])+.5)
    
    #main fig
    fig = figure(x_range=x_range, y_range=y_range, plot_width=800, plot_height=800,
                tools = _tools_to_show, title='', background_fill_color = TTQcolor["PPTbg"])
      
    #hovertool and tooltips
    hover = fig.select(dict(type=HoverTool))
    hover.tooltips = RP_TOOLTIPS
    
    #customise figure attributes
    fig.yaxis.visible = False
    
    linedict = {}
    fig.ygrid.grid_line_color=None
    fig.xgrid.grid_line_color=None
    multiline_count = 0

    #customer selection widget
    customer_name_vals = sorted([n for n in tr['customer_name_1'].unique()], key = lambda x:x.lower()) 
    customer_name_options = [(i,i) for i in customer_name_vals]+ [('All', 'All')]
    customer_select = Select(title='Customer', value=customer_name_vals[0], options = customer_name_options) 

    #debtors selection widget
    debtor_name_vals = sorted([nd for nd in tr['debtor_name_1'].unique()], key = lambda x:x.lower())
    debtor_name_options = [(k,k) for k in debtor_name_vals]
    debtor_select = MultiSelect(title='Debtors', value=debtor_name_vals, options = debtor_name_options) 

    # --------------------------------------------------------------------------------
    # Buttons
    # --------------------------------------------------------------------------------
    rendering_button = Toggle(label="Repayment Patterns ON", button_type='success', active=True)

    # --------------------------------------------------------------------------------
    # Interactive selection functions
    # --------------------------------------------------------------------------------

    def select_customer_df():
        """
        Auxiliar function for update_2() that slices the dataset according with customer or debtor selection
        It affects both repayment patterns visualization and network visualization
        """
        cust_val = customer_select.value
        selected = tr[(tr['customer_name_1']==cust_val)] # & (tr['debtor_name_1'].isin(debt_val))]
        return selected

    def select_debtor_df():
        """
        Auxiliar function for update_2() that slices the dataset according with customer or debtor selection
        It affects both repayment patterns visualization and network visualization
        """
        cust_val = customer_select.value
        debt_val = debtor_select.value
        selected = tr[(tr['customer_name_1']==cust_val) & (tr['debtor_name_1'].isin(debt_val))]
        return selected 

    def selected_node_and_edges():
        """
        Auxiliar function for update_2() and update_3() that sets the attributes to highlight certain parts of the network
        It affects network visualization
        """
        cust_val = customer_select.value
        debt_val = debtor_select.value

        if cust_val != 'All':
            node_conditions = [(((nodes['Company_Name']==cust_val) & (nodes['Type_2'].isin(['customer', 'customer and debtor']))) |
                         ((nodes['Company_Name'].isin(debt_val)) & (nodes['Type_2'].isin(['debtor', 'customer and debtor'])))),
                          ~(((nodes['Company_Name']==cust_val) & (nodes['Type_2'].isin(['customer', 'customer and debtor']))) |
                         ((nodes['Company_Name'].isin(debt_val)) & (nodes['Type_2'].isin(['debtor', 'customer and debtor']))))]

            edge_conditions = [((edges['xs']==cust_val) & (edges['ys'].isin(debt_val))) | ((edges['ys']==cust_val) & (edges['xs'].isin(debt_val))),
                               ~(((edges['xs']==cust_val) & (edges['ys'].isin(debt_val))) | ((edges['ys']==cust_val) & (edges['xs'].isin(debt_val))))]

            graph.edge_renderer.data_source.data['tmp_alpha'] = np.select(edge_conditions, alphaedge_choices) #setting temp alpha
            graph.edge_renderer.data_source.data['tmp_width'] = np.select(edge_conditions, widthedge_choices) #setting temp width
            graph.node_renderer.data_source.data['tmp_alpha'] = np.select(node_conditions, alphanode_choices) #setting temp alpha
            graph.node_renderer.data_source.data['tmp_widthnode'] = np.select(node_conditions, widthnode_choices) #setting temp alpha

            #visualize in case 'All' is NOT selected
            graph.node_renderer.glyph = Circle(size='size', fill_color=factor_cmap('type', nodes_colors,    #creating nodes
                                                                            ['debtor', 'customer and debtor', 'customer']),
                                                                             fill_alpha='tmp_alpha', line_color='white', line_width='tmp_widthnode',
                                                                             line_alpha = 'tmp_alpha')

            graph.edge_renderer.glyph = MultiLine(line_color=linear_cmap('highlight', edges_colors, False,True), #creating non-selected edges
                                                                     line_alpha='tmp_alpha', line_width='tmp_width')

        else:
            #this is the part of code causing error messages when selecting 'All' in the 'Customer' widget
            #ColumnDataSource after this part will have different length for its columns of data
            #it doesn't affect the general use of the dashboard apart from the click and select feature, which seems to go crazy after 'All' has been selected
            #--to solve in the future--
            graph.edge_renderer.data_source.data['tmp_alpha'] = [alphaedge_choices[0]]*len(tr) #setting temp alpha
            graph.edge_renderer.data_source.data['tmp_width'] = [widthedge_choices[0]]*len(tr) #setting temp width
            graph.node_renderer.data_source.data['tmp_alpha'] = [alphanode_choices[0]]*len(tr) #setting temp alpha
            graph.node_renderer.data_source.data['tmp_widthnode'] = [widthnode_choices[0]]*len(tr) #setting temp alpha

    

        

        graph.selection_policy = NodesAndLinkedEdges()
        graph.inspection_policy = NodesAndLinkedEdges()


    # --------------------------------------------------------------------------------
    # Dashboard Update - Auxiliar functions
    # --------------------------------------------------------------------------------

    def show_patterns(df_cust, to_highlight):
        """
        This function includes all the graphic updates at each selection for the repayment patterns.
        This function will be called with all the update() 1,2,3
        It may appear to be a repetition of the above code for patterns, but it is necessary for all the updates in Bokeh.
        """

        #re-set the patterns with the new selected dataframe rows
        new_y, new_right1, new_right2, new_right3, new_left1, new_left2, new_left3 = set_the_patterns(df_cust, reportdate)


        
        #colors for highlight
        colors_ = [warningcol, patterns_latecol]
        colors_2 = [warningcol, patterns_maincol] #patterns_maincol]
        colors_3 = [patterns_earlycol, patterns_earlycol]

        widths = [0.0, 0.25]

        rules = [[False]*len(df_cust), [True]*len(df_cust)]

        if to_highlight!='None':
            rules = [df_cust[to_highlight], ~df_cust[to_highlight]]
     

        #highlighting colors
        imp_colors = np.select(rules, colors_) 
        imp_colors_2 = np.select(rules, colors_2) 
        imp_colors_3 = np.select(rules, colors_3) 

        widthwarning = np.select(rules, widths) 

        #columndatasource (this needs to be hardcoded from case to case) - unfortunately bokeh needs this to be repeated each time since data streams change
        source.data = {'y':new_y, 'right1':new_right1, 'left1':new_left1, 'right2':new_right2, 'left2':new_left2, 
                              'right3':new_right3, 'left3':new_left3,
                              'customers_name': df_cust['customer_name_1'],
                                     'customer_id':df_cust['customer_id'],
                                     'imp_colors':imp_colors,
                                     'imp_colors_2':imp_colors_2,
                                     'imp_colors_3':imp_colors_3,
                                     'widthwarning': widthwarning,
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
                                    'has_impairment2':df_cust['has_impairment2'],
                                     'has_prosecution':df_cust['has_prosecution'],
                                     'score_0':df_cust['score_f0'],
                                     #'score_1':df_cust['score_f1'],
                                     #'score_2':df_cust['score_f2']
                                    }
        #axis ranges
        x_range_new = (np.min(source.data['left1']), max(reportdate+15, np.nanmax(source.data['right2'])))
        y_range_new = (-.5, max(source.data['y'])+.5)

        #main fig ranges
        fig.x_range.start = x_range_new[0] 
        fig.x_range.end = x_range_new[1]
        fig.y_range.start = y_range_new[0]
        fig.y_range.end = y_range_new[1]


        

        #impairment1score highlighting embedded in viz
        color_mapper = LinearColorMapper(palette=Spectral[7], low=0.05, high=1.1 ) #not parametrizable with highlights apparently?
        #mainbar
        fig.hbar(y = 'y', left = 'left1', right='right1', height = .8, line_color = 'imp_colors_2', line_width='widthwarning', #width wiwth rules
                    fill_color={'field': 'score_0', 'transform': color_mapper}, alpha=.8, line_alpha=0.65, source=source)
        #late repayment bar
        fig.hbar(y = 'y', left = 'left2', right='right2', height = .8, 
                    fill_color={'field': 'score_0', 'transform': color_mapper}, alpha=.08, line_alpha=0.65, line_width='widthwarning',
                   line_color='imp_colors', source=source) #reorganize colors in the attributes
        #early repayment bar
        fig.hbar(y = 'y', left = 'left3', right='right3', height = .8, 
                    fill_color='imp_colors_3', alpha=.8, line_alpha=0., source=source)
 

        #report date line
        fig.line(x=[reportdate, reportdate], y=[min(y)-1, max(y)+1], 
                    line_color=replinecol, line_dash='dashed')

        #hover tool
        hover = fig.select(dict(type=HoverTool))
        hover.tooltips = RP_TOOLTIPS


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

        #gridlines update
        if len(linedict.keys())>0:
            for k in linedict.keys():
                linedict[k].visible = False
        
        coords = {}
        coords['xs'] = [[d,d] for d in daystmp[::no_of_dates] if (datetime.fromordinal(int(d))+days_margin < datetime.fromordinal(int(reportdate)) or 
                                                                  datetime.fromordinal(int(d))-days_margin> datetime.fromordinal(int(reportdate)))]
        coords['ys'] = [[min(y)-1, max(y)+1] for d in daystmp[::no_of_dates] if (datetime.fromordinal(int(d))+days_margin < datetime.fromordinal(int(reportdate)) or 
                                                                                 datetime.fromordinal(int(d))-days_margin> datetime.fromordinal(int(reportdate)))]
        tmp_source = ColumnDataSource(data=coords)

        linedict['gridlines']= fig.multi_line(xs = 'xs', ys = 'ys', line_color=TTQcolor["whiteGrey"], line_alpha=0.5, source = tmp_source)

        daynums =  sorted([int(o) for o in daystmp[::no_of_dates] if datetime.fromordinal(int(o))+days_margin < datetime.fromordinal(reportdate)]+[int(reportdate)]) #conversion to int to make major label overrides work
        fig.xaxis.ticker = daynums
        fig.xaxis.major_label_overrides = dict(zip(daynums, [datetime.fromordinal(int(s)).strftime("%Y-%m-%d") for s in daynums]))
        fig.xaxis.major_label_orientation = -math.pi/4
        fig.ygrid.grid_line_color=None
        fig.xgrid.grid_line_color=None

    
    def button_event(event):
        """
        This function re-trigger update() and update_3() functions when the toggle status changes
        """
        if rendering_button.label == "Repayment Patterns ON":
            rendering_button.label = "Repayment Patterns OFF"
            rendering_button.button_type = "default"
            #show_patterns(tr[tr['customer_name_1']=='***///###'], to_highlight='None')

        else:
            rendering_button.label = "Repayment Patterns ON"
            rendering_button.button_type = "success"
            
        #updates to trigger
        update()
        update_3()


    def show_active():
        """
        This function will stop the automatic rendering of repayment patterns, asking for the user to click on the button 'Render'
        to visualize them. This avoid undesired queuing of visualizations when selecting many options quickly.
        """
        return rendering_button.active
  

    def update():
        """
        This function will update the graph edges color depending on the parameter to highlight.
        It affects both network visualization and repayment patterns
        """
        edges_colors = [edges_selected_col, warningcol]    

        ###network
        to_highlight = highlight_select.value

        alphavalues = [alphaedge_choices[1]]*len(edges)
        widthvalues = [widthedge_choices[1]]*len(edges)

        if to_highlight in set(list(edges.columns)):
            graph.edge_renderer.data_source.data['highlight']=[o for o in edges[to_highlight]] #setting the parameter to highlight for edges
            alphavalues = [alphaedge_choices[1] if j else alphaedge_choices[0] for j in edges[to_highlight]]
            widthvalues = [[widthedge_choices[1]] if j else [widthedge_choices[0]] for j in edges[to_highlight]]
        else:
            graph.edge_renderer.data_source.data['highlight']=[False for o in range(len(edges))]

        graph.edge_renderer.data_source.data['alpha'] = alphavalues
        graph.edge_renderer.data_source.data['width'] = widthvalues

        graph.edge_renderer.glyph = MultiLine(line_color=linear_cmap('highlight', edges_colors, False,True), #setting edge colors
                                                                line_alpha='alpha', line_width='width')
        
        ###patterns
        df_cust = select_debtor_df()
        if show_active():
            show_patterns(df_cust, to_highlight)


    def update_2():
        """
        This function will update the customer selection for the repayment patterns graph and the network
        It affects repayment patterns visualization
        """
        ###network
        selected_node_and_edges()

        ###patterns
        df_cust = select_customer_df()
        new_debtor_name_vals = sorted([nd for nd in tr.loc[tr['customer_name_1']==customer_select.value, 'debtor_name_1'].unique()], key = lambda x:x.lower())
        new_debtor_name_options = [(k,k) for k in new_debtor_name_vals]

        debtor_select.options = new_debtor_name_options
        debtor_select.value = new_debtor_name_vals

        to_highlight = highlight_select.value
        if show_active():
            show_patterns(df_cust, to_highlight)


    def update_3():
        """
        This function will update the debtor selection for the repayment patterns graph and the network.
        """
        ###network
        selected_node_and_edges()

        ###patterns
        df_cust = select_debtor_df()
        new_debtor_name_vals = sorted([nd for nd in tr.loc[tr['customer_name_1']==customer_select.value, 'debtor_name_1'].unique()], key = lambda x:x.lower())
        new_debtor_name_options = [(k,k) for k in new_debtor_name_vals]

        debtor_select.options = new_debtor_name_options

        to_highlight = highlight_select.value
        if show_active():
            show_patterns(df_cust, to_highlight)

        

    # --------------------------------------------------------------------------------
    # Callbacks and updates calls, with curdoc from bokeh
    # --------------------------------------------------------------------------------

    #toggle button triggers
    rendering_button.on_click(button_event)

    #javascript callback
    callback = CustomJS(args = dict(source = graph.node_renderer.data_source, source_2 = [i for i in graph_.nodes],
                                    customer_select = customer_select), code =
                        """
                        console.log(cb_obj)
                        inds = cb_data.source.selected['1d'].indices;
                        taptool_val = source_2[inds];
                        customer_select.value=taptool_val
                        customer_select.change.emit();
                        """ )
    tapsel = TapTool(callback = callback)
    plot.add_tools(tapsel)

    #highlight selection widget
    controls = [highlight_select]
    for control in controls:
        control.on_change('value', lambda attrname, old, new: update())

    widgets = widgetbox(*controls)

    #customers and debtors selection widget
    controls_2 = [customer_select]
    for control2 in controls_2:
        control2.on_change('value', lambda attrname, old, new: update_2()) 

    controls_3 = [debtor_select]
    for control3 in controls_3:
        control3.on_change('value', lambda attrname, old, new: update_3())

    widgets_2 = widgetbox(controls_2+controls_3)

    widgets_3 = widgetbox(rendering_button)

    #addition of the network graph to the plot
    plot.renderers.append(graph)

    #layout
    l = gridplot([[widgets, widgets_3, widgets_2],
               [plot, fig]])

    update()
    update_2()
    update_3()

    #app bokeh curdoc
    curdoc().add_root(l)
    curdoc().title = 'Dashboard'

# --------------------------------------------------------------------------------
# Import data
# --------------------------------------------------------------------------------

log.info("Reading the input dataset...")
create_edge_nodes(datafolder + loadprefix+ filename, log=log)
df = pd.read_pickle(datafolder + loadprefix+ filename)

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
nodes_pkl = pd.read_pickle('network_nodes.pkl')
edges_pkl = pd.read_pickle('network_edges.pkl')
graph_pkl = nx.read_gpickle('base_graph.pkl')
# --------------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------------
run_dashboard(df, graph_pkl, edges_pkl, nodes_pkl, ReportDateOrd)
