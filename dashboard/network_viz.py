#===================================================================================
# Author       : Davide Mariani                                                    #
# Company      : Tradeteq                                                          #
# Script Name  : network_viz.py                                                    #
# Description  : Functions for network visualization                               #
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
from datetime import date, datetime
import math
import os

pd.options.mode.chained_assignment = None  # default='warn'

#network analysis
import networkx as nx

#bokeh 
from bokeh.io import show, output_notebook, output_file, curdoc
from bokeh.plotting import figure
from bokeh.layouts import gridplot, widgetbox, layout
from bokeh.models import (
    ColumnDataSource,
    GraphRenderer,
    StaticLayoutProvider,
    Circle,
    HoverTool,
    Range1d,
    Plot,
    MultiLine)
from bokeh.transform import factor_cmap, linear_cmap
from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.models.widgets import Select, MultiSelect

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



def create_main_connections(filetoload, log):
    """
    This function will create nodes and edges for the network visualization, saving them as separate pickle files at the current folder.
    It contains the main network feature engineering processes.
    """

    #import files
    df = pd.read_pickle(filetoload)

    #date we received the data
    ReportDate = pd.to_datetime('2018-09-28', yearfirst=True) 

    #columns names fixing
    col_names = []

    for j in df.columns:
        if j not in ['customer_name_1.1', 'debtor_id']:
            col_names.append(j)
        else:
            if j=='debtor_id':
                col_names.append('debtors_id')
            elif j=='customer_name_1.1':
                col_names.append('debtor_name_1')

    df.columns = col_names

    log.info('Preparing the main connections dataset...')
    #checking debtors which are also customers
    debtors_name = df['debtor_name_1'].unique()
    customers_name = df['customer_name_1'].unique()

    #number of hybrids
    hybrids = [i for i in debtors_name if i in set(customers_name)]

    #some debtors id are shared among more than one debtor
    df['debtors_id+name'] = df['debtors_id'].map(str)+' : '+df['debtor_name_1'].map(str)

    #each customer will be represented by its customer_id
    column_edge = 'debtors_id+name'
    column_id = 'customer_id'
    columns_to_keep = [column_id, 'uid', 'customer_name_1', 'debtors_id', 'debtor_name_1',  column_edge] #'is_new',
    data_to_merge = df[columns_to_keep].dropna(subset = [column_edge]).drop_duplicates()

    #to create connection between companies who have the same debtor, join the data with itself on the 'debtors_id+name' column
    data_to_merge = data_to_merge.merge(data_to_merge[columns_to_keep].rename(columns = {column_id:column_id+"_2", 
                                                                                         'customer_name_1':"customer_name_1_2",
                                                                                         'debtors_id':'debtors_id_2',
                                                                                        'debtor_name_1':"debtor_name_1_2",
                                                                                        'uid':'uid_2' #,'is_new':'is_new_2'
                                                                                        }),
                                                                                         on = column_edge)

    #removing duplicate connections
    d = data_to_merge.copy().drop_duplicates(subset = [column_id, column_edge, column_id+"_2"])
    d.drop(d.loc[d[column_id+"_2"] < d[column_id]].index.tolist(), inplace = True)


    # --------------------------------------------------------------------------------
    # Nodes dataset
    # --------------------------------------------------------------------------------
    log.info('Preparing the nodes dataset...')
    #adding a column to identify the customers which are also debtors
    flag = ['True', 'False']
    rules = [(d['customer_name_1'].isin(hybrids)), ~(d['customer_name_1'].isin(hybrids))]
    d['customer_id_is_also_debtor'] = np.select(rules, flag)
    rules_2 = [(d['debtor_name_1'].isin(hybrids)), ~(d['debtor_name_1'].isin(hybrids))]
    d['debtor_id_is_also_customer'] = np.select(rules_2, flag)


    #selection criteria
    cust_selection = d['customer_id_is_also_debtor']=='False'
    debt_selection = d['debtor_id_is_also_customer']=='False'
    hyb_cust_selection = d['customer_id_is_also_debtor']=='True'
    hyb_debt_selection = d['debtor_id_is_also_customer']=='True'

    #debtors_
    debt_list = d.loc[debt_selection,'debtor_name_1'].unique()

    #customers_
    cust_list = d.loc[cust_selection,'customer_name_1'].unique()

    #customer hybrids_
    hyb_cust_list = d.loc[hyb_cust_selection,'customer_name_1'].unique()

    #debtors hybrids_
    hyb_debt_list = d.loc[hyb_debt_selection,'debtor_name_1'].unique()
    
    #types
    type_customers = ['customer' for i in  d.loc[cust_selection, 'customer_name_1'].unique()]
    type_h_customers = ['customer' for j in  d.loc[hyb_cust_selection, 'customer_name_1'].unique()]
    type_debtors = ['debtor' for k in d.loc[debt_selection,'debtor_name_1'].unique()]
    type_h_debtors = ['debtor' for w in d.loc[hyb_debt_selection,'debtor_name_1'].unique()]
    all_types = type_customers + type_debtors + type_h_customers + type_h_debtors

    #types_2 (for viz)
    type_customers_2 = ['customer' for i in  d.loc[cust_selection,'customer_name_1'].unique()]
    type_h_customers_2 = ['customer and debtor' for j in  d.loc[hyb_cust_selection,'customer_name_1'].unique()]
    type_debtors_2 = ['debtor' for k in d.loc[debt_selection,'debtor_name_1'].unique()]
    type_h_debtors_2 = ['customer and debtor' for w in d.loc[hyb_debt_selection,'debtor_name_1'].unique()]
    all_types_2 = type_customers_2 + type_debtors_2 + type_h_customers_2 + type_h_debtors_2

    #names
    name_customers = [ct for ct in d.loc[cust_selection,'customer_name_1'].unique()]
    name_h_customers = [hct for hct in d.loc[hyb_cust_selection,'customer_name_1'].unique()]
    name_debtors = [db for db in d.loc[debt_selection,'debtor_name_1'].unique()]
    name_h_debtors = [hdb for hdb in d.loc[hyb_debt_selection,'debtor_name_1'].unique()]
    all_nodes = name_customers + name_debtors + name_h_customers + name_h_debtors


    #ids
    id_customers = [str(o).replace("[","").replace("]","") for o in [d.loc[d['customer_name_1']==y,'customer_id'].unique() for y in name_customers]]
    id_h_customers = [str(o).replace("[","").replace("]","") for o in [d.loc[d['customer_name_1']==u,'customer_id'].unique() for u in name_h_customers]]
    id_debtors = [str(o).replace("[","").replace("]","") for o in [d.loc[d['debtor_name_1']==y,'debtors_id'].unique() for y in name_debtors]]
    id_h_debtors = [str(o).replace("[","").replace("]","") for o in [d.loc[d['debtor_name_1']==f,'debtors_id'].unique() for f in name_h_debtors]]
    all_ids = id_customers + id_debtors + id_h_customers + id_h_debtors


    #creation of the nodes dataset
    nodes_df = pd.DataFrame({'Company_Name':all_nodes,
                             'ID': all_ids,
                             'Type': all_types,
                            'Type_2': all_types_2 #, 'is_new': all_isnew
                           })
    nodes_df = nodes_df.drop_duplicates(subset = 'Company_Name') 

    # --------------------------------------------------------------------------------
    # Edges dataset
    # --------------------------------------------------------------------------------
    log.info('Preparing the edges dataset...')

    #start and end point of each edge
    xs = []
    ys = []

    for idx in d.index:
        xs+=[d.loc[idx,'customer_name_1']]
        ys+=[d.loc[idx,'debtor_name_1']]

    #connecting customers and debtors
    edges_couples = [(d.loc[idx,'customer_name_1'], d.loc[idx,'debtor_name_1']) for idx in d.index]

    edges_df = pd.DataFrame(data = {'xs':xs, 'ys':ys}, index = xs) #edges dataframe by names

    #Adding columns to the edge dataset

    def count_column(df, columnname):
        """
        Given a goFactoring dataset, this function counts the number of True values for a certain column indicating
        the relationship between a customer and a debtor
        """
        return df.groupby(['customer_name_1', 'debtor_name_1']).apply(lambda x:x[columnname].sum())

    fields = ['is_pastdue90', 'is_pastdue180', 'is_open', 'has_impairment1', 'has_impairment2']

    features_df = pd.DataFrame()
    features_df['inst_count'] = df.groupby(['customer_name_1', 'debtor_name_1']).apply(lambda x:x['customer_id'].count())

    for f in fields:
        features_df[f+'_count'] = count_column(df,f)
        features_df[f] = [i>0 for i in features_df[f+'_count']]
        features_df[f+'_ratio'] = round(features_df[f+'_count']/features_df['inst_count'], 2)

    #this is slow
    for feature in features_df.columns:
        edges_df[feature] = [features_df.loc[(edges_df.iloc[i]['xs'], 
                                              edges_df.iloc[i]['ys'])].loc[feature] for i in range(len(edges_df))]

    G, edges_df, nodes_df = create_network(edges_df, nodes_df, log)

    log.info('Saving nodes dataset, edges dataset and graph...')
    nodes_df.to_pickle("network_nodes.pkl")
    edges_df.to_pickle('network_edges.pkl')
    nx.write_gpickle(G, 'base_graph.pkl')



def create_edge_nodes(filetoload, log, nodefilepath = 'network_nodes.pkl', edgefilepath = 'network_edges.pkl', graphfilepath = 'base_graph.pkl'):
    if not os.path.exists(nodefilepath) or not os.path.exists(edgefilepath) or not os.path.exists(graphfilepath):
        log.info("Nodes or edges dfs or graph are missing. Creating new dataframes for them (this will take a bit)...")
        create_main_connections(filetoload, log)
    else:
        log.info("Loading nodes and edges...")

def create_network(edges, nodes, log):
    """
    This function will create the network structure using networkx. It will also modify edges and nodes datasets for next visualization steps.
    This is an auxiliar function of netowrk_info
    """
    log.info('Creating the graph with attributes...')
    edges = edges.drop_duplicates(subset = ['xs', 'ys'])
    edges_tuples = [(edges.iloc[i]['xs'], edges.iloc[i]['ys']) for i in range(len(edges))]
    edges['edges_couple'] = edges_tuples #this will be useful for successive sorting after the graph is created on bokeh

    # build the nx graph
    log.info('Creating nx graph...')
    G=nx.Graph()
    G.add_edges_from(edges_tuples)
    nodes_list = list(G.nodes)

    idxs = []
    for i in nodes_list:
        idxs.append(nodes[nodes['Company_Name']==i].index[0])

    #sorting with same graph order
    nodes = nodes.iloc[idxs]

    #nodes analysis to define their centrality
    log.info('Calculating centralities...')
    centrality = nx.degree_centrality(G) #centrality dictionary
    nodes['centrality'] = [centrality[n] for n in list(nodes['Company_Name'])]
    log.info("Nodes df updated with the new column 'centrality'...")

    #coordinates
    log.info('Adding coordinates for circular layout...')
    pos = init_layout(G, nodes)
    coordinates = [np.array(pos[j]) for j in nodes['Company_Name']]
    nodes['coords'] = coordinates
    log.info("Nodes df updated with the new column 'coords'...")

    return G, edges, nodes




def network_info(G, edges, nodes, log, nodes_size_range = (6,15), 
                 to_highlight = 'is_pastdue90', circularLayout = False):
    """
    This function will add information to edges and nodes dataset, as well as the graph, to prepare them for visualization.
    The output will be a networkx graph, an updated edges dataset and an updated nodes dataset.
    """

    edges_tuples = [t for t in G.edges] #update with the actual number of edges after network building

    #nodes size
    max_size = nodes_size_range[1]
    min_size = nodes_size_range[0]
    nodes['size'] = np.around(np.interp(nodes['centrality'], (nodes['centrality'].min(), nodes['centrality'].max()), (min_size, max_size)),2)

    #nodes nx attributes
    node_size = dict([tuple(x) for x in nodes[['Company_Name', 'size']].values])
    node_type = dict([tuple(y) for y in nodes[['Company_Name', 'Type_2']].values])

    edges_highlight = dict(zip(edges_tuples, [edges.loc[(edges['xs'].isin(p)) & (edges['ys'].isin(p)), to_highlight].unique()[0] for p in edges_tuples]))

    nx.set_node_attributes(G, name='size', values=node_size) 
    nx.set_node_attributes(G, name='type', values=node_type)
    nx.set_edge_attributes(G, name='highlight', values=edges_highlight)

    return G, edges, nodes

def visualize_graph(graph, edges, nodes, log, title = 'Network Graph', plot_w = 900, plot_h = 900, file_output = '', nx_k=0.028, nx_iterations=25,
                      to_highlight = 'is_pastdue90', nodes_colors = [TTQcolor['sky'], TTQcolor['Salmon'], TTQcolor['marketplaceOrange']],
                      edges_colors = [TTQcolor['whiteGrey'], TTQcolor['warningRed']], circularLayout=False):

    """
    This function will give visual attributes to the graph.
    """
    log.info("Creating network visual attributes...")

    if circularLayout:
        graph=GraphRenderer()
        graph_layout = dict(zip(list(nodes['Company_Name']), list(nodes['coords'])))
        graph.layout_provider = StaticLayoutProvider(graph_layout = graph_layout)

        edges = edges.drop_duplicates(subset=['xs','ys'])
        graph.edge_renderer.data_source.data = dict(start = list(edges['xs']),
                                               end = list(edges['ys']))
    else:
        graph = from_networkx(graph, nx.spring_layout, k=nx_k, iterations=nx_iterations)

    #unfortunately the use of list comprehension at next step is necessary
    #since bokeh doesn't seem to support other collections like Series or arrays
    graph.node_renderer.data_source.data['index'] = [i for i in nodes['Company_Name']] #setting the company names
    graph.node_renderer.data_source.data['size'] = [s for s in nodes['size']] #setting node sizes
    graph.node_renderer.data_source.data['type'] = [t for t in nodes['Type_2']] #setting types

    graph.node_renderer.glyph = Circle(size='size', fill_color=factor_cmap('type', nodes_colors,    #creating nodes
                                                                    ['debtor', 'customer and debtor', 'customer']),
                                                                     fill_alpha=0.8, line_color='white', line_width=0.5)

    graph.node_renderer.nonselection_glyph = Circle(size='size', fill_color=factor_cmap('type', nodes_colors, #creating non-selected nodes
                                                                nodes['Type_2'].unique()),
                                               fill_alpha=0.1, line_alpha=0.05)

    
    graph.edge_renderer.nonselection_glyph = MultiLine(line_color=linear_cmap('highlight', edges_colors, False,True), #creating non-selected edges
                                                                 line_alpha=0.05, line_width=0.05)

    graph.node_renderer.hover_glyph = Circle(size='size', fill_alpha=0.0, line_width=3, line_color='green') #creating hover settings for circles
    graph.edge_renderer.hover_glyph = MultiLine(line_color='#abdda4', line_width=0.8) #creating hover settings for edges

    graph.selection_policy = NodesAndLinkedEdges()
    graph.inspection_policy = NodesAndLinkedEdges()

    return graph



#CIRCULAR LAYOUT FUNCTIONS

#position of the k-th point out of n total points place on layers circles around (xc,yc)
#all circles have radius within maxlayeroffset from radius r
#instead of specifying layers one may specify number of points per layer pperlayer




def innercircle_vec(labels, r=.4, nlayers = 3, step=7, maxlayeroffset=.42, xc=0, yc=0):
    """
    This function create coordinate positions based on the primetable
    """
    
    primetable = np.array([1,2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,59,61,67,71,73])

    #position points labelled with labels on nlayers of circles around xc,yc
    n = len(labels)
    layeroffset = 0. if nlayers==1 else 2* maxlayeroffset / (nlayers-1)

    #distribute the points on the layers proportional to layer radius
    layernp=[int(np.floor(((n+nlayers)/(nlayers*r)) * (r + (i-(nlayers-1)/2)*layeroffset)))
                  for i in range(nlayers)]
    #print(layernp)
    phi0 = np.random.random(len(layernp))*1.
    pos={}
    for k in range(n):
        insidelayers = np.argwhere(np.cumsum(layernp) > k)
        layeridx = insidelayers[0][0] #idx of layer k-th point is at
        npt = layernp[layeridx]
        step = min(primetable[[npt%p>0 for p in primetable]], key=lambda x:abs(x-(npt/3.5)))
        rl = r + (layeridx - (nlayers-1)/2)*layeroffset
        phi = 2*np.pi*k*step/npt + phi0[layeridx]
        x = xc + rl * np.cos(phi)
        y = yc + rl * np.sin(phi)
        pos[labels[k]] = np.array([x,y])
    return pos
    
    
def init_layout(G, nodes, R=0.54, nperlayer=26, nodescircles=0.065):
    """
    This function creates a graph layout for visualization using innercicle_vec
    """
    #sellers
    sellernames = pd.DataFrame(nodes.loc[nodes.Type == "customer",:]).\
            sort_values(by="centrality", ascending=False).\
            reset_index(drop=True)
    pos = innercircle_vec(list(sellernames.Company_Name), r=R, nlayers=sellernames.shape[0]//nperlayer+1)

    #buyers - around the first seller linked to them
    for seller in list(sellernames.Company_Name):
        blist = list(G.neighbors(seller))
        bnotalloc = [b for b in blist if not b in pos.keys()]
        
        nb = len(bnotalloc)
        if nb>0:
            pos = dict(pos, 
                   **innercircle_vec(bnotalloc, r=nodescircles+0.004*np.sqrt(nb//nperlayer), 
                    nlayers=nb//nperlayer+1, maxlayeroffset=0.004*np.sqrt(nb//nperlayer), 
                    xc=pos[seller][0], yc=pos[seller][1]))
    return pos


