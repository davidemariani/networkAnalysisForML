#==================================================================================#
# Author       : Davide Mariani                                                    #  
# University   : Birkbeck College, University of London                            # 
# Programme    : Msc Data Science                                                  #
# Script Name  : network_bond_graph.py                                             #
# Description  : functions for modelling the network as a bond graph               #
# Version      : 0.1                                                               #
#==================================================================================#
# This file contains several functions to model the network as a bond graph using  #
# networkx                                                                         #
#==================================================================================#

#base modules
import pandas as pd
import numpy as np
import datetime
from datetime import date,datetime
import math
import os

pd.options.mode.chained_assignment = None  # default='warn'

#network analysis
import networkx as nx


def build_bgtfeat(dfi, g, edge_flow='imp_edge_flow', edge_eff='imp_edge_eff', node_flow='imp_node_flow', energy='imp_energy',
                      c_node_eff='imp_c_node_eff', d_node_flow='imp_d_node_flow'):
    """
    This function takes a pandas datafram of instruments dfi and a graph g in order to calculate effort, flow and energy at each edge and node,
    and to return the same dataframe with the new calculations added as features.
    edge_flow: flow at each edge between buyer and seller (it goes from buyer to seller)
    edge_eff: effort at each edge between buyer and seller (referred to sellers)
    node_flow: flow at each node calculated as the sum of the flow of the edges connecting that node
    energy: sum of the products between edge effort and buyer nodes flow
    c_node_eff: effort at the seller node
    d_node_flow: flow at the buyer node
    """    
    df=dfi.copy()

    #debtors related attributes will be whole potential flow
    for d in df.debtor_name_1.unique():
        df.loc[df.debtor_name_1==d, d_node_flow] = df.loc[df.debtor_name_1==d, edge_flow].sum()   #node_flow will be used to calculate the energy in the node
    
    for c in df.customer_name_1.unique():
        df.loc[df.customer_name_1==c, c_node_eff] = np.nansum(df.loc[df.customer_name_1==c, edge_eff])
        df.loc[df.customer_name_1==c, node_flow] = np.nansum(df.loc[df.customer_name_1==c, edge_flow])
        df.loc[df.customer_name_1==c, energy] = np.nansum(df.loc[df.customer_name_1==c, edge_eff]*df.loc[df.customer_name_1==c, d_node_flow])

    return df





def create_flow_graphs(dict_start_end, comp_dict=comp_dict):
    """
    This function, given a dictionary of starting nodes as keys and list of sink nodes as values and a dictionary of components, will use end and start nodes
    to calculate the flow through the corresponding component contained in the second dictionary, returning a graph with a flow assigned to each edge.
    """
    
    graphs = []
    
    
    error_count=0
    
    for start in dict_start_end.keys():
        last_customers = dict_start_end[start][2]
        last_hybrids = dict_start_end[start][1]
        
        comp_number = dict_start_end[start][3]
        di_g = comp_dict[comp_number]
            
        if len(last_customers)==0:
            for end in last_hybrids: #considering the last hybrid
                try:
                    if start!=end:
                        graphs.append(nx.max_flow_min_cost(di_g, start,end))
                except:
                    error_count+=1
                    print("ERROR {} : there were problems with path {}  -  {} of length {} in component {}".format(error_count, start, end, dict_start_end[start][0], comp_number))
                    pass
        else: #the other version just consider last hybrids and works pretty well in predictions
            for end2 in last_customers: #considering the last hybrid
                try:
                    if start!=end2:
                        graphs.append(nx.max_flow_min_cost(di_g, start,end2))
                except:
                    error_count+=1
                    print("ERROR {} : there were problems with path {}  -  {} of length {} in component {}".format(error_count, start, end2, dict_start_end[start][0], comp_number))
                    pass
    
    return graphs