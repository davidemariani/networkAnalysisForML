#==================================================================================#
# Author       : Davide Mariani                                                    #  
# University   : Birkbeck College, University of London                            # 
# Programme    : Msc Data Science                                                  #
# Script Name  : network_bond_graph.py                                             #
# Description  : functions for modelling the network as a bond graph               #
# Version      : 0.3                                                               #
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
    This function takes a pandas dataframe of instruments dfi and a graph g in order to calculate effort, flow and energy at each edge and node,
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
    
    #customers related attributes
    for c in df.customer_name_1.unique():
        df.loc[df.customer_name_1==c, c_node_eff] = np.nansum(df.loc[df.customer_name_1==c, edge_eff])
        df.loc[df.customer_name_1==c, node_flow] = np.nansum(df.loc[df.customer_name_1==c, edge_flow])
        df.loc[df.customer_name_1==c, energy] = np.nansum(df.loc[df.customer_name_1==c, edge_eff]*df.loc[df.customer_name_1==c, d_node_flow])

    return df


def create_flow_graphs(dict_start_end, comp_dict):
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
    print('{} graphs successfully created'.format(len(graphs)))
    return graphs


def create_directed_graphs(components, df, red_coeff, energy='imp_energy'):
    """
    This function, given a list of graph components and a dataframe of instruments df, recreate the corresponding directed graphs,
    assigning weights to the edges depending on energy values at the corresponding buyer nodes.
    Directions go from buyer to seller.
    red_coeff is a 'reduction coefficient' for the flow values to be assigned to each graph's edges
    """
    
    dir_graphs=[]

    #calculating flow value on the whole network
    df_whole_temp = df.copy()
    df_whole_temp['capacity'] = [cap if cap>0 else 1 for cap in round(df[energy]/(red_coeff),3)]
    df_whole_temp['capacity'] = df_whole_temp['capacity'].astype(int)

    df_whole_temp['weight'] = abs(round(red_coeff/df_whole_temp['capacity'],3))
    df_whole_temp['weight'] = df_whole_temp['weight'].replace([np.inf, -np.inf], 0)
    df_whole_temp['weight'] = df_whole_temp['weight'].astype(int)

    for comp in components:
         
         df_temp = df_whole_temp[df_whole_temp.customer_name_1.isin(comp)]

         di_g = nx.from_pandas_edgelist(df_temp,
                                        source = 'debtor_name_1',
                                        target = 'customer_name_1',
                                        edge_attr=['capacity', 'weight'],
                                        create_using = nx.DiGraph())

         dir_graphs.append(di_g)
        
    return dir_graphs



def positive_graphs(graphs):
    """
    This function, given a list of graphs generated by 'create_flow_graphs', will return a list of integers,
    indicating the graphs having flow>0
    """

    #identifying graphs with flow>0 and total flow computed
    values_list = [] #list of values for each graph
    pos_graphs = set() #set of graphs having positive flow values

    for n in range((len(graphs))): #for each graph
        for v in graphs[n].values(): #for each set of values at each path of each graph
            for vv in v.values(): #for each value
                if vv>0:
                    pos_graphs.add(n) 
        values_list.append(sum([sum(list(v.values())) for v in graphs[n].values()]))
    print("Total calculated shock flow is {} over {} graphs with positive flow value".format(sum(values_list), len(pos_graphs)))

    return pos_graphs


def flow_dict(graphs, pos_graphs, print_check=False):
    """
    This function, given a list of graphs generated by 'create_flow_graphs', will return a dictionaries
    of flow overlaps where for each node involved in a path the corresponding flow value is stored.
    This is the dictionary of sub-flows that will be summed up to obtain the total one.
    """

    sum_dict={}

    for pg in pos_graphs:
        check = graphs[pg]

        for j in check.keys():
            for k in check[j]:
                if check[j][k]>0:
                    if print_check:
                        print('graph '+str(pg),'-',j,'-', k,'-', check[j])
                    if (j,k) not in sum_dict.keys():
                        sum_dict[(j,k)] = check[j][k]
                    else:
                        sum_dict[(j,k)] += check[j][k]
    return sum_dict


def sum_of_flows(sum_dict, comp_dict):
    """
    This function uses sum_dict and comp_dict to create a new dictionary where the total flow at each edge is stored
    """

    sum_dict_ = sum_dict.copy()

    for j in comp_dict.keys():
        for e in comp_dict[j].edges:
            if e not in sum_dict.keys():
                sum_dict_[e]=comp_dict[j].get_edge_data(e[0], e[1])['capacity']
            else:
                sum_dict_[e]+=comp_dict[j].get_edge_data(e[0], e[1])['capacity']
    return sum_dict_