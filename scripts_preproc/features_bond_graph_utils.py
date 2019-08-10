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
import networkx as nx

from scripts_network.network_analysis_exp import *
from scripts_network.network_bond_graph import *
from scripts_network.network_modelling import *


def add_bg_features(df, col_to_calc_effort, effort_col, flow_col, col_to_calc_flow,
                    node_flow_col, energy_col, c_node_eff_col,
                    d_node_flow_col, shock_col, red_coeff,
                    seller_col='customer_name_1', buyer_col='debtor_name_1'):
    """
    This function adds bond graph features to the input dataset returning it in a modified version in relation to a given credit event.
    - col_to_calc_effort: the column used to calculate the effort for bond graph formalism
    - effort_col: the name that will be assigned to the effort column
    - flow_col: the name that will be assigned to the flow column
    - col_to_calc_flow: the name of the column used to calculate the flow for bond graph formalism
    - node_flow_col: the name that will be assigned to the node flow
    - energy_col: the name that will be assigned to the energy column
    - c_node_eff_col: the name that will be assigned to the node effort column
    - d_node_flow_col: the name that will be assigned to the node flow column
    - shock_col: the name that will be assigned to the shock column
    - red_coeff: coefficient adjusting the calculation tolerance of the flow in directed graphs
    - seller_col: sellers column name
    - buyer_col: buyers column name

    Example settings for pastdue90 case:

    df, 
    col_to_calc_effort = 'payment_date_mismatch',
    effort_col='p90_edge_eff', 
    flow_col='p90_edge_flow', 
    col_to_calc_flow = 'cd_pastdue90_r',
    node_flow_col ='p90_node_flow', 
    energy_col='p90_energy', 
    c_node_eff_col = 'p90_c_node_eff',
    d_node_flow_col = 'p90_d_node_flow', 
    shock_col = 'flow_shock_p90'
    red_coeff = 10**4
    """
    df = df.copy()

    #calculate edge effort and flow
    print("Calculating effort and flow...")
    edge_effort = df.groupby([seller_col, buyer_col]).apply(lambda x:np.nansum(x[col_to_calc_effort]))
    df[effort_col] = [edge_effort[(df.loc[i, seller_col], df.loc[i, buyer_col])] for i in df.index]
    df[effort_col] = df[effort_col].replace([np.inf, -np.inf, np.nan], 0)
    df[flow_col] = df[col_to_calc_flow].replace([np.inf, -np.inf, np.nan], 0)

    #building the undirected graph
    print("Creating the undirected graph of the whole dataset network...")
    g = nx.Graph() #some networkx functions only work on undirected graphs

    for cus in df[seller_col].unique():
        for debt in df.loc[df[seller_col]==cus, buyer_col].unique():
        
            df_tmp = df[df[seller_col]==cus]
            df_tmp2 = df_tmp[df_tmp[buyer_col]==debt]
            g.add_edge(debt, cus)

    #adding features to the dataset
    print("Adding effort and flow feature to the dataset...")
    df = build_bgtfeat(df, g, edge_flow=flow_col, edge_eff=effort_col, node_flow=node_flow_col, energy=energy_col,
                      c_node_eff=c_node_eff_col, d_node_flow=d_node_flow_col)

    #isolating graph components and creating directed graphs on each of them
    print("Isolating components and creating directed graphs...")
    a = nx.connected_components(g)
    components = [c for c in sorted(a, key=len, reverse=True)]
    directed_graphs = create_directed_graphs(components, df, red_coeff, energy_col)
    
    #dictionary of independent components
    comp_dict = dict(zip(range(len(components)), directed_graphs))

    #finding hybrids
    #the nodes that are both buyers and sellers
    print("Looking for hybrid nodes...")
    customers = df[seller_col].unique()
    debtors = df[buyer_col].unique()
    hybrids = list(set(customers).intersection(set(debtors)))

    # finding the debtor nodes connected to hybrids
    connected_to_hyb = [d for d in debtors if len(node_neighborhood(g, d, 1, hybrids))>1]

    #flow modelling
    print("Modelling the flow...")
    #all debtors connected with hybrids with sources of flow >0
    df_flow_chain = df[df[buyer_col].isin(connected_to_hyb) & df[energy_col]>0] 
    conn_to_hyb_sources = df_flow_chain[buyer_col].unique()

    # for each debtor node connected to hybrids, we find the maximum 'chain degree' (the last node reached by the flow of that particular source)
    max_degrees = [including_degree(g, nn, hybrids, comp_dict) for nn in conn_to_hyb_sources]
    #exlcuding the debtors from the last reached nodes, since only customers can accumulate/dissipate or transmit (if they are hybrids) the flow
    max_degrees = pd.Series(max_degrees).apply(lambda x:(x[0], x[1], list(x[2].intersection(set(df[seller_col]))), x[3]))

    #this dictionary will have as keys the starting buyers nodes, as values tuples of (farest flow degree, last hybrid contained in the path, last nodes reached 1 degree after the last hybrids)
    dict_start_end = dict(zip(conn_to_hyb_sources, max_degrees))

    #create flow graphs
    print("Creating flow graphs...")
    graphs = create_flow_graphs(dict_start_end, comp_dict)

    #identifying graphs with flow>0 and total flow computed
    pos_graphs = positive_graphs(graphs)

    #creating a dictionaries of flow overlaps + check
    print("Overlapping the flows...")
    #it will contain the terminal nodes between edges having positive flow with relative flow value
    sum_dict=flow_dict(graphs, pos_graphs)

    #for each graph component, the flow is added at each edge as capacity
    #all edges flow (addition to the sum_dict of the starting static flows as final overlap)
    sum_dict=sum_of_flows(sum_dict, comp_dict)

    #assigning the flow value to instruments as a new feature
    print("Adding shock-propagation features...")
    for edge in sum_dict.keys():
        df.loc[(df[seller_col]==edge[1]) & (df[buyer_col]==edge[0]), shock_col] = sum_dict[edge]

    print("Done!")
    return df



#-----------------------------------------
# MAIN
#-----------------------------------------

def main():
    print("features_utils.py executed/loaded..")

if __name__ == "__main__":
    main()