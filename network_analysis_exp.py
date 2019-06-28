#==================================================================================#
# Author       : Davide Mariani                                                    #  
# University   : Birkbeck College, University of London                            # 
# Programme    : Msc Data Science                                                  #
# Script Name  : network_analysis_exp.py                                           #
# Description  : functions for network analysis and exploration                    #
# Version      : 0.1                                                               #
#==================================================================================#
# This file contains several functions to analyse and explore a network using      #
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




def node_neighborhood(G, node, n, nfilter=[], n_only=False):
    """
    This function, given a graph G and a starting node, will find the node neighbors into a certain degree n
    returning them as a list.
    Giving a list of nodes as a filter as last attribute, it will select only paths containing those nodes filtering the output.
    If n_only==True, it will select only the nodes at distance n from the starting node.
    """
    neigh = nx.single_source_dijkstra(G, node)
    path_lengths = neigh[0]
    paths = neigh[1]
    
    
    if len(nfilter)==0:
        if not n_only:
            all_neigh = [node for node, length in path_lengths.items() if length <= n]
        else:
            all_neigh = [node for node, length in path_lengths.items() if length == n]
        return all_neigh

    else:

        if not n_only:
            all_neigh = [node for node, length in path_lengths.items() if length <= n]
        else:
            all_neigh = [node for node, length in path_lengths.items() if length == n]

        nodes = set()
        nodes.add(node)
        for f in nfilter:
            for g in all_neigh:
                if f in set(paths[g]):
                    nodes.add(g)
        return list(nodes)


def node_component(node, graphdict):
    """
    This function, given a node and a dictionary of graphs, will return the dictionary index of the graph containing the input node
    """
    for graph in graphdict.keys():
        if node in set(graphdict[graph].node):
            return graph



def including_degree(G, node, nfilter, comp_dict=comp_dict):
    """
    This function, given a graph G, a starting node, and a list of nodes to include in the path, will return the maximum degree of connections including those nodes in a shortest-path plus the immediate
    neighbors to reach.
    In addition to the maximum degree, it will add a set of the last hybrids reached, a set of the last nodes of the last neighborhood and the number of the component this path belongs to. 
    """
    degree=1
    check = node_neighborhood(G, node, degree, nfilter, n_only=True)
    hybrids_at_this_deg = [n for n in check if (n in nfilter and n!=node)]
    last_hybrids=hybrids_at_this_deg[:]
    last_reached = set(check).difference(set(hybrids_at_this_deg))
    
    while len(hybrids_at_this_deg)>0:
        degree+=1
        check = node_neighborhood(G, node, degree, nfilter, n_only=True)
        hybrids_at_this_deg = [n for n in check if (n in nfilter and n!=node)]
        last_reached = set(check).difference(set(hybrids_at_this_deg))
        if len(hybrids_at_this_deg)>1:
            last_hybrids = hybrids_at_this_deg[:]
            
    return degree, last_hybrids, last_reached, node_component(node, comp_dict)   


