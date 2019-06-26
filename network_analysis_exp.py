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


