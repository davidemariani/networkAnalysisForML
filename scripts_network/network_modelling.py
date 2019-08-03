#==================================================================================#
# Author       : Davide Mariani                                                    #  
# University   : Birkbeck College, University of London                            # 
# Programme    : Msc Data Science                                                  #
# Script Name  : network_modelling.py                                              #
# Description  : functions for network modelling                                   #
# Version      : 0.1                                                               #
#==================================================================================#
# This file contains several functions to model a network of buyers and sellers    #
# from an instrument dataset using networkx                                        #
#==================================================================================#


#importing main modules
import pandas as pd
import numpy as np
import datetime
import re
import networkx as nx



def count_column(df, columnname):
        """
        Given an instrument dataset, this function counts the number of True values for a certain column indicating
        the relationship between a customer and a debtor
        """
        return df.groupby(['customer_name_1', 'debtor_name_1']).apply(lambda x:x[columnname].sum())



def prepare_connections(dfi, sellers_colname='customer_name_1', buyers_colname='debtor_name_1', buyersid_colname='debtor_id', sellersid_colname = 'customer_id', uniqueid_colname='uid',
                        fields = []):

    """
    This function will prepare the main connections for network modelling, returning a dataframe without duplicates for nodes and edges creation 
    """
    
    df = dfi.copy()

    #some debtors id are shared among more than one debtor, so to make them unique their id is combined with their name, while each seller will be represented by its customer_id
    df['debtor_id+name'] = df[buyersid_colname].map(str)+' : '+df[buyers_colname].map(str)

    #to create connection between companies who have the same debtor, join the data with itself on the 'debtor_id+name' column
    column_edge = 'debtor_id+name'
    main_cols = [sellersid_colname, uniqueid_colname, sellers_colname, buyersid_colname, buyers_colname]
    columns_to_keep = main_cols + fields + [column_edge]
    data_to_merge = df[columns_to_keep].dropna(subset = [column_edge]).drop_duplicates(subset=main_cols, keep='last')
    renaming_dict = dict(zip([cc for cc in columns_to_keep if cc !='debtor_id+name'], [c+'_2' for c in columns_to_keep if c !='debtor_id+name']))
    data_to_merge = data_to_merge.merge(data_to_merge[columns_to_keep].rename(columns = renaming_dict), on = column_edge)

    #removing duplicate connections
    d = data_to_merge.copy().drop_duplicates(subset = [sellers_colname,  buyers_colname], keep='last') #column_edge,
    #d = d.drop(d.loc[d[sellers_colname+"_2"] < d[sellers_colname]].index.tolist(), inplace = True)

    return d
   

def create_nodes_df(df, sellers_colname='customer_name_1', buyers_colname='debtor_name_1', buyersid_colname='debtor_id', sellersid_colname = 'customer_id'):
    """
    This function will create a dataset of nodes with attributes for analysis and visualization
    """
    d=df.copy()

    #checking debtors which are also customers
    buyers_name = df[buyers_colname].unique()
    sellers_name = df[sellers_colname].unique()
    hybrids = list(set(buyers_name).intersection(set(sellers_name)))

    #adding columns flag the customers which are also debtors
    d['customer_id_is_also_debtor'] = d[sellers_colname].isin(hybrids)
    d['debtor_id_is_also_customer'] = d[buyers_colname].isin(hybrids)

    #selection criteria
    cust_selection = ~d['customer_id_is_also_debtor']
    debt_selection = ~d['debtor_id_is_also_customer']
    hyb_cust_selection = d['customer_id_is_also_debtor']
    hyb_debt_selection = d['debtor_id_is_also_customer']

    #debtors_
    debt_list = d.loc[debt_selection,buyers_colname].unique()

    #customers_
    cust_list = d.loc[cust_selection,sellers_colname].unique()

    #debtors hybrids_
    hyb_debt_list = d.loc[hyb_debt_selection,buyers_colname].unique()

    #customers hybrids_
    hyb_cust_list = d.loc[hyb_cust_selection,sellers_colname].unique()
    
    #types
    type_customers = ['seller' for i in  d.loc[cust_selection, sellers_colname].unique()]
    type_h_customers = ['seller' for j in  d.loc[hyb_cust_selection, sellers_colname].unique()]
    type_debtors = ['buyer' for k in d.loc[debt_selection,buyers_colname].unique()]
    type_h_debtors = ['buyer' for w in d.loc[hyb_debt_selection,buyers_colname].unique()]
    all_types = type_customers + type_debtors + type_h_customers + type_h_debtors

    #types_2 (for viz)
    type_customers_2 = ['seller' for i in  d.loc[cust_selection,sellers_colname].unique()]
    type_h_customers_2 = ['seller and buyer' for j in  d.loc[hyb_cust_selection,sellers_colname].unique()]
    type_debtors_2 = ['buyer' for k in d.loc[debt_selection,buyers_colname].unique()]
    type_h_debtors_2 = ['seller and buyer' for w in d.loc[hyb_debt_selection,buyers_colname].unique()]
    all_types_2 = type_customers_2 + type_debtors_2 + type_h_customers_2 + type_h_debtors_2

    #names
    name_customers = [ct for ct in d.loc[cust_selection,sellers_colname].unique()]
    name_h_customers = [hct for hct in d.loc[hyb_cust_selection,sellers_colname].unique()]
    name_debtors = [db for db in d.loc[debt_selection,buyers_colname].unique()]
    name_h_debtors = [hdb for hdb in d.loc[hyb_debt_selection,buyers_colname].unique()]
    all_nodes = name_customers + name_debtors + name_h_customers + name_h_debtors


    #ids
    id_customers = [str(o).replace("[","").replace("]","") for o in [d.loc[d[sellers_colname]==y,sellersid_colname].unique() for y in name_customers]]
    id_h_customers = [str(o).replace("[","").replace("]","") for o in [d.loc[d[sellers_colname]==u,sellersid_colname].unique() for u in name_h_customers]]
    id_debtors = [str(o).replace("[","").replace("]","") for o in [d.loc[d[buyers_colname]==y,buyersid_colname].unique() for y in name_debtors]]
    id_h_debtors = [str(o).replace("[","").replace("]","") for o in [d.loc[d[buyers_colname]==f,buyersid_colname].unique() for f in name_h_debtors]]
    all_ids = id_customers + id_debtors + id_h_customers + id_h_debtors


    #creation of the nodes dataset
    nodes_df = pd.DataFrame(dict(zip(['Company_Name', 'ID', 'Type', 'Type_2'], [all_nodes, all_ids, all_types, all_types_2])))
    nodes_df = nodes_df.drop_duplicates(subset = 'Company_Name') 
    
    return nodes_df


def create_edges_df(df, sellers_colname='customer_name_1', buyers_colname='debtor_name_1', sellersid_colname = 'customer_id', fields = []):
    """
    This function creates a dataset of edges for network modelling
    """

    d=df.copy()
    
    #connecting customers and debtors
    g_flat = nx.from_pandas_edgelist(d,
                                 source=buyers_colname,
                                 target=sellers_colname,
                                 create_using=nx.DiGraph)

    
    edges_couples = list(g_flat.edges)

    edges_df = pd.DataFrame(list(g_flat.edges), columns=['xs', 'ys']) #edges dataframe indexed by sellers names
    edges_df['edges_couples'] = edges_couples

    if len(fields)>0:
        for f in fields: #this step assumes that the stats have been already created for the whole dataset
            temp_df = pd.DataFrame(df.groupby([buyers_colname, sellers_colname]).sum()[f]>0)
            new_df = pd.DataFrame({f: list(temp_df[f]), 'edges_couples':list(temp_df.index)})
            edges_df = edges_df.merge(new_df, on='edges_couples')

    return edges_df




def create_network(edges, nodes, nodes_size_range = (6,15),
                   R=0.54, nperlayer=26, nodescircles=0.065):
    """
    This function will create the network structure using networkx. It will also modify edges and nodes datasets adding SNA related features.
    This is an auxiliar function of netowrk_info
    """

    edges = edges.drop_duplicates(subset = ['xs', 'ys'])

    # build the nx graph
    G=nx.Graph()
    G.add_edges_from(edges.edges_couples)

    #nodes analysis to define their centrality
    centrality = nx.degree_centrality(G) #centrality dictionary
    nodes['centrality'] = [centrality[n] for n in list(nodes['Company_Name'])]

    #nodes size
    max_size = nodes_size_range[1]
    min_size = nodes_size_range[0]
    nodes['size'] = np.around(np.interp(nodes['centrality'], (nodes['centrality'].min(), nodes['centrality'].max()), (min_size, max_size)),2)

    #coordinates
    pos = init_layout(G, nodes, R, nperlayer, nodescircles)
    coordinates = [np.array(pos[j]) for j in nodes['Company_Name']]
    nodes['coords'] = coordinates

    return G, edges, nodes



def nodes_post(nodes, edges, nodes_name_column, nodes_size_range = (6,15), nxk = 0.35, nxit = 35):
    """
    This function takes a nodes dataframe and add features for the over time graph visualization (size and coordinates) 
    - it needs to be used on the most recent snapshot first, in order to assign coords to all the nodes. 
    These coordinates will be used for the visualization of the previous timeframe after that. -
    nodes: nodes dataframe to modify
    edges: corresponding edges dataframe
    nodes_name_column: name of the column of nodes df containing the name of the company
    nodes_size_range: tuple containing minimum and maximum sizes of the nodes
    """
    n = nx.Graph()
    n.add_edges_from(list(set(list(zip(list(edges['xs']),list(edges['ys']))))))
    
    #size
    bc = nx.degree_centrality(n)
    centralities =pd.Series([bc[i] for i in nodes[nodes_name_column]])
    max_size = nodes_size_range[1]
    min_size = nodes_size_range[0]
    sizes = np.interp(centralities, (centralities.min(), centralities.max()), (min_size, max_size)) 
    nodes['size'] = sizes
    sizes_dict = dict(zip(list(nodes[nodes_name_column]), list(nodes['size'])))
    
    #coordinates
    pos = nx.spring_layout(n, k=nxk, iterations=nxit)
    coordinates = [np.array(pos[j]) for j in nodes[nodes_name_column]]
    nodes['coords'] = coordinates
    
    return pos, sizes_dict


#CIRCULAR LAYOUT FUNCTIONS (Helpers for create_network)

#position of the k-th point out of n total points place on layers circles around (xc,yc)
#all circles have radius within maxlayeroffset from radius r
#instead of specifying layers one may specify number of points per layer pperlayer

def innercircle_vec(labels, r=.4, nlayers = 3, maxlayeroffset=.42, xc=0, yc=0):
    """
    This function create coordinate positions based on the primetable for visualizing the network of buyers and sellers
    on a layout based on circles, where each seller is closer to the center of the plot on the basis of its centrality,
    and the buyers connected to it tend to be in the surrounding circle.
    It takes as attributes the series of object names to sort by centrality,
    the size of the radius, the number of layers to create, the maximum offset between two layers,
    the coordinates of the centre of the plot.
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
    This function creates a graph layout for visualization using innercicle_vec.
    It takes as attributes an undirected graph, the radius of the first circle,
    the number of seller nodes per layer and the radius of the circles containing
    the buyer nodes.
    """
    #sellers
    sellernames = pd.DataFrame(nodes.loc[nodes.Type == "seller",:]).\
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


def network_info(G, edges, nodes, nodes_size_range = (6,15), 
                 to_highlight = 'is_pastdue90', circularLayout = False):
    """
    OUTDATED! Please use create_network instead!

    This function will add information to edges and nodes dataset, as well as the graph, to prepare them for visualization.
    The output will be a networkx graph, an updated edges dataset and an updated nodes dataset.
    """

    edges_tuples = list(G.edges) #update with the actual number of edges after network building

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