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
    hybrids = [i for i in buyers_name if i in set(sellers_name)]

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


def create_edges(df, sellers_colname='customer_name_1', buyers_colname='debtor_name_1', sellersid_colname = 'customer_id', fields = []):
    """
    This function creates a dataset of edges for network modelling
    """

    d=df.copy()

    #start and end point of each edge
    xs = []
    ys = []

    for idx in d.index:
        xs+=[d.loc[idx,sellers_colname]]
        ys+=[d.loc[idx,buyers_colname]]

    #connecting customers and debtors
    edges_couples = [(d.loc[idx,sellers_colname], d.loc[idx,buyers_colname]) for idx in d.index]

    edges_df = pd.DataFrame(data = {'xs':xs, 'ys':ys, 'tuples':edges_couples}, index = xs) #edges dataframe indexed by sellers names

    if len(fields)>0:
        for f in fields: #this step assumes that the stats have been already created for the whole dataset
            edges_df[f]=list([d.loc[(d[sellers_colname] == edges_df.iloc[k]['xs']) & (d[buyers_colname] == edges_df.iloc[k]['ys']), f].values[0] for k in range(len(edges_df))])

    return edges_df