#==================================================================================#
# Author       : Davide Mariani                                                    #  
# University   : Birkbeck College, University of London                            # 
# Programme    : Msc Data Science                                                  #
# Script Name  : network_visualization.py                                          #
# Description  : functions for network visualization                               #
# Version      : 0.1                                                               #
#==================================================================================#
# Functions for Social Network visualization using bokeh 1.0.4                     #
#==================================================================================#

#base modules
import pandas as pd
import numpy as np
import datetime
from datetime import date,datetime
import math

from network_modelling import create_network

#networkx
import networkx as nx

#bokeh basics
from bokeh.io import show, output_notebook, output_file, curdoc
from bokeh.plotting import figure
from bokeh.layouts import gridplot, widgetbox, layout, row
from bokeh.models import (
    ColumnDataSource,
    Circle,
    HoverTool,
    Range1d,
    Plot,
    MultiLine,
    GraphRenderer, 
    StaticLayoutProvider,
    CustomJS,
    Slider)
from bokeh.transform import factor_cmap, linear_cmap
from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.models.widgets import MultiSelect

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



def visualize_graph(graph, edges, nodes, nodes_name_column='Company_Name', nodes_size_column='size', nodes_type_column='Type_2',
                    title = 'Network Graph', plot_w = 900, plot_h = 900, file_output = '', nx_k=0.028, nx_iterations=25,
                      nodes_colors = [TTQcolor['sky'], TTQcolor['Salmon'], TTQcolor['marketplaceOrange']], to_highlight = '',
                      edges_colors = [TTQcolor['whiteGrey'], TTQcolor['warningRed']]):

    """
    This function will give visual attributes to the graph.
    """

    graph = from_networkx(graph, nx.spring_layout, k=nx_k, iterations=nx_iterations)

    #unfortunately the use of list comprehension at next step is necessary
    #since bokeh doesn't seem to support other collections like Series or arrays
    graph.node_renderer.data_source.data['index'] = [i for i in nodes['Company_Name']] #setting the company names
    graph.node_renderer.data_source.data['size'] = [s for s in nodes['size']] #setting node sizes
    graph.node_renderer.data_source.data['type'] = [t for t in nodes['Type_2']] #setting types

    graph.node_renderer.glyph = Circle(size='size', fill_color=factor_cmap('type', nodes_colors,    #creating nodes
                                                                    ['buyer', 'seller and buyer', 'seller']),
                                                                     fill_alpha=0.8, line_color='white', line_width=0.5)

    graph.node_renderer.nonselection_glyph = Circle(size='size', fill_color=factor_cmap('type', nodes_colors, #creating non-selected nodes
                                                                nodes['Type_2'].unique()),
                                               fill_alpha=0.1, line_alpha=0.05)

    
    if len(to_highlight)>0:
        graph.edge_renderer.data_source.data['highlight']=list(edges[to_highlight])
        graph.edge_renderer.glyph = MultiLine(line_color=linear_cmap('highlight', edges_colors, 
                                                                     False,True), line_width=0.15, line_alpha=0.85)
    else:
        graph.edge_renderer.glyph = MultiLine(line_width=0.15, line_alpha=0.85)

    graph.node_renderer.hover_glyph = Circle(size='size', fill_alpha=0.0, line_width=3, line_color='green') #creating hover settings for circles
    graph.edge_renderer.hover_glyph = MultiLine(line_color='#abdda4', line_width=0.8) #creating hover settings for edges

    graph.selection_policy = NodesAndLinkedEdges()
    graph.inspection_policy = NodesAndLinkedEdges()

    #plot
    plot = figure(title = title, x_range=(-1.2,1.2), y_range=(-1.2,1.2), tools = "box_select,tap,wheel_zoom,reset,pan,save")
    
    #tools and plot graphics
    TOOLTIPS = [('Company Name', '@index'), ('Company Type', '@type')] #hover tooltips
    plot.add_tools(HoverTool(tooltips=TOOLTIPS))
    plot.renderers.append(graph)
    plot.xgrid.grid_line_color = None
    plot.ygrid.grid_line_color = None
    plot.axis.visible = False

    return plot


def graph_from_coordinates(nodes, edges, nodes_name_column='Company_Name', nodes_size_column='size', nodes_type_column='Type_2', nodes_coord_column='coords',
                 edges_start_column='xs', edges_end_column='ys',
                nodes_colors = [TTQcolor['sky'], TTQcolor['richPeach'], TTQcolor['Salmon']],
                 edges_colors = [TTQcolor['azureBlue'], TTQcolor['warningRed']],
                to_highlight = 'is_pastdue90', title=''):
    """
    This function visualizes a graph using custom locations defined into the nodes dataset.
    
    nodes: nodes df
    edges: edges df
    nodes_name_column: name of the column from nodes df containing the name of each node
    nodes_size_column: name of the column from nodes df containing the size of each node
    nodes_type_column: name of the column from nodes df containing the type of each node
    nodes_coords_column: name of the column from nodes df containing the x,y coordinates of each node
    edges_start_column: name of the column from edges df containing the start of each edge
    edges_end_column: name of the column from edges df containing the end of each edge
    """
    #edges cleaning
    edges = edges.drop_duplicates(subset=[edges_start_column, edges_end_column])
    
    graph = GraphRenderer()
    
    #nodes
    graph.node_renderer.data_source.add(list(nodes[nodes_name_column]), 'index')
    graph.node_renderer.data_source.add(list(nodes[nodes_size_column]), 'size')
    graph.node_renderer.data_source.add(list(nodes[nodes_type_column]), 'type')
    graph.node_renderer.glyph = Circle(size='size', fill_color=factor_cmap('type', nodes_colors,
                                                                    ['buyer', 'seller and buyer', 'seller']),
                                fill_alpha=0.8, line_color='white', line_width=0.1)
    
    #edges
    graph.edge_renderer.data_source.data = dict(start = list(edges[edges_start_column]),
                                               end = list(edges[edges_end_column]))


    if len(to_highlight)>0:
        graph.edge_renderer.data_source.data['highlight']=list(edges[to_highlight])
        graph.edge_renderer.glyph = MultiLine(line_color=linear_cmap('highlight', edges_colors, 
                                                                     False,True), line_width=0.15, line_alpha=0.85)
    else:
        graph.edge_renderer.glyph = MultiLine(line_width=0.15, line_alpha=0.85)
    
    #inspection policy and hover changes
    graph.node_renderer.hover_glyph = Circle(size='size', fill_alpha=0.0, line_width=3, line_color='green') #creating hover settings for circles
    graph.edge_renderer.hover_glyph = MultiLine(line_color='#abdda4', line_width=0.8) #creating hover settings for edges
    graph.inspection_policy = NodesAndLinkedEdges()
    
    #layout
    graph_layout = dict(zip(list(nodes[nodes_name_column]), list(nodes[nodes_coord_column])))
    graph.layout_provider = StaticLayoutProvider(graph_layout = graph_layout)
    
    #plot
    plot = figure(title = title, x_range=(-1.2,1.2), y_range=(-1.2,1.2), tools = "box_select,tap,wheel_zoom,reset,pan,save")
    
    #tools and plot graphics
    TOOLTIPS = [('Company Name', '@index'), ('Company Type', '@type')] #hover tooltips
    plot.add_tools(HoverTool(tooltips=TOOLTIPS))
    plot.renderers.append(graph)
    plot.xgrid.grid_line_color = None
    plot.ygrid.grid_line_color = None
    plot.axis.visible = False
    
    return plot