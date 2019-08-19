#==================================================================================#
# Author       : Davide Mariani                                                    #  
# University   : Birkbeck College, University of London                            # 
# Programme    : Msc Data SCience                                                  #
# Script Name  : visualization_perfgrid.py                                         #
# Description  : utils for data visualizations                                     #
# Version      : 0.2                                                               #
#==================================================================================#
# This file contains general purpose visualization functions initially based on    #
# bokeh v1.2.0                                                                     #
#==================================================================================#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from scripts_viz.visualization_utils import *
from scripts_viz.visualization_utils import TTQcolor

#bokeh
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.layouts import gridplot, row, column
from bokeh.plotting import figure



#PERFORMANCE VISUALIZATION GRID

def performance_grid(viz, 
                     model_filter = None, 
                     legend_font_size='9pt', 
                     fpr_font_size='9pt', 
                     bestFprOnly=True, 
                     rocs_p_width=600, 
                     rocs_p_height=600,
                     rocs_line_width=2, 
                     single_spider=True, 
                     spider_in_row=1, 
                     spiders_params = ['tp_rate', 'tn_rate', 'fp_rate', 'fn_rate', 'test_auc', 'val_auc'],
                     spider_p_width=300, 
                     spider_p_height=300, 
                     spider_text_size='12pt', 
                     spider_line_width=4.5, 
                     spider_fill_alpha=0.1,
                     spider_margin_distance=0.25, 
                     single_row_folds=True, 
                     folds_p_width=1200, 
                     folds_p_height=1200,
                     folds_xlabelorientation=1.55, 
                     folds_group_text_font_size='6pt',
                     folds_in_row=1, 
                     spreadsheet_cols = ['n_estimators', 'max_depth', 'max_features', 'max_leaf_nodes', 
                                        'min_samples_leaf', 'min_samples_split', 'bootstrap',
                                        'criterion', 'val_auc', 'test_auc'],
                     spr_index_header = 'RF',
                     spr_height=80, 
                     spr_width=1200, 
                     spr_index_width=25,
                     spr_row_height=25,
                     plot_feat_importance=True,
                     normalize_importance=True,
                     fimp_text_group_size = '10pt',
                     colors=[TTQcolor['azureBlue'], TTQcolor['richOrange'], TTQcolor['algae'], TTQcolor['yell'], TTQcolor['redBrown'], TTQcolor['bloodRed']]):
    """
    This function create a gridplot containing several charts for model performance evaluation:
    - ROC curves comparison for testing and validation phases
    - AUC diagrams for validation folds
    - Comparison spiders
    - Feature importances for random forest and SGD classifiers

    viz is the pandas dataframe containing the mlflow data retrieved with the function create_exp_df from scripts_mlflow.mlflow_utils

    """

    #switching between model selection and full experiment viz
    if model_filter == None:
        roc_val_data = [{'fpr':[float(v) for v in viz.loc['roc_val_fpr',model].split(',')[:-1]],
                    'tpr':[float(v) for v in viz.loc['roc_val_tpr',model].split(',')[:-1]],
                     'auc':viz.loc['val_auc',model]} for model in viz.columns]

        roc_test_data = [{'fpr':[float(v) for v in viz.loc['roc_test_fpr',model].split(',')[:-1]],
                    'tpr':[float(v) for v in viz.loc['roc_test_tpr',model].split(',')[:-1]],
                     'auc':viz.loc['test_auc',model]} for model in viz.columns]

        spider_folds_models = [m for m in list(viz.columns)]

        spreadsheet_models = [m for m in list(viz.loc['model_type'].unique())]

    else:
        roc_val_data = [{'fpr':[float(v) for v in viz.loc['roc_val_fpr',model].split(',')[:-1]],
                    'tpr':[float(v) for v in viz.loc['roc_val_tpr',model].split(',')[:-1]],
                     'auc':viz.loc['val_auc',model]} for model in viz.columns if viz.loc['model_type', model] in model_filter]

        roc_test_data = [{'fpr':[float(v) for v in viz.loc['roc_test_fpr',model].split(',')[:-1]],
                    'tpr':[float(v) for v in viz.loc['roc_test_tpr',model].split(',')[:-1]],
                     'auc':viz.loc['test_auc',model]} for model in viz.columns if viz.loc['model_type', model] in model_filter]

        spider_folds_models = [m for m in list(viz.columns) if viz.loc['model_type', m] in model_filter]

        spreadsheet_models = model_filter


    #ROC curves
    val_roc = plot_rocs(roc_val_data, label = list(viz.columns), title_lab = 'Validation performance', 
               p_width=rocs_p_width, p_height=rocs_p_height, line_width=rocs_line_width,
                    colors = colors, legend_font_size=legend_font_size, fpr_font_size=fpr_font_size,
                   bestFprOnly=bestFprOnly, show_legend=False)

    test_roc = plot_rocs(roc_test_data, label = list(viz.columns), title_lab = 'Test performance', 
               p_width=rocs_p_width, p_height=rocs_p_height, line_width=rocs_line_width,
                    colors = colors, legend_font_size=legend_font_size, fpr_font_size=fpr_font_size,
                   bestFprOnly=bestFprOnly, show_legend=False)

    #Spider plots
    if single_spider:
    
        spider = spiderWebChart(spider_folds_models, spiders_params, 
                            [viz.loc[spiders_params, m] for m in spider_folds_models], 
                            colors=colors, text_size=spider_text_size,
                               title='Overall Comparison', p_height=spider_p_height, p_width=spider_p_width, margin_distance=spider_margin_distance,
                               legend_location='top_right', show_legend=False, line_width=spider_line_width, fill_alpha=spider_fill_alpha)

    else:
    #distribution over multiple rows
        cols = []
        rows = []

        s_w_size = spider_p_width//folds_in_row
        if len(spider_folds_models)%spider_in_row==0:
            s_h_size = spider_p_height//folds_in_row
        else:
            s_h_size = spider_p_height//(folds_in_row+1)

        for m in range(len(spider_folds_models)):
            if m%spider_in_row==0 or m==len(spider_folds_models):
                rows = []
                cols.append(rows)

            single_spider = spiderWebChart([spider_folds_models[m]], spiders_params, 
                                    [viz.loc[spiders_params, spider_folds_models[m]]], 
                                    colors=[colors[m]], text_size=spider_text_size,
                                       title='', p_height=s_h_size, p_width=s_w_size, 
                                       margin_distance=spider_margin_distance,
                                       legend_location='top_right', show_legend=False, line_width=spider_line_width, fill_alpha=spider_fill_alpha)

            rows.append(single_spider)
        
        spider = gridplot(cols)

    #Validation Folds
    f = [fold for fold in viz.index if 'val_auc_fold_' in fold]
    sorter = list(pd.Series(f).apply(lambda x:int(x.split('_')[-1])))
    folds_name = list(pd.Series(sorted(list(zip(f,sorter)), key=lambda f:f[1])).apply(lambda x:x[0]))

    if single_row_folds:

        folds = histfolds(spider_folds_models, folds_name, viz, plot_w=folds_p_width, plot_h=folds_p_height, title="Validation folds performance",
                     colors=colors, xlabelorientation=folds_xlabelorientation, group_text_font_size=folds_group_text_font_size)
    else:
        cols = []
        rows = []

        if len(spider_folds_models)%folds_in_row==0:
            h_size = folds_p_height//folds_in_row
        else:
            h_size = folds_p_height//(folds_in_row+1)

        breakp = folds_in_row

        sequence = list(range(breakp, len(spider_folds_models)+1, breakp))
        
        odd=False
        if len(spider_folds_models) not in sequence:
            odd = True
            sequence+=[len(spider_folds_models)]

        for m in sequence:
            if m%breakp==0 or m==sequence[-1]:
                rows = []
                cols.append(rows)
            
            if odd and m==sequence[-1]:
                
                diff = m-sequence[sequence.index(m)-1]
                
                row_folds = histfolds(spider_folds_models[m-diff:m], folds_name, viz, plot_w=folds_p_width//folds_in_row*diff, plot_h=h_size, title="Validation folds performance",
                                 colors=[colors[m]], xlabelorientation=folds_xlabelorientation, group_text_font_size=folds_group_text_font_size)
            else:
                row_folds = histfolds(spider_folds_models[m-breakp:m], folds_name, viz, plot_w=folds_p_width, plot_h=h_size, title="Validation folds performance",
                                 colors=colors[m-breakp:m], xlabelorientation=folds_xlabelorientation, group_text_font_size=folds_group_text_font_size)

            rows.append(row_folds)

        folds = gridplot(cols)

    #Spreadsheet info viz
    ss = modelSpreadsheet(viz, spreadsheet_cols, spreadsheet_models, color_cells=True, colors=colors[:len(spider_folds_models)],
                      index_header=spr_index_header, height=spr_height, width=spr_width, index_width=spr_index_width,
                     row_height=spr_row_height)

    #feature importance and output
    if plot_feat_importance:
        fi = feature_importance(viz, spreadsheet_models, normalize = normalize_importance, colors=colors, xgroup_text_font_size=fimp_text_group_size)

        l = gridplot([[ss],
              [row(val_roc, folds)],
              [row(test_roc, spider)],[fi]])

    else:
        l = gridplot([[ss],
              [row(val_roc, folds)],
              [row(test_roc, spider)]])

    return l