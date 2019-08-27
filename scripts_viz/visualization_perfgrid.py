#==================================================================================#
# Author       : Davide Mariani                                                    #  
# University   : Birkbeck College, University of London                            # 
# Programme    : Msc Data SCience                                                  #
# Script Name  : visualization_perfgrid.py                                         #
# Description  : utils for data visualizations                                     #
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


#UTILS
def color_generator(experiment):
    """
    This function generates a list of colors from the TTQ palette.
    The color depends on the type of experiment, recognised by the words contained in its name
    """

    color_list = []
    for e in experiment:
        if "benchmarks" in e:
            if "shuffle" in e:
                color_list.append(TTQcolor['azureBlue'])
            elif "time_old" in e:
                color_list.append(TTQcolor['electric'])
            elif "time_opt" in e:
                color_list.append(TTQcolor['pea'])
        elif "enriched" in e:
            if "shuffle" in e:
                color_list.append(TTQcolor['yell'])
            elif "time_seq" in e and "opt" not in e:
                color_list.append(TTQcolor['richOrange'])
            elif "time_seq" in e and "opt" in e:
                color_list.append(TTQcolor['bloodRed'])
    return color_list
    


#PERFORMANCE VISUALIZATION GRID

def performance_grid(viz, 
                     model_filter = None, 
                     legend_font_size='9pt', 
                     fpr_font_size='9pt', 
                     bestFprOnly=True, 
                     rocs_p_width=600, 
                     rocs_p_height=600,
                     rocs_line_width=2, 
                     rocs_model_appendix = None,
                     normalize_spider=False,
                     single_row_folds=True, 
                     folds_p_width=1200, 
                     folds_p_height=1200,
                     folds_xlabelorientation=1.55, 
                     folds_group_text_font_size='6pt',
                     folds_in_row=1, 
                     spreadsheet_settings = {},
                     plot_feat_importance=True,
                     normalize_importance=True,
                     fimp_text_group_size = '10pt',
                     colors=[TTQcolor['azureBlue'], TTQcolor['richOrange'], TTQcolor['algae'], TTQcolor['yell'], TTQcolor['redBrown'], TTQcolor['bloodRed']],
                     add_spider = False,
                     single_spider=True, 
                     spider_in_row=1, 
                     spiders_params = ['tp_rate', 'tn_rate', 'fp_rate', 'fn_rate', 'test_auc', 'val_auc'],
                     spiders_params_to_norm = [],
                     spider_p_width=300, 
                     spider_p_height=300, 
                     spider_text_size='12pt', 
                     spider_line_width=4.5, 
                     spider_fill_alpha=0.1,
                     spider_margin_distance=0.25,
                    add_corr_scatter=True,
                    scatter_height=600,
                    scatter_width=600,
                    scatter_title_size='12pt',
                    scatter_circle_size=5
                    ):
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
                     'auc':viz.loc['val_auc',model]} for model in sorted(list(viz.columns))]

        roc_test_data = [{'fpr':[float(v) for v in viz.loc['roc_test_fpr',model].split(',')[:-1]],
                    'tpr':[float(v) for v in viz.loc['roc_test_tpr',model].split(',')[:-1]],
                     'auc':viz.loc['test_auc',model]} for model in sorted(list(viz.columns))]

        spider_folds_models = [m for m in sorted(list(viz.columns))]

        spreadsheet_models = [m for m in sorted(list(viz.loc['model_type'].unique()))]

    else:
        model_filter = sorted(model_filter)

        roc_val_data = [{'fpr':[float(v) for v in viz.loc['roc_val_fpr',model].split(',')[:-1]],
                    'tpr':[float(v) for v in viz.loc['roc_val_tpr',model].split(',')[:-1]],
                     'auc':viz.loc['val_auc',model]} for model in sorted(list(viz.columns)) if viz.loc['model_type', model] in model_filter]

        roc_test_data = [{'fpr':[float(v) for v in viz.loc['roc_test_fpr',model].split(',')[:-1]],
                    'tpr':[float(v) for v in viz.loc['roc_test_tpr',model].split(',')[:-1]],
                     'auc':viz.loc['test_auc',model]} for model in sorted(list(viz.columns)) if viz.loc['model_type', model] in model_filter]

        spider_folds_models = [m for m in sorted(list(viz.columns)) if viz.loc['model_type', m] in model_filter]

        spreadsheet_models = model_filter


    #ROC curves

    val_roc = plot_rocs(roc_val_data, title_lab = 'Validation performance', model_appendix = rocs_model_appendix,
               p_width=rocs_p_width, p_height=rocs_p_height, line_width=rocs_line_width,
                    colors = colors, legend_font_size=legend_font_size, fpr_font_size=fpr_font_size,
                   bestFprOnly=bestFprOnly, show_legend=True)

    test_roc = plot_rocs(roc_test_data, title_lab = 'Test performance', model_appendix = rocs_model_appendix,
               p_width=rocs_p_width, p_height=rocs_p_height, line_width=rocs_line_width,
                    colors = colors, legend_font_size=legend_font_size, fpr_font_size=fpr_font_size,
                   bestFprOnly=bestFprOnly, show_legend=True)

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

    #correlation scatterplot
    if add_corr_scatter:
        if len(spider_folds_models)==2:
            vals_1 = [v for v in viz.loc['test_predictions', spider_folds_models[0]].split(',')[:-1]]
            vals_1 = [float(i) for i in vals_1]
            vals_2 = [v for v in viz.loc['test_predictions', spider_folds_models[1]].split(',')[:-1]]
            vals_2 = [float(i) for i in vals_2]
            scatter = cor_scatterplot(vals_1, vals_2, xaxis_label=spider_folds_models[0], p_height=scatter_height, p_width=scatter_width,
                    yaxis_label=spider_folds_models[1], circle_size=scatter_circle_size, title_size=scatter_title_size)
        #else:
        #implement the other case!

    #Spreadsheet info viz
    if len(spreadsheet_settings)>1:
        ss = []
        for sset in spreadsheet_settings:
            ss.append(modelSpreadsheet(**sset))
        
        ss = [column(ss)]
    else:
        ss = [modelSpreadsheet(**spreadsheet_settings[0])]

    if add_spider:
        #Spider plots
        viz_ = viz.copy()

        if normalize_spider:
            if len(spreadsheet_models)==1:
                for param in spiders_params_to_norm:
                    values = viz_.loc[param, spider_folds_models].values.astype(float)
                    if len(set(values))>1:
                        viz_.loc[param, spider_folds_models] = values/(max(values)) # (values-min(values))/(max(values)-min(values))
                    else:
                        if abs(values[0])==0:
                            viz_.loc[param, spider_folds_models]=0
                        else:
                            viz_.loc[param, spider_folds_models]=1
                        

            else:
                print("SPIDER TRYING TO DISPLAY INFO FOR MORE THAN ONE MODEL TYPE - NORMALIZATION IS DISABLED")

        if single_spider:
                #data[model] = list(pd.Series(data[model]).apply(lambda x:(x-min(data[model]))/(max(data[model])-min(data[model]))))

            spider = spiderWebChart(spider_folds_models, spiders_params, 
                                [viz_.loc[spiders_params, m] for m in spider_folds_models], 
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
                                        [viz_.loc[spiders_params, spider_folds_models[m]]], 
                                        colors=[colors[m]], text_size=spider_text_size,
                                           title='', p_height=s_h_size, p_width=s_w_size, 
                                           margin_distance=spider_margin_distance,
                                           legend_location='top_right', show_legend=False, line_width=spider_line_width, fill_alpha=spider_fill_alpha)

                rows.append(single_spider)
        
            spider = gridplot(cols)

        #feature importance and output if there's spider
        if plot_feat_importance:
            fi = feature_importance(viz, spreadsheet_models, normalize = normalize_importance, colors=colors, xgroup_text_font_size=fimp_text_group_size)

            l = gridplot([ss,
                  [row(val_roc, folds)],
                  [row(test_roc, spider)],[fi]])

        else:
            l = gridplot([ss,
                  [row(val_roc, folds)],
                  [row(test_roc, spider)]])
    else:
        #feature importance and output if there's no spider
        if plot_feat_importance:
            fi = feature_importance(viz, spreadsheet_models, normalize = normalize_importance, colors=colors, xgroup_text_font_size=fimp_text_group_size)

            l = gridplot([ss,
                  [row(val_roc, folds)],
                  [row(test_roc, scatter)],
                  [fi]])

        else:
            l = gridplot([ss,
                  [row(val_roc, folds)],
                  [row(test_roc, scatter)]])


    return l