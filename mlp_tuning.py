import numpy as np
import pandas as pd

# Import TensorFlow
import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.optimizers import RMSprop, Adam, SGD

from scripts_ml.models_utils import *

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.externals.joblib import parallel_backend

from scripts_ml.ann_utils import *



#importing data
preproc_folder = "enriched_time_seq"
datafolder = "../data/preproc_traintest/"+preproc_folder+'/'
prefix_time_seq = 'time_2018-04-30_imp_bg_'
valid_code = '_val_24000_6000_'
trainfile = '_traindata'
testfile = '_testdata'
postfix_time_seq_val = '_190822_99'
postfix_time_seq = '_190822_1018'
preproc_folder = "enriched_time_seq"
datafolder = "../data/preproc_traintest/"+preproc_folder+'/'
indexfile = '_fold_indexes'
expname = "MLP_"+preproc_folder+valid_code.split('_val_')[1][:-1]+"_imp"

[X_train, y_train, feature_labels] = pd.read_pickle(datafolder+prefix_time_seq+trainfile+postfix_time_seq+'.pkl') 
[X_test, y_test, feature_labels] = pd.read_pickle(datafolder+prefix_time_seq+testfile+postfix_time_seq+'.pkl') 
[val_X_train, val_y_train, val_feature_labels] = pd.read_pickle(datafolder+prefix_time_seq+valid_code+trainfile+postfix_time_seq_val+'.pkl') 
[val_X_test, val_y_test, val_feature_labels] = pd.read_pickle(datafolder+prefix_time_seq+valid_code+testfile+postfix_time_seq_val+'.pkl') 
indexes = pd.read_pickle(datafolder+prefix_time_seq+valid_code+indexfile+postfix_time_seq_val+'.pkl')

#recombining folds for grid search

val_X_all = []
val_y_all = []
indexes_tuples = []

count=0
start_tr=0

for idx in indexes:
    val_X_all.append(val_X_train[idx[0]])
    val_y_all.append(val_y_train[idx[0]])
    if count==0:
        test_idx = np.array(range(0, len(idx[1])))
    else:
        test_idx+=len(idx[1])
    val_X_all.append(val_X_test[test_idx])
    val_y_all.append(val_y_test[test_idx])
    
    
    if count==0:
        start_tst = len(idx[0])
    else:
        start_tr+=add_to_tr
        start_tst=start_tr+len(idx[0])
        
    indexes_tuples.append((np.array(range(start_tr, start_tr+len(idx[0]))), 
                          np.array(range(start_tst, start_tst+len(idx[1])))))
    
    add_to_tr = len(idx[0])+len(idx[1])
    
    count+=1

val_X_all = np.concatenate(val_X_all, axis=0)
val_y_all = np.concatenate(val_y_all, axis=0)

#tuning
print("Setting the grid...")
mlp = KerasClassifier(build_fn=create_mlp_model_GS)

input_shape = [X_train.shape[1]]

hidden_layers_no = [2, 3, 4, 5] 

hidden_nodes = [[10,10,10,5,5], [20,10,10,5,5],
                [50,20,20,10,5], [50,40,20,10,5],
                [50,40,40,20,10], [50,40,40,20,5],
                 [5,5,40,20,10], [5,10,40,20,5],
               [100,50,25,10,5], [100,80,50,20,10],
               [200,100,80,20,5], [100,50,50,20,5]] 

#[[5,5], [10,5], 
                #[20,5], [20,10], 
                #[50,5], [50,10], [50,20],
               #[80,5], [80,10], [80,20]]


hl_activations = [[tf.nn.relu]*5] #multiplied for the max number of hidden layers

dropout = [[0.3]*5, [0.4]*5, [0.5]*5, [0.6]*5, None]

optimizer = [RMSprop(), Adam(), SGD()]

batch_size = [128, 256, 512, 1024]

epochs = [50, 100, 200, 500]

class_weights = [{0:1, 1:10}, {0:1, 1:25}, {0:1, 1:50}]  

param_grid = {'hidden_layers_no': hidden_layers_no,
               'hidden_nodes': hidden_nodes,
               'hl_activations': hl_activations,
              'dropout': dropout,
               'optimizer': optimizer,
               'batch_size': batch_size,
              'epochs':epochs,
              'class_weight': class_weights
               }


scoring = {"AUC": "roc_auc", "Accuracy": make_scorer(accuracy_score)}

random_grid_search = True
n_iter = 100
GPU = True
GPU_device_name = "/device:GPU:0"
verbose=2
njobs=4


def main():

    if GPU:
        with tf.device(GPU_device_name):

            if not random_grid_search:
                searchname = 'grid'
                mlp_grid = GridSearchCV(estimator = mlp, param_grid = param_grid, 
                                                cv = rolling_window_idxs(indexes_tuples), 
                                                verbose=verbose, n_jobs =njobs, scoring=scoring, refit='AUC')

            else:
                searchname = 'randomgrid'
                mlp_grid = RandomizedSearchCV(estimator = mlp, param_distributions = param_grid, random_state=42,
                                              n_iter = n_iter,
                                                cv = rolling_window_idxs(indexes_tuples), 
                                                verbose=verbose, n_jobs =njobs, scoring=scoring, refit='AUC')
    else:
        if not random_grid_search:
                searchname = 'grid'
                mlp_grid = GridSearchCV(estimator = mlp, param_grid = param_grid, 
                                                cv = rolling_window_idxs(indexes_tuples), 
                                                verbose=verbose, n_jobs =njobs, scoring=scoring, refit='AUC')

        else:
            searchname = 'randomgrid'
            mlp_grid = RandomizedSearchCV(estimator = mlp, param_distributions = param_grid, random_state=42,
                                            n_iter = n_iter,
                                            cv = rolling_window_idxs(indexes_tuples), 
                                            verbose=verbose, n_jobs =njobs, scoring=scoring, refit='AUC')

    # Fit the grid search model
    print("Fitting the grid...")
    if GPU:
        with parallel_backend('threading'):
            mlp_grid.fit(val_X_all, val_y_all, verbose=0)
    else:
        mlp_grid.fit(val_X_all, val_y_all, verbose=0)
    best_dict = mlp_grid.best_params_

    print("-------------------------{} SEARCH DONE!------------------")

    text_file = open(searchname+"_output.txt", "w")
    for key in best_dict.keys():
        print(str(key)+" : "+str((best_dict[key])))
        text_file.write(str(key)+" : "+str(best_dict[key]))
    text_file.close()


if __name__ == '__main__':
    main()