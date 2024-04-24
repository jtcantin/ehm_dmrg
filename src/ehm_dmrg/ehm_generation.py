import os
import time

import datashader as ds
import h5py

# import scikitplot as skplt
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats as sps
import seaborn as sns
from datashader.mpl_ext import dsshow
from sklearn import linear_model

# from mosek.fusion import *
from sklearn.covariance import ShrunkCovariance
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.inspection import permutation_importance

# import cvxopt
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, SGDRegressor

# from xgboost import XGBRegressor
from sklearn.metrics import log_loss, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


class Score:
    r2 = 0.0                   
    cross_vali_score = 0.0  
    
class Regressor:
    def __init__(self, name):
        self.name = name
        self.score = Score()
        self.y_pred = None
        self.reg = None

models_map = {'Linear': linear_model.LinearRegression(),
              #'Ridge': Ridge(),
              #'Lasso': linear_model.Lasso(),
              'RF': RandomForestRegressor(), 
              'SVR': SVR(),
              #'KNeighbors': KNeighborsRegressor(),
              'Gaussian Process': GaussianProcessRegressor(),
              'DT': DecisionTreeRegressor(),
            #   'NN': MLPRegressor(),
              #'Gradient Descent': SGDRegressor(),
            #   'Xgboost': XGBRegressor()
             }

def get_reg_models(models_map, X_train, X_test, y_train, y_test):
    result_map = {}
    for key, val in models_map.items():
        result = Regressor(key)
        reg = val
        result.score.cross_vali_score = np.mean(cross_val_score(reg, X_train, y_train, cv=5))        
        result.reg = reg.fit(X_train, y_train) 
        result.y_pred = reg.predict(X_test) 
        result.score.r2 = r2_score(y_test, result.y_pred) 
        result.score.rms = mean_squared_error(y_test, result.y_pred, squared=False)
        #result.score.LL = log_loss(y_test, result.y_pred)
        result.score.pcc = sps.pearsonr(y_test, result.y_pred)[0]
        result_map[key] = result 
    return result_map

def get_scores(map):
    for key, val in map.items():
        print(val.name.upper() + ' SCORE: ')
        print('RMS' + '  =  ', val.score.rms)
        #print('Log likelihood score' + '  =  ', val.score.LL)
        print('PCC' + '  =  ', val.score.pcc)
        print('R2' + '  =  ', val.score.r2)
        #print('CV' + '  =  ', val.score.cross_vali_score)
        print('------------------------------------------\n')
        
def datashader(ax, x, y):
    df = pd.DataFrame(dict(x=x, y=y))
    dsartist = dsshow(
        df,
        ds.Point("x", "y"),
        ds.count(),
        vmin = 0,
        vmax = 35,
        norm = "linear",
        aspect = "auto",
        ax = ax)
    plt.colorbar(dsartist,label = 'Number of points per pixel')

def show_predicted_vs_actual(models, y_test):   
    for key, val in models.items():
        fig, ax = plt.subplots()
        x = y_test
        y = val.y_pred
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r-', lw=1)
        ax.set_xlabel('Predicted Sharpe Ratio')
        ax.set_ylabel('Actual Sharpe Ratio')
        ax.title.set_text(val.name.upper())
        datashader(ax, x, y)
        plt.show()

def scores_graph(map):
    comp_df = pd.DataFrame(columns = ('Models', 'RMS', 'R2'#, 'PCC', 'CV'
                                     ))
    for i in map:
        row = {'Models': i,
               'RMS': map[i].score.rms,
               'R2' : map[i].score.r2,
               'PCC': map[i].score.pcc,
               #'CV' : map[i].score.cross_vali_score
              }
        comp_df = comp_df.append(row, ignore_index=True)
    ax = comp_df.plot.bar(x='Models',rot=0 ,figsize=(8,4) #,figsize=(10,5)
                         )
    ax.set_title('')
    #plt.legend(loc='below', ncol=3)
    #ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.15),
          fancybox=True, shadow=True, ncol=3)
    
    for p in ax.containers:
        ax.bar_label(p, fmt='%.2f', label_type='edge',fontsize=8)
        
def scores_graph2(map):
    comp_df = pd.DataFrame(columns = ('Models', 'RMS','PCC'))
    for i in map:
        row = {'Models': i,
               'RMS': map[i].score.rms,
               'R2' : map[i].score.r2,
               'PCC': map[i].score.pcc,
              }
        comp_df = comp_df.append(row, ignore_index=True)
    ax = comp_df.plot.bar(x='Models',rot=0 ,figsize=(8.5,4) #,figsize=(10,5)
                         )
    ax.set_title('')
    
    for p in ax.containers:
        ax.bar_label(p, fmt='%.2f', label_type='edge',fontsize=8)



# features = list(mlDataProc.drop([target], axis = 1, inplace = False))


# See this
# https://christophm.github.io/interpretable-ml-book/feature-importance.html

def feature_importance_permutation(models, X, y, features):
    for key, val in models.items():
        feature_score = permutation_importance(val.reg, X, y, scoring='neg_mean_squared_error')
        feature_score_df = pd.Series(feature_score.importances_mean, index=features).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 8))
        sns.barplot(x=feature_score_df, y=feature_score_df.index)
        plt.xlabel('Feature Importance Score')
        plt.ylabel('Features')
        plt.title("Permutation-Based Feature Importance " + "(" + val.name.upper() +")")
        plt.show()

# tree-based feature importance
def feature_importance_impurity(models, features):
    for key, val in models.items():
        if key == 'RF' or key == 'Xgboost' or key == "DT" or key == "AdaBoost":
            feature_score = pd.Series(val.reg.feature_importances_, index=features).sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(6, 8))
            sns.barplot(x=feature_score, y=feature_score.index)
            plt.xlabel('Feature Importance Score')
            plt.ylabel('Features')
            plt.title("Impurity-Based Feature Importance " + "(" + val.name.upper() +")")
            plt.show()

def preprocess(dataDf):
    
    target = 'discarded_weight'
    dff = dataDf.drop([target], axis =1)

    # remove columns with variance < 0.01
    dff = dff.loc[:, dff.var() > 0.01]
    
    # remove identical columns
    dff = dff.T.drop_duplicates().T
    
    
    dfC = dff.join(dataDf[target])
    
    # remove inf values from data (i.e. retain finite values)
    dfC_trunc = dfC[np.isfinite(dfC).all(1)]
    
    dfC_proc = dfC_trunc.drop([target], axis=1)
    
    # normalize data
    dfC_proc = (dfC_proc - dfC_proc.min())/(dfC_proc.max() - dfC_proc.min())
    
    # standardize data
    dfC_proc = (dfC_proc - dfC_proc.mean())/dfC_proc.std()
    
    # add targets to processed data
    dfC_proc.insert(loc=0, column = target, value = dataDf[target])
    
    return dfC_proc
