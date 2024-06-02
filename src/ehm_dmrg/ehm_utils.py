import time

import datashader as ds
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sps
import seaborn as sns
from datashader.mpl_ext import dsshow
from sklearn import linear_model
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

# from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge, RidgeCV, SGDRegressor
from sklearn.metrics import log_loss, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# Here we define a two class objects, Score and Regressor.
# In Score we have various scores defined for a regression model.
# In Regressor we specify various attributes for a given regressor,
# such as its name and its score.


class Score:
    rms = 0.0
    pcc = 0.0
    r2 = 0.0
    cross_vali_score = 0.0
    log_likelihood_score = 0.0


class Regressor:
    def __init__(self, name):
        self.name = name
        self.score = Score()
        self.y_pred = None
        self.reg = None


# We now specify various regression models as a dictionary,
# with key being the regression name and value being the rgressor.
# Here we have several regressor deactivated, i.e., commented out.
# Dependenting on our need, they can be activated.
# Note that some regressors, such as 'Gaussian Process' sometimes
# take significant time, compared to others, to train.
# If we identified other regressors, we can simply add them here similar to existing regressors.
# The activated regressors are the one that we, the optimization teasm, typically use.

regresssion_models = {
    "Linear": linear_model.LinearRegression(),
    "RF": RandomForestRegressor(),
    "SVR": SVR(),
    "DT": DecisionTreeRegressor(),
    "NN": MLPRegressor(),
    # "Xgboost": XGBRegressor(),
    #'Ridge': Ridge()
    #'Lasso': linear_model.Lasso(),
    #'KNeighbors': KNeighborsRegressor(),
    "Gaussian Process": GaussianProcessRegressor(),
    #'Gradient Descent': SGDRegressor(),
}


def construct_EHMs(X_train, X_test, y_train, y_test):
    """
    Using the train and test data as inputs, this functions constructs
    various regression models (i.e., EHMs) defined in the regresssion_models above
    and computes various scores for the models specified in the Score class above.
    """
    EHMs = {}
    for key, val in regresssion_models.items():
        result = Regressor(key)
        reg = val

        result.reg = reg.fit(X_train, y_train)
        result.y_pred = reg.predict(X_test)

        result.score.rms = mean_squared_error(y_test, result.y_pred, squared=False)
        result.score.pcc = sps.pearsonr(y_test, result.y_pred)[0]
        result.score.r2 = r2_score(y_test, result.y_pred)
        # result.score.cross_vali_score = np.mean(cross_val_score(reg, X_train, y_train, cv=5))
        # result.score.log_likelihood_score = log_loss(y_test, result.y_pred)

        EHMs[key] = result
    return EHMs


def get_scores(EHMs):
    """
    This fuction gets regression models, i.e., EHMs,
    and returns scores of the models.
    Three sores that we typically used are activated here.
    """
    for key, val in EHMs.items():
        print(val.name.upper() + " SCORE: ")
        print("RMS" + "  =  ", val.score.rms)
        print("PCC" + "  =  ", val.score.pcc)
        print("R2" + "  =  ", val.score.r2)
        # print('CV' + '  =  ', val.score.cross_vali_score)
        # print('LL' + '  =  ', val.score.log_likelihood_score)
        print("------------------------------------------\n")


def datashader(ax, x, y):
    """
    This is helper function used in other functions for plotting.
    """
    df = pd.DataFrame(dict(x=x, y=y))
    dsartist = dsshow(
        df,
        ds.Point("x", "y"),
        ds.count(),
        vmin=0,
        vmax=35,
        norm="linear",
        aspect="auto",
        ax=ax,
    )
    plt.colorbar(dsartist, label="Number of points per pixel")


def plot_predicted_vs_actual(EHMs, y_test, actual_axis_label="", predict_axis_label=""):
    """
    This function takes the constructed EHMs and returns their plot,
    where the x axis of the plot is actual hardness and y axis is the predicted hardness.
    Labes of axis are taken as optional input as it may change depening on the problem;
    the hardness may be different. Examples for axis lable is "runtime [sec]"", "Log runtime [sec]"
    or "Sharpe Ratio". For "Log runtime [sec]" we take: predict_axis_label = "Log predicted runtime [sec]"
    and actual_axis_label = "Log actual runtime [sec]". Note that labes are taken as sting inputs.
    """
    for key, val in EHMs.items():
        fig, ax = plt.subplots()
        x = y_test
        y = val.y_pred
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r-", lw=1)
        ax.set_xlabel(actual_axis_label)
        ax.set_ylabel(predict_axis_label)
        ax.title.set_text(val.name.upper())
        # datashader(ax, x, y)
        ax.scatter(x, y)
        plt.show()


def scores_graph(EHMs, plot_title):
    """
    This function takes the constructed EHMs as input and returns plot of their scores.
    The second input is title for the plot. Example: plot_title = 'Scores for LKH'.
    """
    # comp_df = pd.DataFrame(columns=("Models", "RMS", "PCC", "R2"))  # ,'CV','LL'
    row_list = []
    for key in EHMs.keys():
        row = {
            "Models": key,
            "RMS": EHMs[key].score.rms,
            "PCC": EHMs[key].score.pcc,
            "R2": EHMs[key].score.r2,
            #'CV': EHMs[key].score.cross_vali_score
            #'LL': EHMs[key].score.log_likelihood_score
        }
        row_list.append(row)

    comp_df = pd.DataFrame(row_list)
    # pd.concat(objs, *, axis=0, join='outer', ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, sort=False, copy=None)

    ax = comp_df.plot.bar(x="Models", rot=0, figsize=(8, 4))
    ax.set_title(plot_title)
    # plt.legend(loc='below', ncol=3)
    # ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        fancybox=True,
        shadow=True,
        ncol=3,
    )

    for p in ax.containers:
        ax.bar_label(p, fmt="%.2f", label_type="edge", fontsize=8)


def feature_importance_permutation(EHMs, features, X, y):
    """
    This function gets the regression models, i.e., EHMs,
    and computes importance of features in constructiing the model baed on the permutation-based method.
    The function returns plots of feature importace for each model.
    Also see this https://christophm.github.io/interpretable-ml-book/feature-importance.html
    """
    for key, val in EHMs.items():
        feature_score = permutation_importance(
            val.reg, X, y, scoring="neg_mean_squared_error"
        )
        feature_score_df = pd.Series(
            feature_score.importances_mean, index=features
        ).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 8))
        sns.barplot(x=feature_score_df, y=feature_score_df.index)
        plt.xlabel("Feature Importance Score")
        plt.ylabel("Features")
        plt.title(
            "Permutation-Based Feature Importance " + "(" + val.name.upper() + ")"
        )
        # print(feature_score_df)
        plt.show()


def feature_importance_impurity(EHMs, features):
    """
    This function gets the EHMs (regression models)
    and computes importance of features in constructiing the model baed on the impurity-based method.
    The function returns plots of feature importace for each model. As this method applies to tree-based
    mosels, we need to specify them in the if statement below.
    Also see this https://christophm.github.io/interpretable-ml-book/feature-importance.html
    """
    for key, val in EHMs.items():
        if key == "RF" or key == "Xgboost" or key == "DT" or key == "AdaBoost":
            feature_score = pd.Series(
                val.reg.feature_importances_, index=features
            ).sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(6, 8))
            sns.barplot(x=feature_score, y=feature_score.index)
            plt.xlabel("Feature Importance Score")
            plt.ylabel("Features")
            plt.title(
                "Impurity-Based Feature Importance " + "(" + val.name.upper() + ")"
            )
            print(feature_score)
            plt.show()


def preprocess(rawdata_df, target, normalize_target=False,pca_decorrelation=False):
    """
    This functions takes the rawdata as a dataframe and performs the preprocessing step.
    The function returns the preprocessed data as a datarame.
    It is assumed that the rawdata contains problem features and one target value, where
    the taget is the hardness, or solution returned from the solver.
    If the rawdata has target value for multiple solvers, it must be first reduced to data
    for one solver before feeding to this fucntion.
    """

    # df = pd.read_csv(rawdata)

    # remove the target column from the data
    preproc_df = rawdata_df.drop([target], axis=1)

    # # remove columns with variance < 0.01
    # preproc_df = preproc_df.loc[:, preproc_df.var() > 0.01]

    # remove identical columns
    preproc_df = preproc_df.T.drop_duplicates().T

    # join the target column
    preproc_df = preproc_df.join(rawdata_df[target])

    # remove inf values from the data (i.e. retain finite values)
    preproc_df = preproc_df[np.isfinite(preproc_df).all(1)]

    # remove columns with no change in values
    preproc_df = preproc_df.loc[:, preproc_df.nunique() > 1]

    if not normalize_target:
        # remove the target column
        preproc_df = preproc_df.drop([target], axis=1)

    # normalize data
    preproc_df = (preproc_df - preproc_df.min()) / (preproc_df.max() - preproc_df.min())

    # standardize data
    preproc_df = (preproc_df - preproc_df.mean()) / preproc_df.std()

    if pca_decorrelation:
        # PCA decorrelation
        from sklearn.decomposition import PCA
        if normalize_target:
            target_values = preproc_df[target]
            preproc_df = preproc_df.drop([target], axis=1)
        pca = PCA(n_components=preproc_df.shape[1])
        preproc_df = pca.fit_transform(preproc_df)
        preproc_df = pd.DataFrame(preproc_df)
        compenents = pca.components_
        singular_values = pca.singular_values_
        feature_names = pca.feature_names_in_

        if normalize_target:
            # return target to processed data
            
            preproc_df.insert(loc=0, column=target, value=target_values)
            

    if not normalize_target:
        # add  target to processed data
        preproc_df.insert(loc=0, column=target, value=rawdata_df[target])

    if not pca_decorrelation:
        return preproc_df
    else:
        return preproc_df, compenents, singular_values, feature_names


def prune_features(preproc_df, to_keep, target,keep_all=False):
    if keep_all:
        return preproc_df
    # Drop all columns except the target and the columns in to_keep
    for col in preproc_df.columns:
        if col not in to_keep and col != target:
            preproc_df = preproc_df.drop(columns=[col])

    return preproc_df
