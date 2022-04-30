from multiprocessing import Pool, current_process, cpu_count
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

import shap
import numpy as np 
import time
import tqdm
import copy
from collections import defaultdict

from random import choices, seed
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression

from multiprocessing import Pool, current_process, cpu_count
from datetime import datetime
from random import choices, seed

import shap
import numpy as np 
import time
import tqdm
import copy
from collections import defaultdict
from ..utils.helpers import globalize
from ..utils.misc import CustomScaler, plot_lowess, standard_name, add_annotations

# For now, the default model is XGBoost

def parallel_bootstrap_shap_analysis(df, target, xgb_hyperparams, explainer_type, iterations=int(np.round(1000*np.e)), method='sample_with_replacement', train_size=0.8, stratification_factor=None):
    """ By default the bootstrap analysis does sample with replacement. 
    if method == 'train_test_split' each iteration of bootstap will contain train_size portion of the data for training."""
    start = time.time()
    nSamples = df.shape[0] 
    sampleIdxs = range(nSamples)
    mdlFeatures = [feat for feat in df.columns if feat != target]
    print("Running bootstrap analysis using", method)
    tasks=range(iterations)
    X = df[mdlFeatures].values
    y = df[target].values
    @globalize
    def parallel_bootstrap_SHAP_analysis_fcn(seed_value):
        if method == 'sample_with_replacement':
            seed(seed_value)
            train_index = choices(sampleIdxs,k=nSamples-1)# Analogous to LOO approach
            val_index = np.array(list((set(sampleIdxs).difference(train_index)))) # This ensures there's no duplicated valIdxs
        elif method == 'train_test_split':
            if stratification_factor == None:
                sss = ShuffleSplit(n_splits=1, train_size=train_size, random_state=seed_value)
                train_index, val_index = sss.split(X=df[mdlFeatures].values).__next__()
            else:
                sss = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=seed_value)
                train_index, val_index = sss.split(X=df[mdlFeatures].values, y=df[stratification_factor]).__next__()

        X_tr, X_val = np.take(X, train_index,axis=0), np.take(X, val_index,axis=0)
        y_tr, y_val = np.take(y, train_index,axis=0), np.take(y, val_index,axis=0)

        if xgb_hyperparams['objective'] == 'reg:squarederror':
            model = xgb.XGBRegressor(**xgb_hyperparams, n_jobs=1, n_estimators=100, random_state=seed_value)
        elif xgb_hyperparams['objective'] == 'binary:logistic':
            model = xgb.XGBClassifier(**xgb_hyperparams, n_jobs=1, n_estimators=100, use_label_encoder=False, random_state=seed_value)
        elif xgb_hyperparams['objective'] == 'survival:cox':
            model = xgb.XGBRegressor(**xgb_hyperparams, n_jobs=1, n_estimators=100, use_label_encoder=False, random_state=seed_value)
        else:
            print('Unsupported model type')
            return
        model.fit(X_tr,y_tr)

        if explainer_type == 'prob':
            explainer = shap.TreeExplainer(model,data=X, feature_dependence="interventional", 
                                           model_output="probability")
            shap_values_curr = explainer.shap_values(X_val)
        elif explainer_type == 'raw': #aka logit when binary classification
            explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
            shap_values_curr = explainer.shap_values(X_val)
        elif explainer_type == 'interventional': 
            explainer = shap.TreeExplainer(model,data=X, model_output="interventional")
            shap_values_curr = explainer.shap_values(X_val)

        append_arr= np.concatenate((np.array([explainer.expected_value]*val_index.shape[0]), np.array([seed_value]*val_index.shape[0]), val_index)).reshape(-1,len(val_index)).T
        output_arr = np.concatenate((shap_values_curr, append_arr), axis=1)
        return output_arr
    
    pool = Pool(processes=cpu_count())
    r = list(tqdm.notebook.tqdm(pool.imap(parallel_bootstrap_SHAP_analysis_fcn, tasks), total=len(tasks)))
    
    pool.close()
    colnames = copy.deepcopy(mdlFeatures)
    colnames.extend(['expectedValue', 'bootsIteration', 'index'])
    bootsDF = pd.DataFrame(np.concatenate(tuple(r)), columns=colnames).astype(dict(zip(['expectedValue', 'bootsIteration', 'index'], [float,int,int])))
    end = time.time()
    print("   Execution time: {:.2f}s".format(round(end - start,5)))
    return bootsDF
    
def bootstrap_SHAP_analysis(df, target, model, explainer_type = 'prob', iterations=np.round(1000*np.e), method='sample_with_replacement', train_size=0.8, stratification_factor=None):
    """
    This performs a bootstrap SHAP analysis using a tree-based XGB model. 
    Other models aren't supported yet, but could be extended to them. 
    xgb_model : XGBClassifier() or xgb.XGBRFClassifier()
    """
    start = time.time()
    nSamples = df.shape[0] 
    sampleIdxs = range(nSamples)
    mdlFeatures = [feat for feat in df.columns if feat != target]
    save_d = True
    if save_d: 
        modelBoots_d = {}
        explainerBoots_d = {}
        expectedValueBoots_d = {}
        idxsBoots_d = {}
    for i in tqdm.tqdm_notebook(range(iterations)): #2718 : This makes it such that there would be ~1000 bootstrapped models for available each sample
        # seed(i)
        # train_index = choices(sampleIdxs,k=nSamples-1)# Analogous to LOO approach
        # val_index = list(set(sampleIdxs).difference(train_index)) # This ensures there's no duplicated valIdxs
        if method == 'sample_with_replacement':
            seed(i)
            train_index = choices(sampleIdxs,k=nSamples-1)# Analogous to LOO approach
            val_index = np.array(list((set(sampleIdxs).difference(train_index)))) # This ensures there's no duplicated valIdxs
        elif method == 'train_test_split':
            if stratification_factor == None:
                sss = ShuffleSplit(n_splits=1, train_size=train_size, random_state=i)
                train_index, val_index = sss.split(X=df[mdlFeatures].values).__next__()
            else:
                sss = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=i)
                train_index, val_index = sss.split(X=df[mdlFeatures].values, y=df[stratification_factor]).__next__()

        X_tr, X_val = df[mdlFeatures].iloc[train_index], df[mdlFeatures].iloc[val_index]
        y_tr, y_val = df[target].iloc[train_index], df[target].iloc[val_index]

        model.fit(X_tr,y_tr)

        # currPredictions_val = model.predict_proba(X_val)[:,1]
        # currPredictions_train = model.predict_proba(X_tr)[:,1]

        if explainer_type == 'prob':
            explainer = shap.TreeExplainer(model,data=df.drop(columns=target),
                                               feature_dependence="interventional", model_output="probability")
            shap_values_curr = explainer.shap_values(df[mdlFeatures].iloc[val_index])
        elif explainer_type == 'raw':
            explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
            shap_values_curr = explainer.shap_values(df[mdlFeatures].iloc[val_index])

        elif explainer_type == 'interventional': 
            explainer = shap.TreeExplainer(model,data=df.drop(columns=target),
                                               feature_dependence="interventional", model_output="raw")
            shap_values_curr = explainer.shap_values(df[mdlFeatures].iloc[val_index])


        if i == 0: 
            shap_values = shap_values_curr
            val_indecies=val_index
            bootsIteration = [i]*len(val_index)
        else:    
            shap_values = np.concatenate([shap_values, shap_values_curr])
            # val_indecies.extend(val_index) 
            val_indecies=np.concatenate([val_indecies, val_index])
            bootsIteration.extend([i]*len(val_index))
            
        modelBoots_d[i] = model
        explainerBoots_d[i] =explainer
        expectedValueBoots_d[i] = explainer.expected_value
        idxsBoots_d[i] = val_index


    bootsDF = pd.DataFrame(shap_values, columns = mdlFeatures)
    bootsDF['index'] = val_indecies
    bootsDF['bootsIteration'] = bootsIteration
    bootsDF['expectedValue'] = bootsDF.apply(lambda x: expectedValueBoots_d[x['bootsIteration']], axis=1)
    end = time.time()
    print("   Execution time: {:.2f}s".format(round(end - start,5)))
    mean_expected_value_bootstrap = bootsDF.groupby('bootsIteration').agg({'expectedValue':np.mean}).mean()[0]
    return bootsDF, dict(modelBoots_d=modelBoots_d, 
                explainerBoots_d=explainerBoots_d, 
                expectedValueBoots_d=expectedValueBoots_d,
                mean_expected_value_bootstrap= mean_expected_value_bootstrap)

# For logistic regression
def fill_with_median(arr, train_arr):
    """ fills nan in arr with median of train_arr"""
    col_median = np.nanmedian(train_arr, axis=0)
    inds = np.where(np.isnan(arr))
    # print(inds)
    arr[inds] = np.take(col_median, inds[1])
    return arr

def parallel_bootstrap_shap_analysis_LR(df, target, params, iterations=np.round(1000*np.e), method='sample_with_replacement', train_size=0.8, stratification_factor=None):
    """ By default the bootstrap analysis does sample with replacement. 
    if method == 'train_test_split' each iteration of bootstap will contain train_size portion of the data for training."""
    start = time.time()
    nSamples = df.shape[0] 
    sampleIdxs = range(nSamples)
    mdlFeatures = [feat for feat in df.columns if feat != target]
    print("Running bootstrap analysis using", method)
    tasks=range(iterations)
    X = df[mdlFeatures].values
    y = df[target].values
    @globalize
    def parallel_bootstrap_SHAP_analysis_fcn(seed_value):
        if method == 'sample_with_replacement':
            seed(seed_value)
            train_index = choices(sampleIdxs,k=nSamples-1)# Analogous to LOO approach
            val_index = np.array(list((set(sampleIdxs).difference(train_index)))) # This ensures there's no duplicated valIdxs
        elif method == 'train_test_split':
            if stratification_factor == None:
                sss = ShuffleSplit(n_splits=1, train_size=train_size, random_state=seed_value)
                train_index, val_index = sss.split(X=df[mdlFeatures].values).__next__()
            else:
                sss = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=seed_value)
                train_index, val_index = sss.split(X=df[mdlFeatures].values, y=df[stratification_factor]).__next__()

        X_tr, X_val = np.take(X, train_index,axis=0), np.take(X, val_index,axis=0)
        y_tr, y_val = np.take(y, train_index,axis=0), np.take(y, val_index,axis=0)
        X_tr, X_val = fill_with_median(X_tr, X_tr), fill_with_median(X_val, X_tr)
        model = LogisticRegression(**params).fit(X_tr,y_tr)
        # if xgb_hyperparams['objective'] == 'reg:squarederror':
        # 	model = xgb.XGBRegressor(**xgb_hyperparams, n_jobs=1, n_estimators=100, random_state=seed_value)
        # elif xgb_hyperparams['objective'] == 'binary:logistic':
        # 	model = xgb.XGBClassifier(**xgb_hyperparams, n_jobs=1, n_estimators=100, use_label_encoder=False, random_state=seed_value)
        # else:
        # 	print('Unsupported model type')
        # 	return
        # model.fit(X_tr,y_tr)
        explainer = shap.LinearExplainer(model, X_tr) #logit
        shap_values_curr = explainer.shap_values(X_val)
        # if explainer_type == 'prob':
        # 	explainer = shap.TreeExplainer(model,data=X, feature_dependence="interventional", 
        # 								   model_output="probability")
        # 	shap_values_curr = explainer.shap_values(X_val)
        # elif explainer_type == 'raw': #aka logit when binary classification
        # 	explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        # 	shap_values_curr = explainer.shap_values(X_val)
        # elif explainer_type == 'interventional': 
        # 	explainer = shap.TreeExplainer(model,data=X, model_output="interventional")
        # 	shap_values_curr = explainer.shap_values(X_val)

        append_arr= np.concatenate((np.array([explainer.expected_value]*val_index.shape[0]), np.array([seed_value]*val_index.shape[0]), val_index)).reshape(-1,len(val_index)).T
        output_arr = np.concatenate((shap_values_curr, append_arr), axis=1)
        return output_arr
    
    pool = Pool(processes=cpu_count())
    r = list(tqdm.notebook.tqdm(pool.imap(parallel_bootstrap_SHAP_analysis_fcn, tasks), total=len(tasks)))
    
    pool.close()
    colnames = copy.deepcopy(mdlFeatures)
    colnames.extend(['expectedValue', 'bootsIteration', 'index'])
    bootsDF = pd.DataFrame(np.concatenate(tuple(r)), columns=colnames).astype(dict(zip(['expectedValue', 'bootsIteration', 'index'], [float,int,int])))
    end = time.time()
    print("   Execution time: {:.2f}s".format(round(end - start,5)))
    return bootsDF

def bootstrap_SHAP_analysis_LR(df, target, model, iterations=np.round(1000*np.e), method='sample_with_replacement', train_size=0.8, stratification_factor=None):
    """
    This performs a bootstrap SHAP analysis for a LogisticRegression model
    """
    start = time.time()
    nSamples = df.shape[0] 
    sampleIdxs = range(nSamples)
    mdlFeatures = [feat for feat in df.columns if feat != target]
    save_d = True
    if save_d: 
        modelBoots_d = {}
        explainerBoots_d = {}
        expectedValueBoots_d = {}
        idxsBoots_d = {}
    for i in tqdm.tqdm_notebook(range(iterations)): #2718 : This makes it such that there would be ~1000 bootstrapped models for available each sample
        if method == 'sample_with_replacement':
            seed(i)
            train_index = choices(sampleIdxs,k=nSamples-1)# Analogous to LOO approach
            val_index = np.array(list((set(sampleIdxs).difference(train_index)))) # This ensures there's no duplicated valIdxs
        elif method == 'train_test_split':
            if stratification_factor == None:
                sss = ShuffleSplit(n_splits=1, train_size=train_size, random_state=i)
                train_index, val_index = sss.split(X=df[mdlFeatures].values).__next__()
            else:
                sss = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=i)
                train_index, val_index = sss.split(X=df[mdlFeatures].values, y=df[stratification_factor]).__next__()

        # train_index = choices(sampleIdxs,k=nSamples-1)# Analogous to LOO approach
        # val_index = list(set(sampleIdxs).difference(train_index)) # This ensures there's no duplicated valIdxs

        X_tr, X_val = df[mdlFeatures].iloc[train_index], df[mdlFeatures].iloc[val_index]
        y_tr, y_val = df[target].iloc[train_index], df[target].iloc[val_index]
        X_tr, X_val = X_tr.fillna(X_tr.median()), X_val.fillna(X_tr.median())

        model.fit(X_tr,y_tr)

        # currPredictions_val = model.predict_proba(X_val)[:,1]
        # currPredictions_train = model.predict_proba(X_tr)[:,1]
        explainer = shap.LinearExplainer(model, X_tr) #logit
        shap_values_curr = explainer.shap_values(X_val)

        # if explainer_type == 'prob':
        # 	explainer = shap.TreeExplainer(model,data=df.drop(columns=target),
        # 									   feature_dependence="interventional", model_output="probability")
        # 	shap_values_curr = explainer.shap_values(df[mdlFeatures].iloc[val_index])
        # elif explainer_type == 'raw':
        # 	explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        # 	shap_values_curr = explainer.shap_values(df[mdlFeatures].iloc[val_index])

        # elif explainer_type == 'interventional': 
        # 	explainer = shap.TreeExplainer(xgb_model,data=df.drop(columns=target),
        # 									   feature_dependence="interventional", model_output="raw")
        # 	shap_values_curr = explainer.shap_values(df[mdlFeatures].iloc[val_index])

        if i == 0: 
            shap_values = shap_values_curr
            val_indecies=val_index
            bootsIteration = [i]*len(val_index)
        else:    
            shap_values = np.concatenate([shap_values, shap_values_curr])
            # val_indecies.extend(val_index) 
            val_indecies=np.concatenate([val_indecies, val_index])
            bootsIteration.extend([i]*len(val_index))
            
        modelBoots_d[i] = model
        explainerBoots_d[i] =explainer
        expectedValueBoots_d[i] = explainer.expected_value
        idxsBoots_d[i] = val_index

    bootsDF = pd.DataFrame(shap_values, columns = mdlFeatures)
    bootsDF['index'] = val_indecies
    bootsDF['bootsIteration'] = bootsIteration
    bootsDF['expectedValue'] = bootsDF.apply(lambda x: expectedValueBoots_d[x['bootsIteration']], axis=1)
    end = time.time()
    print("   Execution time: {:.2f}s".format(round(end - start,5)))
    mean_expected_value_bootstrap = bootsDF.groupby('bootsIteration').agg({'expectedValue':np.mean}).mean()[0]
    return bootsDF, dict(modelBoots_d=modelBoots_d, 
                explainerBoots_d=explainerBoots_d, 
                expectedValueBoots_d=expectedValueBoots_d,
                mean_expected_value_bootstrap= mean_expected_value_bootstrap)

# General fcns for bootstrap analysis

def get_combinedFeatName(shap_features):
    """ Helper function that generates the combinedFeat_name which is utilized in various functions"""
    tmp = sorted([feat for feat in shap_features if feat != 'expectedValue'])
    if 'expectedValue' in shap_features:
        tmp.append('expectedValue')
    shap_features=tmp
    combinedFeat_name= ", ".join(shap_features)
    combinedFeat_name = f'shap_sum({combinedFeat_name})'
    return combinedFeat_name

def getCombined_BootsCI_DF(bootsDF, df, x_feature, shap_features: 'list of features', ci=.95):
    """ This function extracts the bootstrapped confidence intervals for a combined list of features to generate a feature dependence plot. 
    x_feature : the feature to plot on the x-axis
    shap_features : the SHAP values you want to consider
    """
    combinedFeat_name = get_combinedFeatName(shap_features)
    
    tmpBootsDF = pd.concat([bootsDF['index'],bootsDF[shap_features].sum(axis=1)],axis=1)
    tmpBootsDF.columns=['index', combinedFeat_name]
    lwrS = tmpBootsDF.groupby('index')[combinedFeat_name].quantile(.5-ci/2).reset_index(drop=True)
    lwrS.name = combinedFeat_name+"_lwr"
    
    meanS= tmpBootsDF.groupby('index')[combinedFeat_name].mean().reset_index(drop=True)
    meanS.name = combinedFeat_name+"_mean"
    
    medianS= tmpBootsDF.groupby('index')[combinedFeat_name].median().reset_index(drop=True)
    medianS.name = combinedFeat_name+"_median"
    
    uprS = tmpBootsDF.groupby('index')[combinedFeat_name].quantile(.5+ci/2).reset_index(drop=True)
    uprS.name = combinedFeat_name+"_upr"
    
    ciDF = pd.concat([lwrS,meanS, medianS, uprS], axis=1)
    ciDF = pd.concat([df[x_feature], ciDF], axis=1)
    return ciDF

def plot_bootstrapped_feature_dependence(bootsDF, df, x_feature, shap_features=[], units='', color_by=None,level_type='categorical', bins = None, nQuantiles = 10, nQuantiles2 = None, yaxis_label='∆ prediction', alpha0 = .5, ms0=7, ci = .95, figsize = (16,6), save_fig=False, outputs_dir='./', return_summary_table = False, return_fig=False, ylims=None, xlims=None, legend_loc=None, categorical_mapping=None, fig_labels=False):

    """ 
    This function flexibly plots bootstrapped feature dependence plots:

    plot_feature_dependence(bootsDF=bootsDF, df=df, x_feature='CTR1', shap_feature=['CTR1', 'MAINT_TRT'], color_by='MAINT_TRT',level_type='categorical')

    color_by : must be a feature in df; default is None, which colors by quantiles
    level_type : defines how to color the feature 'sequential' or 'categorical'; default: 'categorical'
                 level_type can also be a color map that gets passed into: sns.palettes.color_palette(palette=level_type, n_colors=nLevels)
                level_type can be: 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 
                'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 
                'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 
                'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 
                'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 
                'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 
                'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 
                'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 
                'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 
                'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r', 
                'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 
                'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 
                'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 
                'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 
                'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 
                'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 
                'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'twilight', 
                'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 
                'winter_r'
    x_feature: The feature value to be plotted on the x-axis
    shap_features: defines what to shap values to combine on the y-axis (e.g. ['Cmin', 'Cmax', 'AUC']), 
        by default, it is simply the SHAP values of x_feature
    
    y-axis: Label for y-axis
    nQuantiles2 : quantiles for the summary plot; Default: None defaults to nQuantiles
    Note: if the y-axis starts with 'Adjusted', the CIs will also incorporate variability of expected value

    """
    assert type(shap_features) == list
    bins_orig = copy.deepcopy(bins)
    plt.style.use('seaborn-talk')
    append_to_y_axis=False
    
    if len(shap_features) == 0:
        shap_features = [x_feature]
    else:
        append_to_y_axis = True
    if yaxis_label.lower().startswith('adjusted'):
        shap_features.append('expectedValue')
        
    combinedFeat_name = get_combinedFeatName(shap_features)
    if append_to_y_axis:
        yaxis_label=yaxis_label+'\n' + combinedFeat_name
        
    feature_bootsDF = getCombined_BootsCI_DF(bootsDF=bootsDF, df=df, x_feature=x_feature, shap_features=shap_features, ci=.95).rename(columns = {combinedFeat_name+"_lwr": 'CI_lwr', 
                                                                                                                          combinedFeat_name+"_median": 'median_SHAP',
                                                                                                                          combinedFeat_name+"_mean": 'mean_SHAP',
                                                                                            combinedFeat_name+"_upr": 'CI_upr'})    
    if (color_by==None) | (color_by==x_feature): # then color by x_feature
        color_by = x_feature+"_binned"
    lwr = feature_bootsDF.apply(lambda x: x.mean_SHAP - x.CI_lwr, axis=1)
    upr = feature_bootsDF.apply(lambda x: x.CI_upr - x.mean_SHAP, axis=1)
    if len(list(feature_bootsDF[x_feature].unique())) <=10:
        classes = [str(v) for v in sorted(list(feature_bootsDF[x_feature].unique()))]
        if color_by == x_feature+ "_binned":
            classes1 = classes
        feature_bootsDF[x_feature+"_binned"] = feature_bootsDF[x_feature].astype(str)
    else:
        classes = None
        if (type(bins) == list) & (color_by==x_feature+"_binned"):
            feature_bootsDF[x_feature+"_binned"] = pd.cut(feature_bootsDF[x_feature], bins = bins, include_lowest=True, duplicates = 'drop')
        elif ((feature_bootsDF[x_feature] == feature_bootsDF[x_feature].min()).sum()/feature_bootsDF[x_feature].shape[0]) > .025: # If greater than 2.5% of the data is the lowest value: Make a bin for the LoQ/Zero value
            lowest_binEdge = (feature_bootsDF[x_feature].min() == feature_bootsDF[x_feature]).sum()/feature_bootsDF[x_feature].notna().sum()
            bins = np.append(np.array(feature_bootsDF[x_feature].min()), np.linspace(lowest_binEdge,1, nQuantiles)) 
            feature_bootsDF[x_feature+"_binned"] = pd.cut(feature_bootsDF[x_feature], bins = [feature_bootsDF[x_feature].quantile(q) for q in bins], include_lowest=True, duplicates = 'drop')
            if type(nQuantiles2) != type(None): 
                bins2 = np.append(np.array([0]), np.linspace(lowest_binEdge,1, nQuantiles)) 
                feature_bootsDF[x_feature+"_binned2"] = pd.cut(feature_bootsDF[x_feature], bins = [feature_bootsDF[x_feature].quantile(q) for q in bins2], include_lowest=True, duplicates = 'drop')
                
        else:
            bins=[feature_bootsDF[x_feature].quantile(q) for q in np.linspace(0,1, nQuantiles+1)]
            bins[0] = bins[0]-.1
            feature_bootsDF[x_feature+"_binned"] = pd.cut(feature_bootsDF[x_feature], bins = bins, include_lowest=True, duplicates = 'drop')
            if type(nQuantiles2) != type(None):
                bins2=[feature_bootsDF[x_feature].quantile(q) for q in np.linspace(0,1, nQuantiles2+1)]
                bins2[0] = bins2[0]-.1
                feature_bootsDF[x_feature+"_binned2"] = pd.cut(feature_bootsDF[x_feature], bins = bins2, include_lowest=True, duplicates = 'drop')

    if (color_by==x_feature+"_binned") & (type(classes)==type(None)): 
        classes = feature_bootsDF[x_feature+"_binned"].cat.categories.astype(str).to_list() # sorted list
        classes.append('nan')
        classes1 = classes
    elif color_by!=x_feature+"_binned": 
        if (type(bins_orig) == list):
            feature_bootsDF[color_by] = pd.cut(df[color_by], bins = bins_orig, include_lowest=True, duplicates = 'drop')
            classes = feature_bootsDF[color_by].cat.categories.astype(str).to_list()
        elif len(df[df[color_by].notna()][color_by].unique()) > 5:
            feature_bootsDF[color_by] = pd.cut(df[color_by], bins = [df[color_by].quantile(q) for q in np.linspace(0,1, nQuantiles+1)], include_lowest=True, duplicates = 'drop')
            classes = feature_bootsDF[color_by].cat.categories.astype(str).to_list()
        else:
            feature_bootsDF[color_by]=df[color_by].astype(str)
            classes = [str(cl) for cl in sorted(df[color_by].unique())] # sorted list
        
        classes1 = feature_bootsDF[x_feature+"_binned"].cat.categories.astype(str).to_list() # sorted list

        feature_bootsDF[x_feature+"_binned"]=feature_bootsDF[x_feature+"_binned"].astype(str)
    
    nLevels = len(classes)
    feature_bootsDF[color_by]=feature_bootsDF[color_by].astype(str)

    if level_type == 'sequential':
        colors0=sns.palettes.color_palette(palette='RdYlBu_r', n_colors=nLevels)
    elif level_type == 'categorical':
        colors0=sns.palettes.color_palette(palette='tab10', n_colors=nLevels)
    else:
        colors0=sns.palettes.color_palette(palette=level_type, n_colors=nLevels)
        
    colors0.append((.4, .4, .4))
    
    levels_d = dict(enumerate(classes))
    levels_d = defaultdict(lambda : 11, {v:k for k,v in levels_d.items()})
    feature_bootsDF['color0']= feature_bootsDF[color_by].astype(str).apply(lambda x: colors0[levels_d[x]] if x!='nan' else (.4, .4, .4))

    fig, ax = plt.subplots(1,2, figsize=figsize)
    for cl in classes:
        if cl == 'nan':
            continue
        currDF = feature_bootsDF[feature_bootsDF[color_by] == cl]
        i = currDF.index
        if len(i)==0:
            continue
        ax[0].errorbar(currDF[x_feature].loc[i], 
                       feature_bootsDF.mean_SHAP.loc[i], 
                       np.array([lwr.loc[i],upr.loc[i]]), 
                       fmt='o', 
                       color=feature_bootsDF['color0'].loc[i[0]],
                     ecolor='lightgray', elinewidth=2, capsize=2, 
                       label = feature_bootsDF[color_by].loc[i[0]], 
                       markeredgecolor='k', markeredgewidth=.5, alpha=alpha0, markersize=ms0)


    ax[0].set_xlabel(x_feature + " value" + units)
    ax[0].set_ylabel(yaxis_label)
    ax[0].set_title(yaxis_label + " vs {}\nw/ bootstrapped {:.0f}% CI".format(x_feature, ci*100))
    if color_by == x_feature+"_binned":
        ax[0].legend(title=color_by, framealpha=.2, loc=legend_loc)
        plt.setp(ax[0].get_legend().get_title(), fontsize=14)
    
    #### Binned feature dependence plot
    if type(nQuantiles2) != type(None):
        feature_bootsDF[x_feature+"_binned"] = feature_bootsDF[x_feature+"_binned2"]


    if color_by == x_feature+"_binned":
        binnedBootsDF = (pd.concat([bootsDF[['index', 'bootsIteration']],bootsDF[shap_features].sum(axis=1)],axis=1).
                    merge(feature_bootsDF[[x_feature, x_feature+"_binned"]].reset_index(), on='index').
                    rename(columns={0:'combined_SHAP'}))
        feature_bootsDF['color1'] = feature_bootsDF['color0']
    else:
        binnedBootsDF = (pd.concat([bootsDF[['index', 'bootsIteration']],bootsDF[shap_features].sum(axis=1)],axis=1).
                    merge(feature_bootsDF[[x_feature, x_feature+"_binned", color_by]].reset_index(), on='index').
                    rename(columns={0:'combined_SHAP'}))

        feature_bootsDF['color1'] = feature_bootsDF['color0'] # 'gray'

    if color_by == x_feature+"_binned":
        for cl_x in classes1: 
            if cl_x == 'nan': 
                pass
            else:
                currQuantileDF = binnedBootsDF[binnedBootsDF[x_feature+"_binned"] == cl_x].pivot_table(columns = 'bootsIteration', index=['index'], values='combined_SHAP')
                mean_cl = currQuantileDF.mean().mean()
                try: 
                    ax[1].errorbar(x=feature_bootsDF[feature_bootsDF[x_feature+"_binned"]== cl_x][x_feature].median(), 
                                y=mean_cl, 
                                yerr=np.array([mean_cl-currQuantileDF.mean().quantile(q=.5-ci/2),
                                                currQuantileDF.mean().quantile(q=.5+ci/2)-mean_cl]).reshape(-1,1),
                                fmt='o', color=feature_bootsDF[feature_bootsDF[x_feature+"_binned"]== cl_x].color1.iloc[0],
                                ecolor='gray', elinewidth=2, capsize=2, alpha=.8, markeredgecolor='k', markeredgewidth=.7)
                except IndexError:
                    raise IndexError("Rounding yielded a bin with no data points. Try decreasing nQuantiles.")
    elif color_by != x_feature+"_binned":
        for cl_x in classes1: 
            if cl_x == 'nan': 
                pass
            else:
                currQuantileDF = binnedBootsDF[binnedBootsDF[x_feature+"_binned"] == cl_x].pivot_table(columns = 'bootsIteration', index=['index'], values='combined_SHAP')
                currQuantileDF = currQuantileDF.merge(feature_bootsDF[[color_by]].reset_index(), on='index')
                for cl in classes:
                    currQuantileDF2 = currQuantileDF[currQuantileDF[color_by]==cl].drop(columns = [color_by, 'index'])
                    mean_cl = currQuantileDF2.mean().mean()
                    try: 
                        ax[1].errorbar(x=feature_bootsDF[(feature_bootsDF[x_feature+"_binned"]== cl_x) & (feature_bootsDF[color_by]==cl)][x_feature].median(), 
                                    y=mean_cl, 
                                    yerr=np.array([mean_cl-currQuantileDF2.mean().quantile(q=.5-ci/2),
                                                    currQuantileDF2.mean().quantile(q=.5+ci/2)-mean_cl]).reshape(-1,1),
                                    fmt='o', color=feature_bootsDF[(feature_bootsDF[color_by]== cl)].color1.iloc[0],
                                    ecolor='gray', elinewidth=2, capsize=2, alpha=.8, markeredgecolor='k', markeredgewidth=.7)

                    except IndexError:
                        raise IndexError("Rounding yielded a bin with no data points. Try decreasing nQuantiles.")

                

    ax[1].set_xlabel(x_feature + " value" + units)
    ax[1].set_ylabel(yaxis_label)
    ax[1].set_title(yaxis_label+" vs {}\nw/ bootstrapped {:.0f}% CI (binned)".format(x_feature, ci*100))
    my_round_lwr = lambda x: np.floor(x*20)/20 # round down to nearest .05
    my_round_upr = lambda x: np.ceil(x*20)/20 # round up to nearest .05

    ylim_lwr = bootsDF.loc[:, [feat for feat in bootsDF.columns if feat not in ['index', 'bootsIteration',
           'expectedValue']]].min().min()
    ylim_upr = bootsDF.loc[:, [feat for feat in bootsDF.columns if feat not in ['index', 'bootsIteration',
           'expectedValue']]].max().max()

    if yaxis_label.lower().startswith('adjusted'): 
        mean_expected_value_bootstrap = bootsDF.groupby('bootsIteration').agg({'expectedValue':np.mean}).mean()[0]
        ylim_lwr = my_round_lwr(ylim_lwr+mean_expected_value_bootstrap)
        ylim_upr = my_round_upr(ylim_upr+mean_expected_value_bootstrap)
    else: 
        ylim_lwr = my_round_lwr(ylim_lwr)
        ylim_upr = my_round_upr(ylim_upr)
    if (ylim_upr < 1) & (ylim_lwr >0):
        ax[0].set_yticks(np.arange(ylim_lwr, ylim_upr, .05))
        ax[1].set_yticks(np.arange(ylim_lwr, ylim_upr, .05))
    ax[0].grid(True, alpha=.2)
    ax[1].grid(True, alpha=.2)
    ax[0].set_facecolor((.95, .95, .95))
    ax[1].set_facecolor((.98, .98, .98))
    if type(ylims)==type(None):
        ax[0].set_ylim(ylim_lwr, ylim_upr)
        ax[1].set_ylim(ax[0].get_ylim())
    else:
        ax[0].set_ylim(ylims)
        ax[1].set_ylim(ylims)


    if ax[0].get_xlim()[1] < 1.33*ax[1].get_xlim()[1]:
        ax[1].set_xlim(ax[0].get_xlim())
    if type(xlims)!=type(None):
        print("Setting xlim manually for left subplot")
        ax[0].set_xlim(xlims)
    if yaxis_label.lower().startswith('adjusted'):
        ax[0].plot(ax[0].get_xlim(), [mean_expected_value_bootstrap, mean_expected_value_bootstrap], '--k')
        ax[1].plot(ax[1].get_xlim(), [mean_expected_value_bootstrap, mean_expected_value_bootstrap], '--k')
    else:
        ax[0].plot(ax[0].get_xlim(), [0, 0], '--k')
        ax[1].plot(ax[1].get_xlim(), [0, 0], '--k')

    if type(categorical_mapping) != type(None):
        ax[0].set_xticks(list(categorical_mapping[x_feature].keys()))
        ax[1].set_xticks(list(categorical_mapping[x_feature].keys()))
        ax[0].set_xticklabels(categorical_mapping[x_feature].values(), rotation=90)
        ax[1].set_xticklabels(categorical_mapping[x_feature].values(), rotation=90)
    plt.tight_layout()
    if save_fig:
        fileName = outputs_dir + "Bootstrapped {} adj prob plot_{:.0f} quantiles_{}_by{}_colored_by{}.png".format(combinedFeat_name, nQuantiles, yaxis_label, x_feature, color_by)
        print( "Saving as " + fileName) 
        plt.savefig(fileName, bbox_inches='tight', dpi=300)

    if return_summary_table: 
        summaryTable = pd.concat([(binnedBootsDF
         .pivot_table(columns = 'bootsIteration', index=['index', x_feature+"_binned"], values='combined_SHAP')
        .groupby(x_feature+"_binned").mean().T.quantile(q=.5-ci/2)),
        (binnedBootsDF
            .pivot_table(columns = 'bootsIteration', index=['index', x_feature+"_binned"], values='combined_SHAP')
            .groupby(x_feature+"_binned").mean().T.mean()),
        (binnedBootsDF
            .pivot_table(columns = 'bootsIteration', index=['index', x_feature+"_binned"], values='combined_SHAP')
            .groupby(x_feature+"_binned").mean().T.quantile(q=.5+ci/2))],axis=1).rename(columns={0:'mean'})
        summaryTable['∆ prediction'] = summaryTable.apply(lambda x: "{:.4f} ({:.4f},{:.4f})".format(x['mean'],x[.5-ci/2],x[.5+ci/2]), axis=1)
        # Sort table and remove 'nan group'
        tmpDF = summaryTable.reset_index()
        import ast 
        def interval_type(s):
            """Parse interval string to Interval"""
            table = str.maketrans({'[': '(', ']': ')'})
            left_closed = s.startswith('[')
            right_closed = s.endswith(']')
            left, right = ast.literal_eval(s.translate(table))
            t = 'neither'
            if left_closed and right_closed:
                t = 'both'
            elif left_closed:
                t = 'left'
            elif right_closed:
                t = 'right'
            return pd.Interval(left, right, closed=t)
        tmpDF = tmpDF[tmpDF[x_feature+"_binned"] != 'nan'].reset_index(drop=True)
        tmpDF[x_feature+"_binned"]= tmpDF[x_feature+"_binned"].apply(interval_type)
        summaryTable = tmpDF.sort_values(x_feature+"_binned")
        if return_fig == True: 
            summaryTable[x_feature+"_binned"]=summaryTable[x_feature+"_binned"].astype(str)
            return fig, summaryTable[[x_feature+"_binned", '∆ prediction']]
        else: 
            return ax, summaryTable[[x_feature+"_binned", '∆ prediction']]
    if fig_labels:
        add_annotations(fig)
    if return_fig == True: 
        return fig
    else: 
        return ax

def plot_bootstrapped_feature_dependence_lowess(bootsDF, df, x_feature, shap_features=[], yaxis_label='∆ prediction', figsize=(3.5,5),
color='k', conf=.05, show_points=True, fill_alpha=.33, marker_alpha=.1, marker_size=10,grid_alpha=.5, label=None, ax=None, ylims=None, xlims=None, save_fig=False, outputs_dir='./'):
    
    append_to_y_axis = False
    if len(shap_features) == 0:
        shap_features = [x_feature]
    else:
        append_to_y_axis = True
    if yaxis_label.lower().startswith('adjusted'):
        shap_features.append('expectedValue')
        
    combinedFeat_name = get_combinedFeatName(shap_features)
    if append_to_y_axis:
        yaxis_label=yaxis_label+'\n' + combinedFeat_name
        
    feature_bootsDF = getCombined_BootsCI_DF(bootsDF=bootsDF, df=df, x_feature=x_feature, shap_features=shap_features, ci=.95).rename(columns = {combinedFeat_name+"_lwr": 'CI_lwr', 
                                                                                                                          combinedFeat_name+"_median": 'median_SHAP',
                                                                                                                          combinedFeat_name+"_mean": 'mean_SHAP',
                                                                                            combinedFeat_name+"_upr": 'CI_upr'})
    x=feature_bootsDF[x_feature].values
    y=feature_bootsDF['mean_SHAP'].values

    ax=plot_lowess(x=x, y=y, color=color, conf=conf, ax=ax, label=label, show_points=show_points, 
    fill_alpha=fill_alpha, marker_alpha=marker_alpha, marker_size=marker_size,grid_alpha=grid_alpha, figsize=figsize)

    plt.xlabel(x_feature)
    plt.ylabel(yaxis_label)
    if type(ylims) != type(None):
        plt.ylim(ylims)
    if type(xlims) != type(None):
        plt.ylim(xlims)
    if save_fig:
        fileName = outputs_dir + "Lowess bootstrapped {} dependence plot {} by {} .png".format(combinedFeat_name, yaxis_label, x_feature)
        print( "Saving as " + fileName) 
        plt.savefig(fileName, bbox_inches='tight', dpi=300)
    return ax

#These functions are used to summarize binomial target variable in bootstrap_feature_summary_table fcn
def binomVec_yerr(vec):
    from statsmodels.stats.proportion import proportion_confint
    lwr, upr = np.abs(proportion_confint(vec.sum(), len(vec), method='wilson')-np.mean(vec))
    point_estimate = np.mean(vec)
    return np.round(point_estimate,4), np.round(lwr,4), np.round(upr,4)

def binom_CI_str(series):
    point_estimate, lwr, upr = binomVec_yerr(series)
    return "{:.3f} ({:.3f}, {:.3f})".format(point_estimate, np.round(point_estimate-lwr,3), np.round(point_estimate+upr,3))
def mean_std_str(vec):
    return "{:.3f}±{:.3f}".format(np.round(np.nanmean(vec),4), np.round(np.nanstd(4)))

def bootstrap_feature_summary_table(bootsDF, df, feature, shap_features=[], bins=None, nQuantiles=10, metric='∆ prediction', ci = .95, categorical_mapping=None, drop_nan=True, target=None):
    """
    if metric.starstwith('adjusted): shows the adjusted prediction rather than ∆ prediction
    target (str) : if supplied the empirical target will be summarized with 95% CI for binomial targets and std for continuous variables
    """
    assert type(shap_features) == list
    lowest_bin_is_min=False
    if len(shap_features) == 0:
        shap_features = [feature]
    else:
        pass
    if metric.lower().startswith('adjusted'):
        shap_features.append('expectedValue')
    combinedFeat_name = get_combinedFeatName(shap_features)

    feature_bootsDF = getCombined_BootsCI_DF(bootsDF=bootsDF, df=df, x_feature=feature, shap_features=shap_features, ci=.95).rename(columns = {combinedFeat_name+"_lwr": 'CI_lwr', 
                                                                                                                          combinedFeat_name+"_median": 'median_SHAP',
                                                                                                                          combinedFeat_name+"_mean": 'mean_SHAP',
                                                                                            combinedFeat_name+"_upr": 'CI_upr'})
    
    if type(target) != type(None):
        target_is_binary=False
        feature_bootsDF[target] = df[target]
        if set(df[target].unique())=={0,1}: 
            target_is_binary=True
        
    lwr = feature_bootsDF.apply(lambda x: x.mean_SHAP - x.CI_lwr, axis=1)
    upr = feature_bootsDF.apply(lambda x: x.CI_upr - x.mean_SHAP, axis=1)
    if type(bins) == str:
        if bins=='binary':
            feature_bootsDF[feature+"_binned"] = feature_bootsDF[feature]
            classes = [str(0), str(1)]
        elif bins=='categorical':
            feature_bootsDF[feature+"_binned"] = feature_bootsDF[feature].replace(categorical_mapping[feature])
            classes = list(categorical_mapping[feature].values())
    else:
        if len(list(feature_bootsDF[feature].unique())) <7:
            classes = [str(v) for v in sorted(list(feature_bootsDF[feature].unique()))]
            feature_bootsDF[feature+"_binned"] = feature_bootsDF[feature].astype(str)
        else:
            if type(bins) == list:
                feature_bootsDF[feature+"_binned"] = pd.cut(feature_bootsDF[feature], bins = bins, include_lowest=True, duplicates = 'drop')
            elif ((feature_bootsDF[feature] == feature_bootsDF[feature].min()).sum()/feature_bootsDF[feature].shape[0]) > .025: # If greater than 2.5% of the data is the lowest value: Make a bin for the LoQ/Zero value
                lowest_bin_is_min = True
                lowest_binEdge = (feature_bootsDF[feature].min() == feature_bootsDF[feature]).sum()/feature_bootsDF[feature].notna().sum()
                bins = np.append(np.array([feature_bootsDF[feature].min()]), np.linspace(lowest_binEdge,1, nQuantiles)) 
                feature_bootsDF[feature+"_binned"] = pd.cut(feature_bootsDF[feature], bins = [feature_bootsDF[feature].quantile(q) for q in bins], include_lowest=True, duplicates = 'drop')
            else:
                bins=[feature_bootsDF[feature].quantile(q) for q in np.linspace(0,1, nQuantiles+1)]
                bins[0] = bins[0]-.1
                feature_bootsDF[feature+"_binned"] = pd.cut(feature_bootsDF[feature], bins = bins, include_lowest=True, duplicates = 'drop')
            classes = feature_bootsDF[feature+"_binned"].cat.categories.astype(str).to_list() # sorted list
    classes.append('nan')
    feature_bootsDF[feature+"_binned"]=feature_bootsDF[feature+"_binned"].astype(str)
    # nLevels = len(classes)
    #### Binned feature dependence plot
    binnedBootsDF = (pd.concat([bootsDF[['index', 'bootsIteration']],bootsDF[shap_features].sum(axis=1)],axis=1).
                  merge(feature_bootsDF[[feature, feature+"_binned"]].reset_index(), on='index').
                  rename(columns={0:'combined_SHAP'}))

    summaryTable = pd.concat([(binnedBootsDF
     .pivot_table(columns = 'bootsIteration', index=['index', feature+"_binned"], values='combined_SHAP')
    .groupby(feature+"_binned").mean().T.quantile(q=np.round(.5-ci/2,3))),
    (binnedBootsDF
        .pivot_table(columns = 'bootsIteration', index=['index', feature+"_binned"], values='combined_SHAP')
        .groupby(feature+"_binned").mean().T.mean()),
    (binnedBootsDF
        .pivot_table(columns = 'bootsIteration', index=['index', feature+"_binned"], values='combined_SHAP')
        .groupby(feature+"_binned").mean().T.quantile(q=np.round(.5+ci/2,3))),
    feature_bootsDF.groupby(feature+"_binned").agg({feature:'count'})],axis=1).rename(columns={0:'mean '+ metric, feature:'N'})
    median_vals = feature_bootsDF.groupby(feature+"_binned").agg({feature:'median'})
    if type(target) != type(None):
        if target_is_binary:
            targets = feature_bootsDF.groupby(feature+"_binned").agg({target:binom_CI_str})
        else: 
            targets = feature_bootsDF.groupby(feature+"_binned").agg({target:mean_std_str})
        mean_targets = feature_bootsDF.groupby(feature+"_binned").mean()[[target]].rename(columns={target: 'mean ' + target})
        targets = pd.concat([targets, mean_targets], axis=1)
    summaryTable=summaryTable.merge(median_vals.rename(columns={feature: 'median feature val'}), on=feature+"_binned")
    summaryTable[metric] = summaryTable.apply(lambda x: "{:.3f} ({:.3f}, {:.3f})".format(x['mean '+ metric],x[np.round(.5-ci/2,3)],x[np.round(.5+ci/2,3)]), axis=1)
    summaryTable[metric + " 95% CI"] = summaryTable.apply(lambda x: "({:.3f}, {:.3f})".format(x[np.round(.5-ci/2,3)],x[np.round(.5+ci/2,3)]), axis=1)
        # Sort table and remove 'nan group'
    tmpDF = summaryTable.reset_index()
    import ast 
    def interval_type(s):
        """Parse interval string to Interval"""
        table = str.maketrans({'[': '(', ']': ')'})
        left_closed = s.startswith('[')
        right_closed = s.endswith(']')
        left, right = ast.literal_eval(s.translate(table))
        t = 'neither'
        if left_closed and right_closed:
            t = 'both'
        elif left_closed:
            t = 'left'
        elif right_closed:
            t = 'right'
        return pd.Interval(left, right, closed=t)
    tmpDF = tmpDF[(tmpDF[feature+"_binned"] != 'nan')].reset_index(drop=True)
    if type(bins)==list:
        tmpDF[feature+"_binned"]= tmpDF[feature+"_binned"].apply(interval_type)
    elif (bins == 'binary'):
        tmpDF[feature] = tmpDF[feature+"_binned"].astype(int)
    elif bins=='categorical':
        tmpDF = tmpDF[tmpDF[feature+"_binned"] != str(-1.0)]

    summaryTable = tmpDF.sort_values('median feature val')
    summaryTable[feature+"_binned"]=summaryTable[feature+"_binned"].astype(str)
    summaryTable['Variable']=feature
    cols2use = ['Variable', feature+"_binned", 'median feature val', metric, 'mean '+metric,np.round(.5-ci/2,3),np.round(.5+ci/2,3), "N"]
    if type(target)!=type(None):
        summaryTable = summaryTable.merge(targets, on = feature+"_binned", how='left')
        cols2use.extend([target, 'mean '+ target])
    summaryTable=summaryTable[cols2use].rename(columns = {feature+"_binned":'Value'})
    if lowest_bin_is_min:
        print(f"Setting lowest bin = {feature_bootsDF[feature].min()}")
        summaryTable.loc[0, "Value"] = feature_bootsDF[feature].min()
    return summaryTable # This elimiates the catch all categorical  

def generate_bootstrap_summaryDF(self, summary_params = None, metric = "∆Prediction", target=None, n_features=10, nQuantiles=4, **kwargs): 
    """
    if metric.starstwith('adjusted): shows the adjusted prediction rather than ∆ prediction
    Generates an overall summary of analysis given manually selected parameters:
    e.g.: 
    summary_params = {'Cminsd': {'nQuantiles': 4}, 
                     'BHBA1C': {'bins':[2.5,5.7,6.4,12.2]},
                     'BGLUC': {'bins': [60,100,120, 130, 350]},
                     'BHDL': {'bins': [0,40,60, 210]},
                     'Region_EUROPE': {},# binary
                     'Race_BLACK_OR_AFRICAN_AMERICAN' : {}
                    }
    """
    if type(summary_params)==type(None):
        print('No bootstrap_summaryDF was found. To customize the summary further run self.generate_bootstrap_summaryDF(summary_params) with defined summary parameters.')
        summary_params = {v: {'nQuantiles': nQuantiles} for v in self.orderedFeatures[:n_features]}
        print('For now, genrating figure using summary_params=', summary_params)

    bootstrap_summary = pd.DataFrame([])
    for key in summary_params:
        print(f"Adding {key}")
        tmpDF= self.bootstrap_feature_summary_table(feature=key, metric=metric, target=target, **summary_params[key], **kwargs)
        # if key == 'Cminsd': 
        # 	tmpDF.loc[0, "Value"] = 0
        bootstrap_summary = pd.concat([bootstrap_summary, tmpDF])
    
    return bootstrap_summary.reset_index(drop=True)

def plot_bootstrap_summary_table(bootstrap_summary, vline=0, v_line_label = '',
                           ax=None,color='k', show=True, title="Impact of covariates on predictions",
                                 x_axis='∆Prediction', label=None, offset=0, figsize=None, xlims=None, show_n=True, return_fig=False):
    """
    Label: This can the name of the analysis
    Ensure the x_axis coincides with the metric in the bootstrap summary table
    """
    if type(xlims) == type(None):
        my_round_lwr = lambda x,r=.1: np.round(np.floor(x/r)*r,4) # round down to nearest r=.1
        my_round_upr = lambda x,r=.1: np.round(np.ceil(x/r)*r,4) # round up to nearest r=.1
        xlims = my_round_lwr(bootstrap_summary[0.025].min()), my_round_upr(bootstrap_summary[0.975].max())
        # xlims = bootstrap_summary["mean "+x_axis].min() - bootstrap_summary["mean "+x_axis].std(), bootstrap_summary["mean "+x_axis].max() + bootstrap_summary["mean "+x_axis].std()

    if type(ax)==type(None):
        fig,ax = plt.subplots(1,1,figsize=(9,bootstrap_summary.shape[0]*.5))
        plot_on_existing = False
    else: 
        plot_on_existing = True

    ax.errorbar(x=bootstrap_summary["mean "+x_axis].values, 
                 y=-bootstrap_summary.index+ offset, 
                xerr=np.array(bootstrap_summary.apply(lambda x: (x["mean "+x_axis]-x[0.025], x[0.975]-x["mean "+x_axis]), axis=1).to_list()).T, 
                 fmt='o', color = color, ecolor=color, alpha=.5, label=label)
    if show_n: 
        plt.yticks(-bootstrap_summary.index, labels=bootstrap_summary.apply(lambda x: x['Variable']+ ":" + str(x['Value']) + f" (n={x['N']})", axis=1) );
    else:	
        plt.yticks(-bootstrap_summary.index, labels=bootstrap_summary.apply(lambda x: x['Variable']+ ":" + str(x['Value']), axis=1));
    if type(vline) != type(None): 
        ax.vlines(x=vline, ymin=-bootstrap_summary.shape[0],
                        ymax=0.5, linestyle='dashed', linewidth = 2, label=v_line_label, color='gray')
    if not plot_on_existing: 
        for i in np.arange(0, -bootstrap_summary.shape[0]-1,-2):
            ax.fill_between(x=[xlims[0], xlims[1]], y1=[i+.5, i+.5], y2=[i-.5, i-.5], color='gray', alpha=.05) #TODO: automatically change this range
        variables = bootstrap_summary['Variable'].unique()
        colors = sns.color_palette('tab10',n_colors=len(variables))
        s = 0;
        i=0
        for variable in variables:
            h=(bootstrap_summary['Variable']==variable).sum()
            ax.fill_between(x=[xlims[0], xlims[1]], y1=[s+.5, s+.5], y2=[s-h+.5, s-h+.5], color=colors[i], alpha=.2)
            i+=1
            s-=h
        ax.set_xlim(xlims) 
        ax.set_xlabel(x_axis)
        ax.set_ylim(top = 0.5, bottom=-bootstrap_summary.shape[0])
        ax.set_title(title)
        if type(figsize) != type(None):
            fig.set_size_inches(figsize)
    if type(label)!=type(None):
        plt.legend(fontsize=10, loc='best', frameon=True, framealpha=.3)
    if return_fig:
        return fig
    if show==True:
        return ax
    else: 
        plt.show()