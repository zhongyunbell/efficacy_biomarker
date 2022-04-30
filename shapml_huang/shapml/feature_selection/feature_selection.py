import time, tqdm, copy
from numpy.core.defchararray import mod
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import time, itertools, copy, tqdm

import hyperopt as hp
from hyperopt.pyll.base import scope
from hyperopt import STATUS_OK, fmin, hp, tpe, SparkTrials
from hyperopt.early_stop import no_progress_loss


import shap
import xgboost as xgb
from boruta import BorutaPy
from ..utils.superclasses import analysis
from ..utils.helpers import LazyProperty
from ..utils.misc import uniform_permutation_sampling

SEED = 314159265
num_boost_round=100
n_folds_CV = 5 # for CV performance evaluation
n_folds_SHAP = 10 # for CV when doing SHAP analysis

currently_supported_models= "'xgb_cox', 'xgb_binary'"
# currently_supported_models= "supports: 'xgb_binary', 'logistic_regression', 'xgb_regression', 'xgb_cox'" TODO: update xgb_hyperparams to hyperparams, score_model, explicitly update Boruta mdl


model_search_spaces = {
    'xgb_binary':{
        'eta':                         hp.loguniform('eta', np.log(0.01), np.log(1)),
        'max_depth':                   scope.int(hp.quniform('max_depth', 2,5,1)),
        'min_child_weight':            hp.loguniform('min_child_weight', np.log(0.01), np.log(10)),
        'reg_alpha':                   hp.loguniform('reg_alpha', np.log(0.2), np.log(10)),
        'reg_lambda':                  hp.loguniform('reg_lambda', np.log(0.001), np.log(10)),
        'subsample':                   hp.uniform('subsample', 0.6, 1),
        "objective": "binary:logistic", 
        'tree_method':"exact",
        'eval_metric':"error"
    },
    'logistic_regression':{
        'C':            hp.loguniform('C', np.log(0.01), np.log(10000)),
        "penalty": "l2",
    }, 
    'xgb_regression':{
        'eta':                         hp.loguniform('eta', np.log(0.01), np.log(1)),
        'max_depth':                   scope.int(hp.quniform('max_depth', 2,5,1)),
        'min_child_weight':            hp.loguniform('min_child_weight', np.log(0.01), np.log(10)),
        'reg_alpha':                   hp.loguniform('reg_alpha', np.log(0.2), np.log(10)),
        'reg_lambda':                  hp.loguniform('reg_lambda', np.log(0.001), np.log(10)),
        'subsample':                   hp.uniform('subsample', 0.6, 1),
        "objective": "reg:squarederror",         
        'tree_method':"exact",
        'eval_metric':"error"
    }, 
    'xgb_cox': {
        'eta':                         hp.loguniform('eta', np.log(0.01), np.log(.2)),
        'max_depth':                   scope.int(hp.quniform('max_depth', 2,5,1)),
        'min_child_weight':            hp.loguniform('min_child_weight', np.log(0.01), np.log(10)),
        'reg_alpha':                   hp.loguniform('reg_alpha', np.log(0.2), np.log(10)),
        'reg_lambda':                  hp.loguniform('reg_lambda', np.log(0.001), np.log(10)),
        'subsample':                   hp.uniform('subsample', 0.6, 1),
        'objective': "survival:cox",         
    }
    }
#_____________
# TODO: update this with proper functionality in this section
from ..binary_classification.shap_based_analysis import xgb_shap as xgb_shap_BC
from ..binary_classification.shap_based_analysis import score_model as score_model_xgb_BC
from ..binary_classification.shap_based_analysis import SHAP_CV as SHAP_CV_xgb_BC

# from ..binary_classification.shap_based_analysis_logistic import logistic_shap
# from ..binary_classification.shap_based_analysis_logistic import score_model as score_model_logistic
# from ..regression.shap_based_analysis import xgb_shap as xgb_shap_regression
# from ..regression.shap_based_analysis import score_model as score_model_xgb_regression

from ..survival_cox.shap_based_analysis import xgb_shap as xgb_shap_cox
from ..survival_cox.shap_based_analysis import score_model as score_model_xgb_cox
from ..survival_cox.shap_based_analysis import SHAP_CV as SHAP_CV_xgb_cox

model_shap_classes = {'xgb_cox': xgb_shap_cox,
'xgb_binary': xgb_shap_BC, 
# 'logistic_regression': logistic_shap,
# 'xgb_regression': xgb_shap_regression,
}

model_scoring_fcns = {'xgb_cox': score_model_xgb_cox,
'xgb_binary': score_model_xgb_BC, 
'logistic_regression': lambda : ValueError("Need to define this scoring function for this model type"),
'xgb_regression': lambda : ValueError("Need to define this scoring function for this model type")}

# This generates shap values by cross-validation for the different models: 
model_shap_cv_fcns = {'xgb_cox': SHAP_CV_xgb_cox, 'xgb_binary': SHAP_CV_xgb_BC}
#_____________


def objective_fcn(params, train_x, train_y, model_type, n_folds=n_folds_CV, **kwargs):
    """Objective function for Hyperparameter Optimization"""   
    scores= model_scoring_fcns[model_type](params, train_x, train_y, n_folds = n_folds, n_reps=1, verbose=False, **kwargs) # return_val_predictions = False
    score = np.mean(scores) 
    loss = 1 - score
    # Dictionary with information for evaluation
    return {'loss': loss,  'status': STATUS_OK}

def optimize_model(objective_fcn, model_type: currently_supported_models, random_state=SEED, verbose=True, max_evals = 25):
    """
    This is the optimization function that given a space (space here) of 
    hyperparameters and a scoring function (score here), finds the best hyperparameters.
    """
    # Optimal grid search parameters: 
    # print(space)	
    # Use the fmin function from Hyperopt to find the best hyperparameters
    rstate = np.random.RandomState(SEED)
    space=model_search_spaces[model_type]
    best = fmin(objective_fcn, space, algo=tpe.suggest, max_evals=max_evals, 
                verbose=verbose,rstate=rstate, 
                early_stop_fn=no_progress_loss(iteration_stop_count=50, percent_increase=0)) #, trials = SparkTrials(parallelism = 4))
    for parameter_name in ['max_depth']:
        best[parameter_name] = int(best[parameter_name])
    return best

# exponential_spacing = lambda stop, n_steps=25: np.unique(np.round(10**np.linspace(0,np.log10(stop), num=n_steps))).astype(int)
def exponential_spacing(stop, n_steps=30):
    if n_steps > stop: 
        print(f"Steps ({n_steps}) > number of features ({stop}). Setting # of steps to {stop}")
        n_steps = stop
    result = [1]
    if n_steps>1:  # just a check to avoid ZeroDivisionError
        ratio = (float(stop)/result[-1]) ** (1.0/(n_steps-len(result)))
    while len(result)<n_steps:
        next_value = result[-1]*ratio
        if next_value - result[-1] >= 1:
            # safe zone. next_value will be a different integer
            result.append(next_value)
        else:
            # problem! same integer. we need to find next_value by artificially incrementing previous value
            result.append(result[-1]+1)
            # recalculate the ratio so that the remaining values will scale correctly
            if len(result)==n_steps:
                break
            else:
                ratio = (float(stop)/result[-1]) ** (1.0/(n_steps-len(result)))
            
    # round, re-adjust to 0 indexing (i.e. minus 1) and return np.uint64 array
    return np.array(list(map(lambda x: round(x), result)), dtype=np.uint64).astype(int)


def SHAP_based_FE(df, target, orderedFeatures_initial, n_steps, model_type: currently_supported_models, hyperparams, max_evals=0, verbose=True):
    """
    FE based on an initial SHAP-based ranking of features
    max_evals : hyperparamter tuning iterations during backwards elimination (set to 0 if wanting to omit this step)
    """
    FE_method = "SHAP_FE"
    all_scores = []
    n_features = list(exponential_spacing(stop=len(orderedFeatures_initial), n_steps=n_steps))[::-1]
    n_features2remove = np.diff(n_features)
    all_features = copy.deepcopy(orderedFeatures_initial)
    features_removed = []
    for f,n_feats in tqdm.notebook.tqdm(enumerate(n_features), total = len(n_features), disable=False, desc='SHAP FE'):
        curr_step = model_shap_classes[model_type](df = pd.concat([df[orderedFeatures_initial[:n_feats]], df[target]],axis=1), target=target, hyperparams=hyperparams, verbose=verbose)
        if max_evals > 0:
            if verbose: 
                print("Step:", f, "n_feats", n_feats, "Tuning current...")
            curr_step.tune_model(verbose=verbose)
        curr_score = curr_step.model_performance(verbose=verbose)
        all_scores.append(curr_score)
        if f!=0:
            features_removed.append(all_features[n_features2remove[f-1]:])
            all_features= all_features[:n_features2remove[f-1]]
    
    mean_score = np.array(all_scores).mean(axis=1)
    optimal_features = orderedFeatures_initial[:n_features[np.where(mean_score==np.max(mean_score))[0][0]]]
    
    return dict(FE_method = FE_method,
                mean_score = mean_score,
                std_score = np.array(all_scores).std(axis=1),
                n_features = n_features, 
                features_removed = features_removed,
                optimal_features = optimal_features)

def SHAP_based_RFECV(df, target, orderedFeatures_initial, model_type: currently_supported_models, hyperparams=None, n_steps=25, max_evals=25,verbose=True):
    """
    FE based on an SHAP-based ranking of features, whereby features are re-ranked at each iteration
    max_evals : hyperparamter tuning iterations during backwards elimination (set to 0 if wanting to omit this step)
    """
    FE_method = "SHAP_RFECV"
    all_scores = []
    n_features = list(exponential_spacing(stop=len(orderedFeatures_initial), n_steps=n_steps))[::-1]
    n_features2remove = np.diff(n_features)
    remaining_features = copy.deepcopy(orderedFeatures_initial)
    features_removed = []
    
    for f,n_feats in tqdm.notebook.tqdm(enumerate(n_features), total = len(n_features), disable=False, desc='SHAP RFECV'):    
        if f!=0:
        #         print("removing {:.2f}: ".format(n_features2remove[f-1]), remaining_features[n_features2remove[f-1]:])
            features_removed.append(remaining_features[n_features2remove[f-1]:])
            remaining_features= remaining_features[:n_features2remove[f-1]]
        #         print("Remaining features: ", remaining_features)
        # Feature ordering is recalculated at each iteration
        curr_step = model_shap_classes[model_type](df = pd.concat([df[remaining_features], df[target]],axis=1),
            target=target, max_evals=max_evals, n_folds_SHAP=10, verbose=verbose, hyperparams=hyperparams)
        if max_evals > 0:
            if verbose: 
                print("Step:", f, "n_feats", n_feats, "Tuning current...")
            curr_step.tune_model(verbose=False) 
        curr_score = curr_step.model_performance(verbose=verbose)
        all_scores.append(curr_score)
        if f!=0:
            # recalculate feature importance after each step starting with the first step
            remaining_features = curr_step.orderedFeatures
    all_features = copy.deepcopy(features_removed)
    all_features.append(remaining_features)
    ordered_features_posthoc = list(itertools.chain(*list(reversed(all_features))))
    mean_score = np.array(all_scores).mean(axis=1)
    optimal_features = ordered_features_posthoc[:n_features[np.where(mean_score==np.max(mean_score))[0][0]]]
    
    return dict(FE_method = FE_method,
                mean_score = np.array(all_scores).mean(axis=1),
                std_score = np.array(all_scores).std(axis=1),
                n_features = n_features, 
                features_removed = features_removed,
               optimal_features = optimal_features)

def boruta_FE(df, target, hyperparams, model_type: currently_supported_models, max_iter=100, n_steps=25, max_evals=0, verbose=True):      
    """
    This function gets the ranked order importance of features using Boruta and then performs backwards feature elimination. 
    max_iter : number of iterations the data set is shuffled to obtain feature rankings
    max_evals : hyperparamter tuning iterations during backwards elimination (set to 0 if wanting to omit this step)
    
    """
    FE_method = "Boruta"
    # mdl = xgb.XGBModel(**hyperparams)
    if model_type.startswith('xgb'):
        mdl = xgb.XGBModel(**hyperparams)
    elif model_type == 'logistic_regression':
        from sklearn.linear_model import LogisticRegression
        mdl = LogisticRegression(**hyperparams)
    else: 
        raise ValueError(f"Need to define model for Boruta-based FE {model_type}.")
    
    boruta = BorutaPy(estimator = mdl, n_estimators = 'auto', max_iter = max_iter)
    X = df.fillna(df.median()).drop(columns=target) # Dataset needs to be imputed for boruta
    y=df[target]
    boruta.fit(np.array(X), np.array(y))
    borutaDF = pd.DataFrame(zip(X.columns,list(boruta.ranking_)), columns = ['Feature', 'Ranking']).sort_values('Ranking')
    # boruta_d = borutaDF.groupby('Ranking')['Feature'].apply(list).to_dict() 
    orderedFeatures_initial = borutaDF.Feature.to_list()
    
    all_scores = []
    n_features = list(exponential_spacing(stop=len(orderedFeatures_initial), n_steps=n_steps))[::-1]
    n_features2remove = np.diff(n_features)
    all_features = copy.deepcopy(orderedFeatures_initial)
    features_removed = []
    for f,n_feats in tqdm.notebook.tqdm(enumerate(n_features), total = len(n_features), disable=False, desc='Boruta FE'):    
        if f!=0:
            features_removed.append(all_features[n_features2remove[f-1]:])
            all_features= all_features[:n_features2remove[f-1]]
        curr_step = model_shap_classes[model_type](df = pd.concat([df[all_features], df[target]],axis=1), target=target, max_evals=max_evals, verbose=verbose, hyperparams=hyperparams)
        if max_evals > 0:
            if verbose: 
                print("Step:", f, "n_feats", n_feats, "Tuning current...")
            curr_step.tune_model(verbose=False) 
        curr_score = curr_step.model_performance(verbose=False)
        all_scores.append(curr_score)
    mean_score = np.array(all_scores).mean(axis=1)
    optimal_features = orderedFeatures_initial[:n_features[np.where(mean_score==np.max(mean_score))[0][0]]]
    
    return dict(FE_method = FE_method,
                mean_score = mean_score,
                std_score = np.array(all_scores).std(axis=1),
                n_features = n_features, 
                features_removed = features_removed,
                optimal_features = optimal_features)

class feature_selection(analysis):
    def __init__(self, df, target, model_type: currently_supported_models, max_evals=25, hyperparams=None, keep_features=[], verbose=True, outputs_dir='./', remove_outliers=False): 
        analysis.__init__(self, df, target=target, remove_outliers=remove_outliers, outputs_dir=outputs_dir)
        self.max_evals=max_evals
        self.hyperparams = hyperparams
        self.n_folds_SHAP=10
        self.SHAP_based_FE_outputs = "call 'run_SHAP_based_FE()' first"
        self.SHAP_based_RFECV_outputs = "call 'run_SHAP_based_RFECV()', first" 
        self.Boruta_based_FE_outputs = "call 'run_boruta_FE()', first"
        self.keep_features = keep_features
        self.verbose = verbose
        self.model_type=model_type

    def tune_model(self, verbose=True, **kwargs):
        """ kwargs:  additionl kwargs to supple to the scoring fcn (this could be model dependent - like penalty terms)"""
        start=time.time()
        def objective_model_curr(hyperparams, train_x=self.X, train_y=self.y):
            return objective_fcn(hyperparams, train_x=train_x, train_y=train_y, model_type=self.model_type, **kwargs)
        hyperparams = optimize_model(objective_model_curr, model_type=self.model_type, verbose=verbose, max_evals=self.max_evals)
        if self.model_type== 'xgb_binary':
            hyperparams.update({"objective": "binary:logistic", 'tree_method':"exact", "eval_metric":"error"})
        elif self.model_type=='xgb_regression':
            hyperparams.update({"objective": "reg:squarederror", 'tree_method':"exact", "eval_metric":"error"})
        elif self.model_type=='xgb_cox':
            hyperparams.update({"objective": "survival:cox"})        
        end=time.time()
        if verbose:
            print("Done: took", (end-start), "seconds")
            print("The best hyperparameters are: ", "\n")
            print(hyperparams)
            setattr(self, 'hyperparams', hyperparams)
        return hyperparams
        
    # SHAP analysis
    @LazyProperty
    def shap_values(self):
        SHAP_outputs = model_shap_cv_fcns[self.model_type](df=self.df, mdlFeatures=self.mdlFeatures, target=self.target, hyperparams=self.hyperparams, nFolds=self.n_folds_SHAP)
        setattr(self, 'score_CV_SHAP', SHAP_outputs['CV_score'])
        setattr(self, 'predictionsCV_SHAP', SHAP_outputs['valPredictions_vec'])
        setattr(self, 'SHAP_outputs', SHAP_outputs)
        setattr(self, 'meanExpValue', SHAP_outputs['meanExpValue'])
        return SHAP_outputs['shap_values']

    @LazyProperty
    def shapDF(self):
        shapDF = pd.DataFrame(self.shap_values, columns=self.mdlFeatures)
        shapDF['expectedValue'] = list(self.SHAP_outputs['expectedValues_d'].values())
        return shapDF
    
    @LazyProperty
    def orderedFeatures_initial(self):
        """ Order of importance based on SHAP analysis"""
        print("Calculating feature importance on full model")
        if self.hyperparams == None: 
            self.tune_model()
        vals= np.abs(self.shap_values).mean(0) # Ordered by average |SHAP value| 
        ordered_Features = list(pd.DataFrame(zip(self.mdlFeatures,vals)).sort_values(1, ascending=False).reset_index(drop=True)[[0]].values.flatten())
        return ordered_Features
    
    def initial_shap_summary_plots(self, show=True, figsize=(40,12), max_display=30, save_fig=False): 
        fig = plt.figure(figsize=figsize)
        #fig.subplot(1,2,1)
        gs = fig.add_gridspec(1, 3)
        fig.add_subplot(gs[0, 0])
        shap.summary_plot(self.shap_values, self.df[self.mdlFeatures], max_display=max_display, show=False, plot_size=(figsize[0]/3,figsize[1]), plot_type='bar')
        #plt.subplot(1,2,2)
        fig.add_subplot(gs[0, 1:])
        shap.summary_plot(self.shap_values, self.df[self.mdlFeatures], max_display=max_display, show=False, plot_size=(2*figsize[0]/3,figsize[1]))
        fig.text(.1, 1, "score (CV): {:.3f}".format(self.score_CV_SHAP), ha='left', fontsize = 18)
        plt.tight_layout()
        if save_fig:
            file_name= self.outputs_dir + "SHAP summary plots {}.png".format(self.target)
            plt.savefig(file_name, bbox_inches='tight')
            print("Saved: ", file_name)
            return
        if show: 
            pass
        else:
            plt.close()
            return fig
    
    def run_SHAP_based_FE(self, n_steps=30, max_evals=0, verbose=False):
        
        out = SHAP_based_FE(df=self.df, target=self.target, orderedFeatures_initial= self.orderedFeatures_initial, n_steps=n_steps, model_type = self.model_type, hyperparams=self.hyperparams, verbose=verbose)
        setattr(self, 'SHAP_based_FE_outputs', out)
        return out
    
    def run_SHAP_based_RFECV(self, n_steps=30, max_evals=25, verbose=False):
        out = SHAP_based_RFECV(df=self.df, target=self.target, orderedFeatures_initial= self.orderedFeatures_initial, n_steps=n_steps, max_evals=max_evals, verbose=verbose, model_type = self.model_type, hyperparams=self.hyperparams)
        setattr(self, 'SHAP_based_RFECV_outputs', out)
        return out
    
    def run_boruta_FE(self, n_steps=30, max_iter=100, max_evals=0, verbose=False):
        if self.hyperparams == None:
            self.tune_model(verbose=False)
        out = boruta_FE(df=self.df, target=self.target, model_type = self.model_type, hyperparams=self.hyperparams, max_iter=max_iter, n_steps=n_steps, max_evals=max_evals, verbose=verbose)
        setattr(self, 'Boruta_based_FE_outputs', out)
        return out
    
    def plot_FE_comparison(self, save_fig=False, return_fig=False, outputs_dir=None):
        if outputs_dir == None:
            outputs_dir = self.outputs_dir
        else: 
            outputs_dir=outputs_dir
        plt.style.use('seaborn-poster')
        all_FE_outputs=[self.SHAP_based_FE_outputs, self.SHAP_based_RFECV_outputs, self.Boruta_based_FE_outputs]
        all_FE_outputs = [out for out in all_FE_outputs if type(out) == dict]
        # import pdb; pdb.set_trace()
        if len(all_FE_outputs)>0:
            fig,ax = plt.subplots(1,1, figsize=(30,8)) 
            for t,out in enumerate(all_FE_outputs): 
                curr_plot = ax.plot(out['n_features'], out['mean_score'], '-o', label = out['FE_method'])
                color= curr_plot[0].get_color()
                ax.fill_between(out['n_features'], out['mean_score']-out['std_score'], out['mean_score']+out['std_score'], alpha = .2, label=out['FE_method'])  
                # Optimal lines:
                ax.vlines(x=len(out['optimal_features'])- .1*t, ymin =.5 , ymax=np.mean(out['mean_score'] - 1.5*out['std_score']), linestyles='--', lw = 3, color=color)
                for idx, x in enumerate(out['n_features']):
                    if idx%2==1:
                        ax.text(x, out['mean_score'][idx] - .4*t*out['std_score'][idx], "{:.3f}±{:.2f}".format(out['mean_score'][idx], out['std_score'][idx]), 
                                size=16, ha='right', va='top', color=color, rotation=45)

            ax.invert_xaxis()
            ax.legend()
            ax.set_ylabel('Score (CV)')
            ax.set_xlabel('# of features')

            plt.title("Comparison of feature elimination methods")
            #plt.ylim([.5,.85])
            plt.grid(axis='both',ls='--')
            if save_fig:
                file_name= outputs_dir + "Comparison of FE methods {}.png".format(self.target)
                plt.savefig(file_name, bbox_inches='tight', dpi=300)
                print("Saved: ", file_name)
            if return_fig: 
                return fig
            else: 
                return ax
        else:
            print("Must run feature elimination method(s) first")
     
    def get_union_features(self, importance_sort=True):
        all_FE_outputs=[self.SHAP_based_FE_outputs, self.SHAP_based_RFECV_outputs, self.Boruta_based_FE_outputs]
        all_FE_outputs = [out for out in all_FE_outputs if type(out) == dict]
        if len(all_FE_outputs)>0:
            union_features = list(set(list(itertools.chain(*[out['optimal_features'] for out in all_FE_outputs]))))
            # if type(self.keep_features) != type(None):
            union_features.extend(self.keep_features)
            union_features=list(set(union_features))
        else: 
            raise ValueError("Must run feature elimination method(s) first")
            

        if importance_sort:
            if self.verbose: 
                print("Sorting features by importance")
            mdlVars = copy.deepcopy(union_features)
            mdlVars.append(self.target)
            curr_analysis = model_shap_classes[self.model_type](df=self.df[mdlVars], target=self.target, max_evals=25)
            curr_analysis.tune_model()
            union_features = curr_analysis.orderedFeatures
            setattr(self, 'union_features', union_features)
        return union_features

    def get_intersect_features(self, importance_sort=True):
        all_FE_outputs=[self.SHAP_based_FE_outputs, self.SHAP_based_RFECV_outputs, self.Boruta_based_FE_outputs]
        all_FE_outputs = [out for out in all_FE_outputs if type(out) == dict]
        if len(all_FE_outputs)>0:
            selected_features = [out['optimal_features'] for out in all_FE_outputs]
            intersect_features = list(set.intersection(*map(set,selected_features)))
            # if type(self.keep_features) != type(None):
            intersect_features.extend(self.keep_features)
        else: 
            print("Must run feature elimination method(s) first")

        if importance_sort:
            if self.verbose: 
                print("Sorting features by importance")
            mdlVars = copy.deepcopy(intersect_features)
            mdlVars.append(self.target)
            curr_analysis = model_shap_classes[self.model_type](df=self.df[mdlVars], target=self.target, max_evals=25)
            curr_analysis.tune_model()
            intersect_features = curr_analysis.orderedFeatures
            setattr(self, 'intersect_features', intersect_features)
        return intersect_features

    def run_forward_selection(self, features=None, verbose=True, n_folds=5, n_reps=5, seed=0, hyperparams=None, order_initial_features=True, include_random_var=True):
        tmpDF = self.df.copy()
        from scipy.stats import sem
        if type(features) == type(None):
            mdlVars = copy.deepcopy(self.union_features)
        else:
            mdlVars = copy.deepcopy(features)

        if include_random_var:
            mdlVars.append('random_var')
            np.random.seed(seed+1000)
            tmpDF['random_var'] = np.random.random(tmpDF.shape[0])
        mdlVars.append(self.target)    

        init_analysis = model_shap_classes[self.model_type](df=tmpDF[mdlVars], target=self.target, max_evals=25, n_folds_SHAP=10)
        if type(hyperparams) == type(None):
            hyperparams = init_analysis.tune_model()
        else:
            init_analysis.hyperparams=hyperparams

        if order_initial_features:
            features = copy.deepcopy(init_analysis.orderedFeatures)
            print("Initial ordering of features: ", features)
            mdlVars = copy.deepcopy(features)
            mdlVars.append(self.target) 
        else: 
            mdlVars = mdlVars
        # hyperparams = {'eta': 0.08725904914699555, 'max_depth': 3, 'min_child_weight': 0.023737374638472843, 'reg_alpha': 0.897750469255347, 'reg_lambda': 0.04481058665391393, 'subsample': 0.6851140206115516, 'objective': 'binary:logistic', 'tree_method': 'exact', 'eval_metric': 'error'}

        # include_opts = ['Not included', 'Tentative', 'Included']
        # Features are included if they significantly increase model performance. 
        # Tentative inclusion if they cause no change or increase model performance slightly
        # Not included if they decrease model performance

        prev_mean = .5
        prev_inc_limit = .5
        currMdlVars = [self.target]
        curr_list_d = {k:v for k,v in enumerate(features)}
        steps = []
        exculded_features = []; included_features=[]; tentative_features=[];
        step_outcomes = []
        mean_scores = []
        std_scores = []
        sem_scores = []
        for f in tqdm.notebook.tqdm(range(len(features)), disable=False, desc='Forward selection'): # defines the initial order we'll do the step_wise selection
            curr_feat = curr_list_d[f]
            currMdlVars.append(curr_feat)
            if verbose:
                print(f"Step {f} :{curr_feat}\nTesting {len(currMdlVars)-1} feature(s) currently")
            curr_analysis = model_shap_classes[self.model_type](df=tmpDF[currMdlVars], target=self.target, hyperparams=hyperparams)
            # import pdb; pdb.set_trace()
            curr_scores = curr_analysis.model_performance(n_folds=n_folds, n_reps=int(n_reps), verbose=verbose, seed=seed)
            mean_score, std_score, sem_score = np.mean(curr_scores), np.std(curr_scores), sem(curr_scores)
            mean_scores.append(mean_score)
            std_scores.append(std_score)
            sem_scores.append(sem_score)
             
            if verbose:
                print("Mean: {:.4f}, STDEV: {:.4f} SEM {:.4f}".format(mean_score, std_score, sem_score))
            if mean_score > prev_inc_limit:# Include
                steps.append("+ "+curr_feat)
                include="Included"
                prev_inc_limit = mean_score + sem_score
                prev_mean = mean_score
                if curr_feat != 'random_var':
                    included_features.append(curr_feat)
                # currMdlVars = currMdlVars
                # curr_list = curr_list
            elif mean_score >= prev_mean: # Tentative
                steps.append("+ "+curr_feat + "*")
                include="Tentatively included"
                prev_mean = mean_score
                prev_inc_limit = mean_score + sem_score
                if curr_feat != 'random_var':
                    tentative_features.append(curr_feat)
                # currMdlVars = currMdlVars
                # curr_list = curr_list
            elif mean_score < prev_mean: # excluded
                steps.append("- "+curr_feat)
                include="Excluded"
                exculded_features.append(currMdlVars.pop())
                # Recalculate feature importance and
                def update_curr_list_d(curr_list_d, currMdlVars, exculded_features, f, hyperparams=hyperparams):
                    currMdlFeatures = copy.deepcopy(currMdlVars)
                    currMdlFeatures.pop(currMdlFeatures.index(self.target))
                    viableMdlVars = list(set(mdlVars).difference(exculded_features))
                    # viableMdlVars.append(self.target)
                    tmpAnalysis= model_shap_classes[self.model_type](df=tmpDF[viableMdlVars], target=self.target, hyperparams=hyperparams, n_folds_SHAP=10, verbose=verbose)
                    # import pdb; pdb.set_trace()
                    remaining_features = tmpAnalysis.orderedFeatures
                    for feat in currMdlFeatures:
                        remaining_features.pop(remaining_features.index(feat));
                    curr_list_d.update({k+f+1:v for k,v in enumerate(remaining_features)})
                    return
                update_curr_list_d(curr_list_d, currMdlVars, exculded_features, f, hyperparams=hyperparams)
            step_outcomes.append(include)
            if verbose:
                print(f"Step {f} outcome: {include} {curr_feat} \n")
        
        # Put the target at the end of the list:
        currMdlVars.pop(currMdlVars.index(self.target));
        currMdlVars.append(self.target)
        mdl_vars_forward_selection = currMdlVars # Note: this includes ones tentatively accepted
        print("Variables selected by forawrd selection :\n", mdl_vars_forward_selection)
        df_d = dict(step = steps, mean_score = mean_scores,
                    std_score = std_scores, sem_score = sem_scores,
                    outcome = step_outcomes)

        forward_selection_results_df = pd.DataFrame(df_d)
        setattr(self, 'forward_selection_results', forward_selection_results_df)
        setattr(self, 'mdl_vars_forward_selection', mdl_vars_forward_selection)
        setattr(self, 'outputs_forward_selection', dict(included_features = included_features, 
                    tentative_features = tentative_features, 
                    exculded_features = exculded_features,
                    n_reps = n_reps,
                    n_folds = n_folds))
        if verbose:
            self.plot_forward_selection(save_fig=False)
        return forward_selection_results_df

    def run_repeated_forward_selection(self, features=None, verbose=False, n_folds=5, n_iterations=50, n_reps=5, include_random_var=True, hyperparams=None, order_initial_features=True, parallel=False, for_loop=False, plot_results=False):
        # TODO: modify code to make this compatible with parallelization
        # for_loop is a temporary fix that chekcpoints results since jupyter lab crashes during this time consuming algo on Rosalind
        tmpDF = self.df.copy()
        if type(features) == type(None): 
            mdlVars = copy.deepcopy(self.union_features)
            mdlVars.append(self.target)
        else: 
            mdlVars = copy.deepcopy(features)
            mdlVars.append(self.target)

        if include_random_var:
            mdlVars.append('random_var')
            tmpDF['random_var'] = np.random.random(tmpDF.shape[0])

        init_analysis = model_shap_classes[self.model_type](df=tmpDF[mdlVars], target=self.target, max_evals=25)
        if type(hyperparams) == type(None):
            print('Calculating hyperparams for repeated forward selection')
            init_hyperparms = init_analysis.tune_model()
        else:
            init_analysis.hyperparams=hyperparams

        if order_initial_features:
            features = copy.deepcopy(init_analysis.orderedFeatures)
            features2save = copy.deepcopy(init_analysis.orderedFeatures)
            print("Initial ordering of features: ", mdlVars)
            mdlVars = copy.deepcopy(features)
            mdlVars.append(self.target) 
        else: 
            features2save = copy.deepcopy(mdlVars)
            features2save.pop(features2save.index(self.target))
            mdlVars = mdlVars
        print("Initial feature order for forward selection: ", mdlVars)
        
        if 'random_var' in features: 
            features.pop(features.index('random_var'))
        # from multiprocessing import Pool, cpu_count
        tasks=range(n_iterations)
        from multiprocessing import Pool, current_process, cpu_count
        from ..utils.helpers import globalize
        @globalize
        def fcn(seed, features=features, include_random_var=include_random_var, n_folds=n_folds, n_reps=n_reps):
            curr_FS = self.copy()
            if include_random_var:
                np.random.seed(seed)
                curr_FS.df['random_var'] = np.random.random(curr_FS.df.shape[0])
            # print(curr_FS.df.columns)
            # print(seed)
            curr_FS.run_forward_selection(features=features, seed=seed, n_folds=n_folds, n_reps=n_reps, verbose=verbose, hyperparams=init_hyperparms); # seed for cross-validation
            curr_selected_features = copy.deepcopy(curr_FS.outputs_forward_selection['included_features'])
            curr_selected_features.extend(curr_FS.outputs_forward_selection['tentative_features'])
            return curr_selected_features
        
        if parallel: 
            pool = Pool(processes=cpu_count())
            r = list(tqdm.notebook.tqdm(pool.imap(fcn, tasks), total=len(tasks), desc='Repeating forward selection')) # seed with the task value
            pool.close()
        elif for_loop:
            r = []
            for ii in tasks:
                r.append(fcn(seed=ii))
                import pickle
                fileName = self.outputs_dir + "_".join([self.target, "repeated_forward_results"]) +".p"
                out = {'repeated_forward_selection_results': r,
                'n_iterations_for_repeated_forward_selection_results': n_iterations, 
                'repeated_forward_selection_features': features2save}
                pickle.dump(out, open(fileName, "wb"))
        else:
            r = list(tqdm.notebook.tqdm(map(fcn, tasks), total=len(tasks), desc='Repeating forward selection')) # seed with the task value

        
        setattr(self, 'repeated_forward_selection_results', r)
        setattr(self, 'repeated_forward_selection_features', features2save) # This is the initial set of features (including random_var if applicable)
        setattr(self, 'n_iterations_for_repeated_forward_selection_results', n_iterations)

        from collections import Counter, defaultdict
        import itertools
        selected_vars_flattened = list(itertools.chain(*list(r)))
        selected_vars_d = defaultdict(lambda : 0, dict(Counter(selected_vars_flattened).most_common(len(features2save))))
        for feat in self.repeated_forward_selection_features:
            tmp = selected_vars_d[feat] # This is just to include all features in the default dictionary

        print("Features selected more than 50% of the time:")
        tmp_list=[]
        for feat in features2save:
            if selected_vars_d[feat] > n_iterations/2:
                tmp_list.append(feat)
        tmp_list.extend(self.keep_features)
        tmp_list=list(set(tmp_list))
        print(tmp_list)
        setattr(self, 'features_selected_above_half', tmp_list)

        if "random_var" in features2save:
            tmp_list=[]
            print("Variables selected more than random_var")
            for feat in features2save:
                if selected_vars_d[feat] > selected_vars_d['random_var']:
                    tmp_list.append(feat)
            tmp_list.extend(self.keep_features)
            tmp_list=list(set(tmp_list))
            print(tmp_list)
            setattr(self, 'features_selected_above_random_var', tmp_list)
        print("Saved: self.repeated_forward_selection_results")
        if plot_results:
            self.plot_repeated_forward_selection_results()
        return r

    def plot_repeated_forward_selection_results(self, outputs_dir=None, save_fig=False, return_fig=False):
        r = self.repeated_forward_selection_results
        n_iterations = self.n_iterations_for_repeated_forward_selection_results
        from collections import Counter, defaultdict
        import itertools
        selected_vars_flattened = list(itertools.chain(*list(r)))
        selected_vars_d = defaultdict(lambda : 0, dict(Counter(selected_vars_flattened).most_common(50)))
        for feat in self.repeated_forward_selection_features:
            tmp = selected_vars_d[feat] # This is just to include all features in the default dictionary
        import scipy
        critical_value = .05/len(self.repeated_forward_selection_features)
        print("Critical value from Bonferroni correction: {:.5f}".format(critical_value))
        p = 1
        n_for_p05 = False
        n_for_pCrit = np.nan
        while p > critical_value: 
            for i in range(n_iterations):
                p = scipy.stats.binom_test(i, n_iterations, .5, alternative='greater')
                if (p < .05) and not n_for_p05:
                    n_for_p05 = i
                    print("n required for p<0.05 : ", n_for_p05)
                if p < critical_value:
                    n_for_pCrit = i
                    print("n required for p<{:.5f} : ".format(critical_value), n_for_pCrit)
                    break
            break

        labels, data = [*zip(*selected_vars_d.items())]  # 'transpose' items to parallel key, value lists
        fig=plt.figure(figsize=(16,6))
        plt.bar(range(1, len(labels) + 1), height=data)
        plt.hlines(y=np.ceil(n_iterations/2), xmin=0, xmax=len(labels)+1, ls='--', lw=1, label='Chance')
        plt.hlines(y=n_for_p05, xmin=0, xmax=len(labels)+1, ls='--', color='g', lw=2, label='p<0.05')
        plt.hlines(y=n_for_pCrit, xmin=0, xmax=len(labels)+1, ls='--', lw=2, color='r', label='p<{:.5f}'.format(critical_value))
        plt.hlines(y=n_iterations, xmin=0, xmax=len(labels)+1, ls='-', lw=2, color='k', label=f'N iterations')
        plt.xticks(range(1, len(labels) + 1), labels, rotation=90)
        plt.ylabel("# of times selected")
        
        print(f"This plot shows how often features improved model performance when repeating forward selection {n_iterations}")
        # generate errorbars
        def binomVec_yerr(vec):
            from statsmodels.stats.proportion import proportion_confint
            lwr, upr = np.abs(proportion_confint(vec.sum(), len(vec), method='wilson')-np.mean(vec))
            point_estimate = np.mean(vec)
            return point_estimate, lwr, upr
        def binom_err(series):
            point_estimate, lwr, upr = binomVec_yerr(series)
            return [lwr,upr]
        all_arr = []
        for curr_row in r:
            curr_arr=[]
            for feat in self.repeated_forward_selection_features:
                curr_arr.append(curr_row.count(feat))
            all_arr.append(curr_arr)
        # import pdb; pdb.set_trace()
        tmpDF = pd.DataFrame(all_arr, columns=self.repeated_forward_selection_features)
        errDF = tmpDF.agg(['mean', binom_err, 'sum'])
        
        for i,k in enumerate(selected_vars_d):
            # print(i+1,k)
            plt.errorbar(x=i+1, y=errDF.loc['sum', k],yerr=(np.array(errDF.loc['binom_err', k]) * errDF.loc['sum', k]).reshape(-1,1), fmt='.', color='gray')
        # plt.show()
        print("Starting features: ", self.repeated_forward_selection_features)
        print("Variables significantly improving performance (p<{:.5f})(Bonferoni correction):".format(critical_value))
        tmp_list=[]
        for feat in self.repeated_forward_selection_features:
            if selected_vars_d[feat] >= n_for_pCrit:
                tmp_list.append(feat)
        print(tmp_list)
        

        print("Variables improving performance p<0.05:")
        tmp_list=[]
        for feat in self.repeated_forward_selection_features:
            if selected_vars_d[feat] >= n_for_p05:
                tmp_list.append(feat)
        print(tmp_list)
        if "random_var" in self.repeated_forward_selection_features:
            tmp_list=[]
            print("Variables selected more than random_var (standard normal random variable)")
            for feat in self.repeated_forward_selection_features:
                if selected_vars_d[feat] > selected_vars_d['random_var']:
                    tmp_list.append(feat)
            print(tmp_list)
            plt.hlines(y=selected_vars_d['random_var'], xmin=0, xmax=len(labels)+1, ls='--', lw=2, color='gray', label='random')
        
        plt.title("Repeated forward selection ({:.0f} times)".format(self.n_iterations_for_repeated_forward_selection_results))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        if outputs_dir == None:
            outputs_dir = self.outputs_dir
        else: 
            outputs_dir=outputs_dir
        if save_fig:
            file_name= outputs_dir + "Repeated forward selection results {}.png".format(self.target)
            plt.savefig(file_name, bbox_inches='tight', dpi=300)
            print("Saved: ", file_name)
        if return_fig:
            return fig
        else:
            return selected_vars_d

    def plot_forward_selection(self, save_fig=False, return_fig=False, outputs_dir=None, figsize=(20,8)):
        if outputs_dir == None:
            outputs_dir = self.outputs_dir
        else: 
            outputs_dir=outputs_dir
        plt.style.use('seaborn-poster')
        out = self.forward_selection_results.to_dict()
        for k in out:
            if k != 'step':
                out[k] = np.array(list(out[k].values()))
        fig,ax = plt.subplots(1,1, figsize=figsize)
        ax.plot(np.array(self.forward_selection_results.index)+1, 
                            out['mean_score'], '-o', color='gray', alpha=.2, label = "forward selection")

        ax.fill_between(np.array(self.forward_selection_results.index)+1, 
                        out['mean_score']-out['std_score'], 
                        out['mean_score']+out['std_score'], color='gray', alpha = .2, label='STD')  
            
        for idx, x in enumerate(self.forward_selection_results.index+1):
            ax.text(x, out['mean_score'][idx] - 1.1* out['std_score'][idx], "{} {:.3f}±{:.2f}".format(out['step'][idx], out['mean_score'][idx], out['std_score'][idx]), 
                    size=16, ha='center', va='top', color='k', rotation=90)
        for curr_outcome, color in zip(['Included', 'Tentatively included', 'Excluded'],['g', 'yellow', 'r']):
            idxs = [idx+1 for idx,outcome in enumerate(out['outcome']) if (outcome==curr_outcome) ]
            mean_scores = [out['mean_score'][idx] for idx,outcome in enumerate(out['outcome']) if (outcome==curr_outcome) ]
            if curr_outcome == 'Tentatively included':
                curr_outcome = 'Included (minor improvement)'
            ax.scatter(x=idxs, y=mean_scores, color=color, label = curr_outcome, s=100)
            
        ax.legend(framealpha=.2, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_ylabel('Score (CV)')
        ax.set_xlabel('Step #')
        ax.set_xlim(left=0)
        n_folds = self.outputs_forward_selection['n_folds']
        n_reps = self.outputs_forward_selection['n_reps']
        plt.title(f"Forward selection results\n using {n_folds} fold CV with {n_reps} reps")
        plt.ylim([.1,.95])
        plt.grid(axis='both',ls='--')
        if save_fig:
            file_name= outputs_dir + "Forward selection results {}.png".format(self.target)
            plt.savefig(file_name, bbox_inches='tight', dpi=300)
            print("Saved: ", file_name)
        if return_fig: 
            return fig
        else: 
            return ax

    def run_model_comparison(self, verbose=None, n_folds=5, n_reps=10, max_evals=25):
        try: 
            assert type(self.union_features) == list
        except: 
            raise AssertionError("Must run at least 1 feature elimination method. i.e. self.run_SHAP_based_FE(), and then forward selection i.e. self.run_forward_selection()")
        try: 
            assert type(self.outputs_forward_selection) == dict
        except: 
            raise AssertionError("Must run forward selection i.e. self.run_forward_selection()")
            
        if type(verbose) == type(None):
            verbose=self.verbose
            try:
                performance_comparison_outputs = self.performance_comparison_outputs
                print("Loaded cached performance evaluation")
                assert n_folds == performance_comparison_outputs['n_folds']
                assert n_reps == performance_comparison_outputs['n_reps']
                self.plot_model_comparison()
            except:
                print("Running performance evaluation")
                mean_scores = []
                std_scores = []
                models = ['full', 'union', 'post_forward_selection_including_tentative', 'post_forward_selection_excluding_tentative']
                
                for model in models:
                    if model == 'full':
                        features = copy.deepcopy(self.orderedFeatures_initial)
                    elif model == 'union':
                        features = copy.deepcopy(self.union_features)
                    elif model == 'post_forward_selection_including_tentative':
                        features = copy.deepcopy(self.outputs_forward_selection['included_features'])
                        features.extend(self.outputs_forward_selection['tentative_features']) 
                    elif model == 'post_forward_selection_excluding_tentative':
                        features = self.outputs_forward_selection['included_features']
                    if verbose: 
                        print(model)
                        print(features)

                    mdlVars=copy.deepcopy(features)
                    mdlVars.append(self.target)
                    curr_analysis = model_shap_classes[self.model_type](df=self.df[mdlVars], target=self.target, max_evals=max_evals)
                    curr_analysis.tune_model()
                    curr_scores = curr_analysis.model_performance(n_folds=n_folds, n_reps=int(n_reps), verbose=verbose, seed=2) # seed for forward selection was 0, so we are getting an independent assessment of perfomance here
                    mean_score, std_score = np.mean(curr_scores), np.std(curr_scores)
                    mean_scores.append(mean_score)
                    std_scores.append(std_score)

                performance_comparison_outputs = dict(models=models, 
                                            mean_scores = mean_scores, 
                                            std_scores = std_scores,
                                            n_folds=n_folds, 
                                            n_reps=n_reps)
                setattr(self, 'performance_comparison_outputs', performance_comparison_outputs)
                self.plot_model_comparison()

    def plot_model_comparison(self, outputs_dir=None, return_fig=False, save_fig=False, figsize=(6,4)):
        assert type(self.performance_comparison_outputs)==dict
        performance_comparison_outputs = self.performance_comparison_outputs
        labels=[]
        fig, ax = plt.subplots(1,1, figsize=figsize)
        for idx,model in enumerate(performance_comparison_outputs['models']):
            label = model + "\n{:.3f}±{:.2f}".format(performance_comparison_outputs['mean_scores'][idx], performance_comparison_outputs['std_scores'][idx])
            plt.errorbar(x=idx, y=performance_comparison_outputs['mean_scores'][idx], 
                        yerr=performance_comparison_outputs['std_scores'][idx], 
                        label=label, fmt='ok')
            labels.append(label)
        plt.xticks(range(len(performance_comparison_outputs['models'])), labels=labels, rotation=45, ha='right')
        n_folds = performance_comparison_outputs['n_folds']
        n_reps = performance_comparison_outputs['n_reps']
        plt.title(f"Model comparison\n{n_folds} fold-CV repeated {n_reps}")
        ylims = (np.mean(performance_comparison_outputs['mean_scores']) - 2* np.mean(performance_comparison_outputs['std_scores']),
            np.mean(performance_comparison_outputs['mean_scores']) + 2* np.mean(performance_comparison_outputs['std_scores']))
        
        plt.yticks(np.linspace(.5,1,21))
        plt.ylim(list(ylims))
        plt.ylabel('Score (CV)')
        plt.grid(axis='both',ls='--', alpha=.2)
        if outputs_dir == None:
            outputs_dir = self.outputs_dir
        else: 
            outputs_dir=outputs_dir
        if save_fig:
            file_name= outputs_dir + "FS model comparison {}.png".format(self.target)
            plt.savefig(file_name, bbox_inches='tight')
            print("Saved: ", file_name)
        if return_fig: 
            return fig
        else: 
            return ax

    def run_workflow(self, SHAP_FE=True, SHAP_RFECV=True, borutaFE=True, forward_selection=True, 
    n_steps=30, verbose=False, n_reps_forward_selection=5, RFECV_max_evals=25, run_model_comparison_post_forward_selection=True):
        """
        Currently, model comparisons are only run after running forward selection. 
        run_model_comparison_post_forward_selection ==True
        """
        self.orderedFeatures_initial;
        if SHAP_FE:
            self.run_SHAP_based_FE(n_steps=n_steps, verbose=verbose)
        if SHAP_based_RFECV:
            self.run_SHAP_based_RFECV(n_steps=n_steps, verbose=verbose, max_evals=RFECV_max_evals)
        if boruta_FE: 
            self.run_boruta_FE(n_steps=n_steps, verbose=verbose)
        self.plot_FE_comparison()
        self.get_union_features(importance_sort=True)
        if forward_selection:
            self.run_forward_selection(n_reps=n_reps_forward_selection, verbose=verbose, seed=0)
            self.run_model_comparison(verbose=verbose)

    @LazyProperty
    def union_features(self): 
        return self.get_union_features(importance_sort=True)

    @LazyProperty
    def intersect_features(self): 
        return self.get_intersect_features(importance_sort=True)



