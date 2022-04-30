import time, tqdm, copy
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import time, itertools, copy, tqdm


from .shap_based_analysis import xgb_shap, LazyProperty, optimize_xgb, objective_model, LazyProperty, nargout, SHAP_CV
import shap
import xgboost as xgb
from boruta import BorutaPy
from ..utils.superclasses import analysis
# from RHML.utils import msnoMatrix

######################## Feature Elimnation (This requires the xgb_shap class)


# exponential_spacing = lambda stop, n_steps=25: np.unique(np.round(10**np.linspace(0,np.log10(stop), num=n_steps))).astype(int)
def exponential_spacing(stop, n_steps=30):
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
            ratio = (float(stop)/result[-1]) ** (1.0/(n_steps-len(result)))
    # round, re-adjust to 0 indexing (i.e. minus 1) and return np.uint64 array
    return np.array(list(map(lambda x: round(x), result)), dtype=np.uint64).astype(int)


def SHAP_based_FE(df, target, orderedFeatures_initial, n_steps, xgb_hyperparams, max_evals=0, verbose=True):
    """
    FE based on an initial SHAP-based ranking of features
    max_evals : hyperparamter tuning iterations during backwards elimination (set to 0 if wanting to omit this step)
    """
    FE_method = "SHAP_FE"
    all_aurocs = []
    n_features = list(exponential_spacing(stop=len(orderedFeatures_initial), n_steps=n_steps))[::-1]
    n_features2remove = np.diff(n_features)
    all_features = copy.deepcopy(orderedFeatures_initial)
    features_removed = []
    for f,n_feats in tqdm.notebook.tqdm(enumerate(n_features), total = len(n_features), disable=False, desc='SHAP FE'):
        curr_step = xgb_shap(df = pd.concat([df[orderedFeatures_initial[:n_feats]], df[target]],axis=1), target=target, xgb_hyperparams=xgb_hyperparams, verbose=verbose)
        if max_evals > 0:
            if verbose: 
                print("Step:", f, "n_feats", n_feats, "Tuning current...")
            curr_step.tune_model(verbose=verbose)
        curr_auroc = curr_step.model_performance(verbose=verbose)
        all_aurocs.append(curr_auroc)
        if f!=0:
            features_removed.append(all_features[n_features2remove[f-1]:])
            all_features= all_features[:n_features2remove[f-1]]
    
    mean_auroc = np.array(all_aurocs).mean(axis=1)
    optimal_features = orderedFeatures_initial[:n_features[np.where(mean_auroc==np.max(mean_auroc))[0][0]]]
    
    return dict(FE_method = FE_method,
                mean_auroc = mean_auroc,
                std_auroc = np.array(all_aurocs).std(axis=1),
                n_features = n_features, 
                features_removed = features_removed,
                optimal_features = optimal_features)

def SHAP_based_RFECV(df, target, orderedFeatures_initial, xgb_hyperparams=None, n_steps=25, max_evals=25,verbose=True):
    """
    FE based on an SHAP-based ranking of features, whereby features are re-ranked at each iteration
    max_evals : hyperparamter tuning iterations during backwards elimination (set to 0 if wanting to omit this step)
    """
    FE_method = "SHAP_RFECV"
    all_aurocs = []
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
        curr_step = xgb_shap(df = pd.concat([df[remaining_features], df[target]],axis=1),
            target=target, max_evals=max_evals, n_folds_SHAP=10, verbose=verbose, xgb_hyperparams=xgb_hyperparams)
        if max_evals > 0:
            if verbose: 
                print("Step:", f, "n_feats", n_feats, "Tuning current...")
            curr_step.tune_model(verbose=False) 
        curr_auroc = curr_step.model_performance(verbose=verbose)
        all_aurocs.append(curr_auroc)
        if f!=0:
            # recalculate feature importance after each step starting with the first step
            remaining_features = curr_step.orderedFeatures
    all_features = copy.deepcopy(features_removed)
    all_features.append(remaining_features)
    ordered_features_posthoc = list(itertools.chain(*list(reversed(all_features))))
    mean_auroc = np.array(all_aurocs).mean(axis=1)
    optimal_features = ordered_features_posthoc[:n_features[np.where(mean_auroc==np.max(mean_auroc))[0][0]]]
    
    return dict(FE_method = FE_method,
                mean_auroc = np.array(all_aurocs).mean(axis=1),
                std_auroc = np.array(all_aurocs).std(axis=1),
                n_features = n_features, 
                features_removed = features_removed,
               optimal_features = optimal_features)

def boruta_FE(df, target, xgb_hyperparams, max_iter=100, n_steps=25, max_evals=0, verbose=True):      
    """
    This function gets the ranked order importance of features using Boruta and then performs backwards feature elimination. 
    max_iter : number of iterations the data set is shuffled to obtain feature rankings
    max_evals : hyperparamter tuning iterations during backwards elimination (set to 0 if wanting to omit this step)
    
    """
    FE_method = "Boruta"
    mdl = xgb.XGBModel(**xgb_hyperparams)
    boruta = BorutaPy(estimator = mdl, n_estimators = 'auto', max_iter = max_iter)
    X = df.fillna(df.median()).drop(columns=target) # Dataset needs to be imputed for boruta
    y=df[target]
    boruta.fit(np.array(X), np.array(y))
    borutaDF = pd.DataFrame(zip(X.columns,list(boruta.ranking_)), columns = ['Feature', 'Ranking']).sort_values('Ranking')
    boruta_d = borutaDF.groupby('Ranking')['Feature'].apply(list).to_dict() 
    orderedFeatures_initial = borutaDF.Feature.to_list()
    
    all_aurocs = []
    n_features = list(exponential_spacing(stop=len(orderedFeatures_initial), n_steps=n_steps))[::-1]
    n_features2remove = np.diff(n_features)
    all_features = copy.deepcopy(orderedFeatures_initial)
    features_removed = []
    for f,n_feats in tqdm.notebook.tqdm(enumerate(n_features), total = len(n_features), disable=False, desc='Boruta FE'):    
        if f!=0:
            features_removed.append(all_features[n_features2remove[f-1]:])
            all_features= all_features[:n_features2remove[f-1]]
        curr_step = xgb_shap(df = pd.concat([df[all_features], df[target]],axis=1), target=target, max_evals=max_evals, verbose=verbose, xgb_hyperparams=xgb_hyperparams)
        if max_evals > 0:
            if verbose: 
                print("Step:", f, "n_feats", n_feats, "Tuning current...")
            curr_step.tune_model(verbose=False) 
        curr_auroc = curr_step.model_performance(verbose=False)
        all_aurocs.append(curr_auroc)
    mean_auroc = np.array(all_aurocs).mean(axis=1)
    optimal_features = orderedFeatures_initial[:n_features[np.where(mean_auroc==np.max(mean_auroc))[0][0]]]
    
    return dict(FE_method = FE_method,
                mean_auroc = mean_auroc,
                std_auroc = np.array(all_aurocs).std(axis=1),
                n_features = n_features, 
                features_removed = features_removed,
                optimal_features = optimal_features)

class feature_elimination(analysis):
    def __init__(self, df, target, max_evals=25, xgb_hyperparams=None, keep_features=[], verbose=True, outputs_dir='./', remove_outliers=False): 
        analysis.__init__(self, df, target=target, remove_outliers=remove_outliers)
        self.max_evals=max_evals
        self.xgb_hyperparams = xgb_hyperparams
        self.n_folds_SHAP=10
        self.SHAP_based_FE_outputs = "call 'run_SHAP_based_FE()' first"
        self.SHAP_based_RFECV_outputs = "call 'run_SHAP_based_RFECV()', first" 
        self.Boruta_based_FE_outputs = "call 'run_boruta_FE()', first"
        self.keep_features = keep_features
        self.verbose = verbose
        self.outputs_dir = outputs_dir

    def tune_model(self, verbose=True):
        start=time.time()
        def objective_model_curr(params, train_x=self.X, train_y=self.y):
            return objective_model(params, train_x=train_x, train_y=train_y)
        xgb_hyperparams = optimize_xgb(objective_model_curr, verbose=verbose, max_evals=self.max_evals)
        xgb_hyperparams.update({"objective": "binary:logistic", 'tree_method':"exact", "eval_metric":"error"})
        end=time.time()
        if verbose:
            print("Done: took", (end-start), "seconds")
            print("The best hyperparameters are: ", "\n")
            print(xgb_hyperparams)
            setattr(self, 'xgb_hyperparams', xgb_hyperparams)
        n_out = nargout()
        if n_out == 1:
            return xgb_hyperparams
        
    # SHAP analysis
    @LazyProperty
    def shap_values_prob(self):
        SHAP_outputs = SHAP_CV(df=self.df, mdlFeatures=self.mdlFeatures, target=self.target, xgb_hyperparams=self.xgb_hyperparams, nFolds=self.n_folds_SHAP)
        setattr(self, 'AUROC_CV_SHAP', SHAP_outputs['CV_AUROC'])
        setattr(self, 'predictionsCV_SHAP', SHAP_outputs['valPredictions_vec'])
        setattr(self, 'SHAP_outputs', SHAP_outputs)
        setattr(self, 'meanExpProb', SHAP_outputs['meanExpProb'])
        return SHAP_outputs['shap_values_prob']

    @LazyProperty
    def shapDF_prob(self):
        shapDF_prob = pd.DataFrame(self.shap_values_prob, columns=self.mdlFeatures)
        shapDF_prob['meanExpProb'] = self.meanExpProb
        return shapDF_prob
    
    @LazyProperty
    def orderedFeatures_initial(self):
        """ Order of importance based on SHAP analysis"""
        print("Calculating feature importance on full model")
        if self.xgb_hyperparams == None: 
            self.tune_model()
        vals= np.abs(self.shap_values_prob).mean(0) # Ordered by average |SHAP value| 
        ordered_Features = list(pd.DataFrame(zip(self.mdlFeatures,vals)).sort_values(1, ascending=False).reset_index(drop=True)[[0]].values.flatten())
        return ordered_Features
    
    def initial_shap_summary_plots(self, show=True, figsize=(40,12), max_display=30, save_fig=False): 
        fig = plt.figure(figsize=figsize)
        #fig.subplot(1,2,1)
        gs = fig.add_gridspec(1, 3)
        fig.add_subplot(gs[0, 0])
        shap.summary_plot(self.shap_values_prob, self.df[self.mdlFeatures], max_display=max_display, show=False, plot_size=(figsize[0]/3,figsize[1]), plot_type='bar')
        #plt.subplot(1,2,2)
        fig.add_subplot(gs[0, 1:])
        shap.summary_plot(self.shap_values_prob, self.df[self.mdlFeatures], max_display=max_display, show=False, plot_size=(2*figsize[0]/3,figsize[1]))
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
        out = SHAP_based_FE(df=self.df, target=self.target, orderedFeatures_initial= self.orderedFeatures_initial, n_steps=n_steps, xgb_hyperparams=self.xgb_hyperparams, verbose=verbose)
        setattr(self, 'SHAP_based_FE_outputs', out)
        return out
    
    def run_SHAP_based_RFECV(self, n_steps=30, max_evals=25, verbose=False):
        out = SHAP_based_RFECV(df=self.df, target=self.target, orderedFeatures_initial= self.orderedFeatures_initial, n_steps=n_steps, max_evals=max_evals, verbose=verbose, xgb_hyperparams=self.xgb_hyperparams)
        setattr(self, 'SHAP_based_RFECV_outputs', out)
        return out
    
    def run_boruta_FE(self, n_steps=30, max_iter=100, max_evals=0, verbose=False):
        if self.xgb_hyperparams == None:
            self.tune_model(verbose=False)
        out = boruta_FE(df=self.df, target=self.target, xgb_hyperparams=self.xgb_hyperparams, max_iter=max_iter, n_steps=n_steps, max_evals=max_evals, verbose=verbose)
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
                curr_plot = ax.plot(out['n_features'], out['mean_auroc'], '-o', label = out['FE_method'])
                color= curr_plot[0].get_color()
                ax.fill_between(out['n_features'], out['mean_auroc']-out['std_auroc'], out['mean_auroc']+out['std_auroc'], alpha = .2, label=out['FE_method'])  
                # Optimal lines:
                ax.vlines(x=len(out['optimal_features'])- .1*t, ymin =.5 , ymax=np.mean(out['mean_auroc'] - 1.5*out['std_auroc']), linestyles='--', lw = 3, color=color)
                for idx, x in enumerate(out['n_features']):
                    if idx%2==1:
                        ax.text(x, out['mean_auroc'][idx] - .4*t*out['std_auroc'][idx], "{:.3f}±{:.2f}".format(out['mean_auroc'][idx], out['std_auroc'][idx]), 
                                size=16, ha='right', va='top', color=color, rotation=45)

            ax.invert_xaxis()
            ax.legend()
            ax.set_ylabel('AUROC (CV)')

            plt.title("Comparison of feature elimination methods")
            #plt.ylim([.5,.85])
            plt.grid(axis='both',ls='--')
            if save_fig:
                file_name= outputs_dir + "Comparison of FE methods {}.png".format(self.target)
                plt.savefig(file_name)
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
        else: 
            raise ValueError("Must run feature elimination method(s) first")
            

        if importance_sort:
            if self.verbose: 
                print("Sorting features by importance")
            mdlVars = copy.deepcopy(union_features)
            mdlVars.append(self.target)
            curr_analysis = xgb_shap(df=self.df[mdlVars], target=self.target, max_evals=25)
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
            curr_analysis = xgb_shap(df=self.df[mdlVars], target=self.target, max_evals=25)
            curr_analysis.tune_model()
            intersect_features = curr_analysis.orderedFeatures
            setattr(self, 'intersect_features', intersect_features)
        return intersect_features

    def run_forward_selection(self, features=None, verbose=True, n_folds=5, n_reps=5):
        from scipy.stats import sem
        if type(features) == type(None):
            init_list = copy.deepcopy(self.union_features)
        else:
            init_list = features
        allMdlVars = copy.deepcopy(init_list)
        allMdlVars.append(self.target)
        init_analysis = xgb_shap(df=self.df[allMdlVars], target=self.target, max_evals=25, n_folds_SHAP=10)
        init_analysis.tune_model()
        xgb_hyperparams = init_analysis.xgb_hyperparams
        # xgb_hyperparams = {'eta': 0.08725904914699555, 'max_depth': 3, 'min_child_weight': 0.023737374638472843, 'reg_alpha': 0.897750469255347, 'reg_lambda': 0.04481058665391393, 'subsample': 0.6851140206115516, 'objective': 'binary:logistic', 'tree_method': 'exact', 'eval_metric': 'error'}

        # include_opts = ['Not included', 'Tentative', 'Included']
        # Features are included if they significantly increase model performance. 
        # Tentative inclusion if they cause no change or increase model performance slightly
        # Not included if they decrease model performance

        prev_mean = .5
        prev_inc_limit = .5
        currMdlVars = [self.target]
        curr_list_d = {k:v for k,v in enumerate(init_list)}
        steps = []
        exculded_features = []; included_features=[]; tentative_features=[];
        step_outcomes = []
        mean_scores = []
        std_scores = []
        sem_scores = []
        # For loop
        for f in tqdm.notebook.tqdm(range(len(init_list)), disable=False, desc='Forward selection'): # defines the initial order we'll do the step_wise selection
            curr_feat = curr_list_d[f]
            currMdlVars.append(curr_feat)
            if verbose:
                print(f"Step {f} :{curr_feat}\nTesting {len(currMdlVars)-1} feature(s) currently")
            curr_analysis = xgb_shap(df=self.df[currMdlVars], target=self.target, xgb_hyperparams=xgb_hyperparams)
            curr_scores = curr_analysis.model_performance(n_folds=n_folds, n_reps=int(n_reps), verbose=verbose)
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
                included_features.append(curr_feat)
                # currMdlVars = currMdlVars
                # curr_list = curr_list
            elif mean_score >= prev_mean: # Tentative
                steps.append("+ "+curr_feat + "*")
                include="Tentatively included"
                prev_mean = mean_score
                prev_inc_limit = mean_score + sem_score
                tentative_features.append(curr_feat)
                # currMdlVars = currMdlVars
                # curr_list = curr_list
            elif mean_score < prev_mean: # excluded
                steps.append("- "+curr_feat)
                include="Excluded"
                exculded_features.append(currMdlVars.pop())
                # Recalculate feature importance and
                def update_curr_list_d(curr_list_d, currMdlVars, exculded_features, init_list, f, xgb_hyperparams=xgb_hyperparams):
                    currMdlFeatures = copy.deepcopy(currMdlVars)
                    currMdlFeatures.pop(currMdlFeatures.index(self.target))
                    viableMdlVars = list(set(allMdlVars).difference(exculded_features))
                    tmpAnalysis= xgb_shap(df=self.df[viableMdlVars], target=self.target, xgb_hyperparams=xgb_hyperparams, n_folds_SHAP=10, verbose=verbose)
                    remaining_features = tmpAnalysis.orderedFeatures
                    for feat in currMdlFeatures:
                        remaining_features.pop(remaining_features.index(feat));
                    curr_list_d.update({k+f+1:v for k,v in enumerate(remaining_features)})
                    return
                update_curr_list_d(curr_list_d, currMdlVars, exculded_features, init_list, f, xgb_hyperparams=xgb_hyperparams)
            step_outcomes.append(include)
            if verbose:
                print(f"Step {f} outcome: {include} {curr_feat} \n")
        mdl_vars_forward_selection = currMdlVars
        print("Features selected by forawrd selection :\n", mdl_vars_forward_selection)
        df_d = dict(step = steps, mean_auroc = mean_scores,
                    std_auroc = std_scores, sem_auroc = sem_scores,
                    outcome = step_outcomes)

        forward_selection_results_df = pd.DataFrame(df_d)
        setattr(self, 'forward_selection_results', forward_selection_results_df)
        setattr(self, 'mdl_vars_forward_selection', mdl_vars_forward_selection)
        setattr(self, 'outputs_forward_selection', dict(included_features = included_features, 
                    tentative_features = tentative_features, 
                    exculded_features = exculded_features,
                    n_reps = n_reps,
                    n_folds = n_folds))
        self.plot_forward_selection(save_fig=True)
        return forward_selection_results_df

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
                            out['mean_auroc'], '-o', color='gray', alpha=.2, label = "forward selection")

        ax.fill_between(np.array(self.forward_selection_results.index)+1, 
                        out['mean_auroc']-out['std_auroc'], 
                        out['mean_auroc']+out['std_auroc'], color='gray', alpha = .2, label='STD')  
            
        for idx, x in enumerate(self.forward_selection_results.index+1):
            ax.text(x, out['mean_auroc'][idx] - 1.1* out['std_auroc'][idx], "{} {:.3f}±{:.2f}".format(out['step'][idx], out['mean_auroc'][idx], out['std_auroc'][idx]), 
                    size=16, ha='center', va='top', color='k', rotation=90)
        for curr_outcome, color in zip(['Included', 'Tentatively included', 'Excluded'],['g', 'yellow', 'r']):
            idxs = [idx+1 for idx,outcome in enumerate(out['outcome']) if (outcome==curr_outcome) ]
            mean_aurocs = [out['mean_auroc'][idx] for idx,outcome in enumerate(out['outcome']) if (outcome==curr_outcome) ]
            ax.scatter(x=idxs, y=mean_aurocs, color=color, label = curr_outcome, s=100)
            
        ax.legend(framealpha=.2, bbox_to_anchor=(1.2, .5), loc=5)
        ax.set_ylabel('AUROC (CV)')
        ax.set_xlabel('Step #')
        ax.set_xlim(left=0)
        n_folds = self.outputs_forward_selection['n_folds']
        n_reps = self.outputs_forward_selection['n_reps']
        plt.title(f"Forward selection results\n using {n_folds} fold CV with {n_reps} reps")
        plt.ylim([.1,.95])
        plt.grid(axis='both',ls='--')
        if save_fig:
            file_name= outputs_dir + "Forward selection results {}.png".format(self.target)
            plt.savefig(file_name)
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
                mean_aurocs = []
                std_aurocs = []
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
                    curr_analysis = xgb_shap(df=self.df[mdlVars], target=self.target, max_evals=max_evals)
                    curr_analysis.tune_model()
                    curr_scores = curr_analysis.model_performance(n_folds=n_folds, n_reps=int(n_reps), verbose=verbose, seed=2) # seed for forward selection was 0, so we are getting an independent assessment of perfomance here
                    mean_score, std_score = np.mean(curr_scores), np.std(curr_scores)
                    mean_aurocs.append(mean_score)
                    std_aurocs.append(std_score)

                performance_comparison_outputs = dict(models=models, 
                                            mean_aurocs = mean_aurocs, 
                                            std_aurocs = std_aurocs,
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
            label = model + "\n{:.3f}±{:.2f}".format(performance_comparison_outputs['mean_aurocs'][idx], performance_comparison_outputs['std_aurocs'][idx])
            plt.errorbar(x=idx, y=performance_comparison_outputs['mean_aurocs'][idx], 
                        yerr=performance_comparison_outputs['std_aurocs'][idx], 
                        label=label, fmt='ok')
            labels.append(label)
        plt.xticks(range(len(performance_comparison_outputs['models'])), labels=labels, rotation=45, ha='right')
        n_folds = performance_comparison_outputs['n_folds']
        n_reps = performance_comparison_outputs['n_reps']
        plt.title(f"Model comparison\n{n_folds} fold-CV repeated {n_reps}")
        ylims = (np.mean(performance_comparison_outputs['mean_aurocs']) - 2* np.mean(performance_comparison_outputs['std_aurocs']),
            np.mean(performance_comparison_outputs['mean_aurocs']) + 2* np.mean(performance_comparison_outputs['std_aurocs']))
        
        plt.yticks(np.linspace(.5,1,21))
        plt.ylim(list(ylims))
        plt.ylabel('AUROC (CV)')
        plt.grid(axis='both',ls='--', alpha=.2)
        if outputs_dir == None:
            outputs_dir = self.outputs_dir
        else: 
            outputs_dir=outputs_dir
        if save_fig:
            file_name= outputs_dir + "Forward selection results {}.png".format(self.target)
            plt.savefig(file_name, bbox_inches='tight')
            print("Saved: ", file_name)
        if return_fig: 
            return fig
        else: 
            return ax

    @LazyProperty
    def union_features(self): 
        return self.get_union_features(importance_sort=True)

    @LazyProperty
    def intersect_features(self): 
        return self.get_intersect_features(importance_sort=True)

    def save(self, name="FE"): 
        import time, pickle
        fileName = self.outputs_dir + "_".join([self.target, name,time.asctime()]) +".p"
        pickle.dump(self, open(fileName, "wb"))
        print("Saving FE as: ", fileName)

