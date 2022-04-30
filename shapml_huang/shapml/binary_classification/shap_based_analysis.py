from operator import ge
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time, itertools, copy, tqdm

# from multiprocessing import Pool, current_process, cpu_count #Value, Array
# from random import choices, seed

from collections import Counter

import xgboost as xgb
import shap
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split, KFold #StratifiedKFold

import hyperopt as hp
from hyperopt.pyll.base import scope
from hyperopt.pyll.stochastic import rng_from_seed
from hyperopt import STATUS_OK, fmin, hp, tpe, SparkTrials
from hyperopt.early_stop import no_progress_loss

from ..bootstrap_shap.bootstrap_shap_analysis import parallel_bootstrap_shap_analysis, bootstrap_SHAP_analysis
from ..utils.helpers import LazyProperty, nargout
from ..utils.cutoff_analysis_fcns import cutoff_analysis
from ..utils.superclasses import analysis, bootstrap_summary, shap_analysis
from ..utils.simulations import generate_synthetic_binary_classification_df
from ..utils.misc import add_annotations
from .bokeh_plots import binary_classificaiton_plots, plot_ROC


from collections import defaultdict
import seaborn as sns

def score_model(params, train_x, train_y, n_folds = 5, test_x = None, test_y=None, n_reps =1, verbose = True, return_val_predictions=False, return_train_scores=False, seed = 0, matrix_format=False): # 
	val_scores = []
	train_scores = []
	y_vals = np.empty((len(train_y),n_folds*n_reps))
	y_vals[:] = np.nan
	y_trains = y_vals.copy() # Deep copies
	predictionsVal = y_vals.copy()
	predictionsTrain = y_vals.copy()
	
	if n_reps > 10000: 
		warnings.warn("n_reps is very high, you may start overlapping random seeds now")
	for r in range(n_reps):
		kf = KFold(n_splits=n_folds, shuffle=True, random_state=r+seed*10000) 
		i = 0
		for train_index, val_index in kf.split(train_x): 
			X_tr, X_val = train_x.iloc[train_index], train_x.iloc[val_index]
			y_tr, y_val = train_y.iloc[train_index], train_y.iloc[val_index]
			try:
				if y_val.sum() < 3: 
					print("Warning less than 3 positive targets. Moving to next iteration")
					continue 
			except:
				print("This can happen if there are multiple target columns")
				import pdb; pdb.set_trace()
			dtrain = xgb.DMatrix(X_tr, label=y_tr)
			dval   = xgb.DMatrix(X_val, label=y_val)
			xgb_model = xgb.train(params, dtrain, num_boost_round)

			currPredictions_val = xgb_model.predict(dval)
			currPredictions_train = xgb_model.predict(dtrain)
			predictionsVal[val_index, i+r*n_folds] = currPredictions_val
			predictionsTrain[train_index, i+r*n_folds] = currPredictions_train
			y_vals[val_index, i+r*n_folds] = y_val
			y_trains[train_index, i+r*n_folds] = y_tr

			currScore_val = roc_auc_score(y_val.to_numpy(),currPredictions_val)
			currScore_train = roc_auc_score(y_tr.to_numpy(),currPredictions_train)
			val_scores.append(currScore_val)
			train_scores.append(currScore_train)
			i+=1
			
	if verbose: 
		print("-- Model performance:\n {}-fold cross validation repeated {} times".format(n_folds,n_reps))
		print(" Score (val)   : {:.3f}±{:.2f}".format(np.mean(val_scores), np.std(val_scores)))
		print(" Score (train) : {:.3f}±{:.3f}".format(np.mean(train_scores), np.std(train_scores)))
		
	if (type(test_x) != type(None)) & (type(test_y) != type(None)):
		dtrain = xgb.DMatrix(train_x, label=train_y)
		dval   = xgb.DMatrix(test_x, label=test_y)
		xgb_model = xgb.train(params, dtrain, num_boost_round)
		predictions_all = xgb_model.predict(dval)
		test_score = roc_auc_score(test_y.to_numpy(),predictions_all)
		if verbose: 
			print(" Score (Holdout)  : {:.3f}".format(test_score))
	
	if (type(test_x) != type(None)) & (type(test_y) != type(None)):
		return val_scores, test_score
	elif return_val_predictions ==True: 
		if return_train_scores:
			return val_scores, np.nanmean(predictionsVal, axis=1), train_scores
		else: 
			if matrix_format: 
				return val_scores, np.nanmean(predictionsVal.reshape(-1,n_reps, n_folds), axis=2)
			else:
				return val_scores, np.nanmean(predictionsVal, axis=1)
	elif return_train_scores:
		return val_scores, train_scores
	else:
		return val_scores

def reliability_diagram(y_true, y_prob, analysis_name, figsize=(6,2.5), nBins = 10, ax=None, fig=None):
	"""
	If supplied ax should be a list of two axes for the two plots
	"""
	from scipy import stats
	plt.style.use('seaborn-talk')
	if type(ax) == type(None):
		fig, (ax1, ax2) = plt.subplots(1,2, figsize=figsize)
	else:
		(ax1,ax2) = ax
	error_bar_alpha=.05
	c='red'   
	x = y_prob
	y = y_true
	set_name =analysis_name

	bins = [0]
	bins.extend(np.linspace(np.quantile(x[~np.isnan(x)], q =.05), np.quantile(x, q =.95), nBins-1))
	bins.extend([1])
	digitized_x = np.digitize(x, bins)

	mean_count_array = np.array([[np.mean(y[digitized_x == i]),
								  len(y[digitized_x == i]),
								  np.mean(x[digitized_x==i])] 
								  for i in np.unique(digitized_x)])
	x_pts_to_graph = mean_count_array[:,2]
	y_pts_to_graph = mean_count_array[:,0]
	bin_counts = mean_count_array[:,1]
	
	ax1.plot(np.linspace(0,1,100),(np.linspace(0,1,100)),'k--')
	ax1.scatter(x_pts_to_graph,y_pts_to_graph, c=c)
	ax1.axis([-0.1,1.1,-0.1,1.1])
	yerr_mat = stats.binom.interval(1-error_bar_alpha,bin_counts,x_pts_to_graph)/bin_counts - x_pts_to_graph
	yerr_mat[0,:] = -yerr_mat[0,:]
	ax1.errorbar(x_pts_to_graph, x_pts_to_graph, yerr=yerr_mat, capsize=5)
	ax1.set_xlabel('Predicted')
	ax1.set_ylabel('Empirical')
	ax1.set_title('Reliability diagram\n {}'.format(set_name))

	ax2.hist(x,bins=bins, ec='w')
	ax2.set_title("Predicted probabilities\n{}".format(set_name))

	plt.tight_layout()
	if type(ax)==type(None):
		return fig
	else:
		return fig

# Hyperparameter tuning

SEED = 314159265
num_boost_round=100
n_folds_CV = 5 # for CV performance evaluation
n_folds_SHAP = 10 # for CV when doing SHAP analysis

def objective_model(params, train_x, train_y, b_coeff=0, n_folds=n_folds_CV):
	"""Objective function for XGBoost Hyperparameter Optimization"""   
	scores, predictionCV = score_model(params, train_x, train_y, n_folds = n_folds_CV, 
										 verbose=False, return_val_predictions = True) 
	y_prob = np.array(predictionCV).flatten()
	score = np.mean(scores) 
	loss = 1 - score + b_coeff*brier_score_loss(train_y, y_prob)
	# Dictionary with information for evaluation
	return {'loss': loss,  'status': STATUS_OK}

def optimize_xgb(score, random_state=SEED, verbose=True, max_evals = 25):
	"""
	This is the optimization function that given a space (space here) of 
	hyperparameters and a scoring function (score here), finds the best hyperparameters.
	"""
	# Optimal grid search parameters: 


	space = {
		'eta':                         hp.loguniform('eta', np.log(0.01), np.log(1)),
		'max_depth':                   scope.int(hp.quniform('max_depth', 2,5,1)),
		'min_child_weight':            hp.loguniform('min_child_weight', np.log(0.01), np.log(10)),
		'reg_alpha':                   hp.loguniform('reg_alpha', np.log(0.2), np.log(10)), 
		'reg_lambda':                  hp.loguniform('reg_lambda', np.log(0.001), np.log(100)),#Was 10 on 102721
		'subsample':                   hp.uniform('subsample', 0.6, 1),
		"objective": "binary:logistic", 
		'tree_method':"exact",
		'eval_metric':"error"
	}
	# print(space)	
	# Use the fmin function from Hyperopt to find the best hyperparameters
	rstate = np.random.RandomState(SEED)
	
	best = fmin(score, space, algo=tpe.suggest, max_evals=max_evals, 
				verbose=verbose,rstate=rstate, 
				early_stop_fn=no_progress_loss(iteration_stop_count=50, percent_increase=0)) #, trials = SparkTrials(parallelism = 4))
	for parameter_name in ['max_depth']:
		best[parameter_name] = int(best[parameter_name])
	return best

def SHAP_CV(df, mdlFeatures, target, hyperparams, nFolds, verbose=True, generate_interaction_vals=True, synthetic_rows=0): 
	if verbose: 
		print(f"Extracting SHAP values using {nFolds}-fold CV ...")
	start = time.time()
	train_scores = []
	models_d = {}
	explainer_prob_d = {}
	explainer_logit_d = {}
	expectedLogit_d={}
	expectedProb_d={}

	y_true = []
	predictionsTrain = np.empty((df.shape[0],nFolds))
	predictionsTrain[:] = np.nan

	y_vals = predictionsTrain.copy()
	y_vals[:] = np.nan
	y_trains = y_vals.copy()
	predictionsVal = y_vals.copy()

	kf = KFold(n_splits=nFolds)
	i = 0 
				
	for train_index, val_index in tqdm.tqdm_notebook(kf.split(df), total=nFolds, desc='SHAP values', disable=(not verbose)): 
		X_tr, X_val = df[mdlFeatures].iloc[train_index], df[mdlFeatures].iloc[val_index]
		y_tr, y_val = df[target].iloc[train_index], df[target].iloc[val_index]
		y_true.extend(y_val)
		if synthetic_rows > 0:
			synthDF = generate_synthetic_binary_classification_df(df.iloc[train_index], y_tr.name, desired_rows=synthetic_rows, verbose=verbose)
			X_tr = pd.concat([X_tr, synthDF[X_tr.columns]])
			y_tr = pd.concat([y_tr, synthDF[y_tr.name]])

		dtrain = xgb.DMatrix(X_tr, label=y_tr)
		dval   = xgb.DMatrix(X_val, label=y_val)
		# import pdb; pdb.set_trace()	
		xgb_model = xgb.train(hyperparams, dtrain, num_boost_round)
		currPredictions_val = xgb_model.predict(dval)
		currPredictions_train = xgb_model.predict(dtrain)

	#     predictionsVal.extend(currPredictions_val)
		predictionsVal[val_index, i] = currPredictions_val
		if synthetic_rows == 0:
			predictionsTrain[train_index, i] = currPredictions_train
			y_trains[train_index, i] = y_tr
		y_vals[val_index, i] = y_val
		currScore_train = roc_auc_score(y_tr.to_numpy(),currPredictions_train)
		train_scores.append(currScore_train)

		# Calculate Shap values: 
		explainer_prob = shap.TreeExplainer(xgb_model,data=df.drop(columns=target),
											   feature_dependence="interventional", model_output="probability")
		shap_values_prob_curr = explainer_prob.shap_values(df[mdlFeatures].iloc[val_index])

		explainer_logit = shap.TreeExplainer(xgb_model, feature_perturbation="tree_path_dependent")
		shap_values_logit_curr = explainer_logit.shap_values(df[mdlFeatures].iloc[val_index])
		if generate_interaction_vals: 
			curr_shap_interaction_values = explainer_logit.shap_interaction_values(df[mdlFeatures].iloc[val_index])

		if i == 0: 
			shap_values_logit = shap_values_logit_curr
			shap_values_prob = shap_values_prob_curr
			if generate_interaction_vals:
				shap_interaction_values=curr_shap_interaction_values
		else:
			shap_values_logit = np.concatenate([shap_values_logit, shap_values_logit_curr])
			shap_values_prob = np.concatenate([shap_values_prob, shap_values_prob_curr])
			if generate_interaction_vals:
				shap_interaction_values = np.concatenate((shap_interaction_values, curr_shap_interaction_values), axis=0)    

		for v in val_index:
			models_d[v] = xgb_model
			explainer_logit_d[v] = explainer_logit
			explainer_prob_d[v] =explainer_prob
			expectedLogit_d[v] = explainer_logit.expected_value
			expectedProb_d[v] = explainer_prob.expected_value
		i+=1
	
	# Processing after For-loop
	meanExpProb = np.mean(list(expectedProb_d.values()))
	meanExpLogit = np.mean(list(expectedLogit_d.values()))

	CV_score = roc_auc_score(y_true,np.nanmean(predictionsVal, axis=1)) 
	if verbose:
		print("  CV score: {:.3f}".format(CV_score))
		print("  CV score (train): {:.3f}±{:.3f}".format(np.mean(train_scores), np.std(train_scores)))
		end = time.time()
		print("   Execution time: {:.2f}s".format(np.round(end - start,5)))

	valPredictions_vec = np.nansum(predictionsVal, axis=1) # Only 1 value in each row
	outputs_d = dict(models_d=models_d,
					 shap_values_logit=shap_values_logit,
					 shap_values=shap_values_prob, # default for this class is prob
					 explainer_d=explainer_prob_d, # default
					 expectedValues_d = expectedProb_d, # default
					 explainer_logit_d=explainer_logit_d,
					 expectedLogit_d=expectedLogit_d, 
					 meanExpValue=meanExpProb, # default for this class is prob
					 meanExpLogit =meanExpLogit, 
					 predictionsVal=predictionsVal,
					 valPredictions_vec=valPredictions_vec,
					 predictionsTrain=predictionsTrain, 
					 y_vals=y_vals, 
					 y_trains=y_trains,
					 CV_score=CV_score
					)
	if generate_interaction_vals: 
		outputs_d.update(dict(shap_interaction_values=shap_interaction_values))
	return outputs_d


class xgb_shap(analysis, bootstrap_summary, shap_analysis):
	def __init__(self, df, target, max_evals=50, hyperparams=None, n_folds_SHAP=20, bootstrap_iterations=np.round(1000*np.e), remove_outliers=False, verbose=True, outputs_dir='./', categorical_encoding='oridinal', generate_interaction_vals=True, **kwargs):
		"""
		Generates a binary classification XGBoost object with integrated SHAP analysis
		categorical_encoding='oridinal' ordered by the prevalance of subgroup
		generate_interaction_vals : defines defaul behavior when generating SHAP values (One can still generate interaction values later if desired)
		""" 
		analysis.__init__(self, df=df, target=target, remove_outliers=remove_outliers, outputs_dir=outputs_dir, **kwargs)
		bootstrap_summary.__init__()
		# shap_analysis.__init__() # Not sure if I need this
		self.max_evals = max_evals
		if hyperparams == None: 
			default_hyperparams = xgb.XGBClassifier().get_xgb_params()
			del default_hyperparams['n_jobs'] # This becomes and issue when running bootstrap analysis where n_jobs 
			default_hyperparams.update({'eval_metric': 'error'})
			self.hyperparams = default_hyperparams
		else:
			self.hyperparams = hyperparams
		
		self.score_CV_SHAP = "Call 'shap_values' first"
		self.predictionsCV_SHAP = "Call 'shap_values' first"
		self.SHAP_outputs = "Call 'shap_values' first"
		self.n_folds_SHAP=n_folds_SHAP
		self.meanExpValue = None
		self.bootstrap_SHAP_outputs = """Call 'run_bootstrap_SHAP_analysis_local' first. 
		run_bootstrap_SHAP_analysis_parallel does not support outputing models yet.
		"""
		self.bootstrap_iterations = bootstrap_iterations
		self.verbose = verbose 
		self.score_name = 'AUROC'
		self.generate_interaction_vals = generate_interaction_vals
		if 'exposure_var' not in self.__dir__():
			pass #print("No exposure_var was defined (optional).")
	
	def run_feature_selection_workflow(self, model_type='xgb_binary', keep_features=[], **kwargs):
		"kwargs: SHAP_FE=True, SHAP_RFECV=True, borutaFE=True, forward_selection=True, n_steps=30, verbose=False, n_reps_forward_selection=5, run_model_comparison_post_forward_selection=True"
		from ..feature_selection.feature_selection import feature_selection
		FS_obj = feature_selection(self.df, self.target, model_type=model_type, keep_features=keep_features)
		FS_obj.run_workflow(**kwargs)
		setattr(self, 'feature_selection', FS_obj)

	def tune_model(self, max_evals=None, verbose=True):      
		if max_evals==None: 
			max_evals = self.max_evals
		start=time.time()
		def objective_model_curr(params, train_x=self.X, train_y=self.y):
			return objective_model(params, train_x=train_x, train_y=train_y)
		hyperparams = optimize_xgb(objective_model_curr, verbose=verbose, max_evals=max_evals)
		hyperparams.update({"objective": "binary:logistic", 'tree_method':"exact", "eval_metric":"error"})
		end=time.time()
		if verbose:
			print("Done: took", (end-start), "seconds")
			print("The best hyperparameters are: ", "\n")
			print(hyperparams)
			setattr(self, 'hyperparams', hyperparams)
		n_out = nargout()
		if n_out == 1:
			return hyperparams

	def tune_model_extensive(self, max_evals=25, iters=25, verbose=False, min_viable_hyperparams=5):
		""" This takes longer, but it should  yield a set of hyperparamters that is not overfit (if that is a concern)"""
		from .hyperparameter_selection import hyperparameter_selection 
		hyp_search=hyperparameter_selection(df=self.df, target=self.target, 
									max_evals=max_evals, verbose=verbose, prediction_type="binary_classification",
									outputs_dir=self.outputs_dir)
		hyp_search.run_models(iters=iters)
		hyp_search.plot_CV_holdout_performance(figsize=(8,6))
		# Don't use depth that was selected <min_viable_hyperparams even if it had better performance
		hyperparams=hyp_search.get_optimal_hyperparameters(n_clusters=1, min_viable_hyperparams=min_viable_hyperparams)
		setattr(self, 'hyperparams', hyperparams)
		setattr(self, 'hyp_search', hyp_search)
		return hyperparams

	def model_performance(self, n_folds=n_folds_CV, n_reps=1, verbose=True, seed=0, return_predictions_matrix=False):
		if return_predictions_matrix == True: 			
			scores, y_pred_mat = score_model(self.hyperparams, self.X, self.y, n_folds = n_folds, n_reps=n_reps, verbose=verbose, seed=seed, return_val_predictions=True, matrix_format=True)
		else:
			scores = score_model(self.hyperparams, self.X, self.y, n_folds = n_folds, n_reps=n_reps, verbose=verbose, return_val_predictions=False, seed=seed)
		if verbose:
			print_str= "{} from {}-fold cross-validation ({} reps): ".format(self.score_name, n_folds, n_reps) + "{:.3f}±{:.2f}".format(np.mean(scores), np.std(scores))
			print(print_str)
		if return_predictions_matrix:
			return scores, y_pred_mat
		if nargout() == 2: 
			return scores, print_str
		return scores
		
	def plot_ROC(self, stratify=None, n_reps=5, **kwargs):
		return plot_ROC(self, stratify=stratify, n_reps=n_reps, **kwargs)

	def stratified_random_subsampling_validation(self, test_size=.2, n_iters=10):
		val_scores = []
		for i in range(n_iters):
			X_tr, X_val, y_tr, y_val = train_test_split(self.df[self.mdlFeatures], self.df[self.target], stratify=self.df[self.target], test_size =test_size, random_state=i)
			dtrain = xgb.DMatrix(X_tr, label=y_tr)
			dval   = xgb.DMatrix(X_val, label=y_val)
			xgb_model = xgb.train(self.hyperparams, dtrain, num_boost_round)
			currPredictions_val = xgb_model.predict(dval)
			currScore_val = roc_auc_score(y_val.to_numpy(),currPredictions_val)
			val_scores.append(currScore_val)
		return val_scores

	def plot_reliability(self, show=True, n_folds = n_folds_CV, stratify=None, nBins=10, n_reps=1):
		if type(stratify) == type(None): 
			scores, predictionsCV = score_model(self.hyperparams, self.X, self.y, n_folds = n_folds, n_reps=n_reps, verbose=False, return_val_predictions = True)
			fig = reliability_diagram(self.y, predictionsCV, analysis_name="XGBoost model", figsize=(10,4), nBins=nBins)
			fig.text(.1, 1, "{} from {}-fold cross-validation: ".format(self.score_name, n_folds) + "{:.3f}±{:.2f}".format(np.mean(scores), np.std(scores)), ha='left', fontsize = 18)
			if show: 
				pass
			else:
				plt.close()
				return fig
		else: 
			print(f"Stratified reliability diagram by {stratify}")
			fig, ax=plt.subplots(2,2, figsize=(10,8))
			scores=score_model(self.hyperparams, self.X, self.y, n_folds = n_folds,n_reps=n_reps, verbose=False)
			fig.text(.1, 1, "{} from {}-fold cross-validation and {} rep(s): ".format(self.score_name, n_folds, n_reps) + "{:.3f}±{:.2f}".format(np.mean(scores), np.std(scores)), ha='left', fontsize = 18)
			if 'meta_df' in self.__dir__():
				rel_df = pd.concat([self.df, self.meta_df], axis=1)
			else:
				rel_df = self.df.copy()
			unique_values = rel_df[stratify].unique()
			if len(unique_values) ==2:
				for ii,u in enumerate(unique_values):
					select_vec = (rel_df[stratify]==u)
					_, predictionsCV = score_model(self.hyperparams, self.X, self.y, n_folds = n_folds,n_reps=n_reps, verbose=False, return_val_predictions = True)
					score = roc_auc_score(self.y[select_vec], predictionsCV[select_vec])
					analysis_str = f"{stratify}=={str(u)} {self.score_name}:{np.round(score,3)}"
					fig = reliability_diagram(self.y[select_vec], predictionsCV[select_vec], analysis_name=analysis_str, figsize=(10,4), nBins=nBins, fig=fig, ax=ax[ii])
			elif len(unique_values)>2:
				med_val = rel_df[stratify].median()
				for ii in [0,1]:
					if ii ==0: 
						select_vec = (rel_df[stratify]<med_val)
						comparator = "<"
					elif ii==1:
						select_vec = (rel_df[stratify]>=med_val)
						comparator = "≥"						
					_, predictionsCV = score_model(self.hyperparams, self.X, self.y, n_folds = n_folds,n_reps=n_reps, verbose=False, return_val_predictions = True)
					score = roc_auc_score(self.y[select_vec], predictionsCV[select_vec])
					analysis_str = f"{stratify}{comparator}{med_val} {self.score_name}:{np.round(score,3)}"
					fig = reliability_diagram(self.y[select_vec], predictionsCV[select_vec], analysis_name=analysis_str, nBins=nBins, fig=fig, ax=ax[ii])	

			if show:	
				pass
			else:
				pass
				# plt.close()
			
			return fig

	def plot_reliability_SHAP(self, show=True, save_fig=False, outputs_dir=None, stratify=None, nBins=10):
		if self.predictionsCV_SHAP == "Call 'shap_values' first":
			self.shap_values;
		if type(stratify) == type(None):
			fig = reliability_diagram(self.y, self.predictionsCV_SHAP, analysis_name="XGBoost model", figsize=(10,4), nBins=nBins)
			fig.text(.1, 1, "{} (on validation sets in generation of SHAP values): {:.3f}".format(self.score_name, self.score_CV_SHAP), ha='left', fontsize = 18)
			if save_fig:
				if type(outputs_dir)==type(None): 
					outputs_dir = self.outputs_dir
				else:
					pass
				file_name= outputs_dir + "Reliability diagram {}.png".format(self.target)
				plt.savefig(file_name, bbox_inches='tight')
				print("Saved: ", file_name)
				return
			if show: 
				pass
			else:
				plt.close()
				return fig
		else: 
			fig, ax=plt.subplots(2,2, figsize=(10,8))
			fig.text(.1, 1, "{} (on validation sets in generation of SHAP values): {:.3f}".format(self.score_name, self.score_CV_SHAP), ha='left', fontsize = 18)
			print(f"Stratified reliability diagram by {stratify}")
			if 'meta_df' in self.__dir__():
				rel_df = pd.concat([self.df, self.meta_df],axis=1)
			else:
				rel_df = self.df.copy()
			unique_values = rel_df[stratify].unique()
			if len(unique_values) == 2:
				for ii, u in enumerate(unique_values):
					select_vec = (rel_df[stratify]==u)
					curr_score = roc_auc_score(self.y[select_vec], self.predictionsCV_SHAP[select_vec])
					analysis_str = f"{stratify}=={str(u)}\n{self.score_name}: {np.round(curr_score,3)}"
					fig = reliability_diagram(self.y[select_vec], self.predictionsCV_SHAP[select_vec], analysis_name=analysis_str, figsize=(10,4), nBins=nBins, fig=fig, ax=ax[ii])	
			elif len(unique_values)>2:
				med_val = rel_df[stratify].median()
				if med_val == 0:
					med_val+=.01
				for ii in [0,1]:
					if ii ==0: 
						select_vec = (rel_df[stratify]<med_val)
						comparator = "<"
					elif ii==1:
						select_vec = (rel_df[stratify]>=med_val)
						comparator = "≥"	

					curr_score = roc_auc_score(self.y[select_vec], self.predictionsCV_SHAP[select_vec])
					analysis_str = f"{stratify}{comparator}{str(med_val)}\n{self.score_name}: {np.round(curr_score,3)} (n={np.sum(select_vec)})"
					fig = reliability_diagram(self.y[select_vec], self.predictionsCV_SHAP[select_vec], analysis_name=analysis_str, figsize=(10,4), nBins=nBins, fig=fig, ax=ax[ii])
			else: 
				UserWarning("Could not properly stratify")
			
			plt.tight_layout()
			if show: 
				pass
			else:
				plt.close()
				return fig

	def plot_binary_classification_graphs(self, threshold=.5, return_layout=False): 
		layout = binary_classificaiton_plots(self.df[self.target], self.predictionsCV_SHAP, threshold=threshold)
		if return_layout==True:
			return layout
		else:
			from bokeh.io import show
			show(layout)

	def _generate_shap_values(self, synthetic_rows=0, verbose=False):
		SHAP_outputs = SHAP_CV(df=self.df, mdlFeatures=self.mdlFeatures, target=self.target, hyperparams=self.hyperparams, nFolds=self.n_folds_SHAP, verbose=self.verbose, generate_interaction_vals=self.generate_interaction_vals, synthetic_rows=synthetic_rows)
		setattr(self, 'score_CV_SHAP', SHAP_outputs['CV_score'])
		setattr(self, 'predictionsCV_SHAP', SHAP_outputs['valPredictions_vec'])
		setattr(self, 'SHAP_outputs', SHAP_outputs)
		setattr(self, 'meanExpValue', SHAP_outputs['meanExpValue'])
		setattr(self, 'shap_values', SHAP_outputs['shap_values'])
		if self.generate_interaction_vals:
			setattr(self, 'shap_interaction_values', SHAP_outputs['shap_interaction_values'])
		return SHAP_outputs['shap_values']

	@LazyProperty
	def shap_values(self):
		# SHAP_outputs = SHAP_CV(df=self.df, mdlFeatures=self.mdlFeatures, target=self.target, hyperparams=self.hyperparams, nFolds=self.n_folds_SHAP, verbose=self.verbose, generate_interaction_vals=self.generate_interaction_vals)
		# setattr(self, 'score_CV_SHAP', SHAP_outputs['CV_score'])
		# setattr(self, 'predictionsCV_SHAP', SHAP_outputs['valPredictions_vec'])
		# setattr(self, 'SHAP_outputs', SHAP_outputs)
		# setattr(self, 'meanExpValue', SHAP_outputs['meanExpValue'])
		# if self.generate_interaction_vals:
		# 	setattr(self, 'shap_interaction_values', SHAP_outputs['shap_interaction_values'])
		return self._generate_shap_values()

	@LazyProperty
	def shapDF_prob(self):
		shapDF_prob = pd.DataFrame(self.shap_values, columns=self.mdlFeatures)
		shapDF_prob['meanExpValue'] = self.meanExpValue
		return shapDF_prob

	@LazyProperty
	def shap_interaction_values(self):
		for pt_idx in tqdm.tqdm_notebook(range(self.df.shape[0])): #
			append_arr = self.SHAP_outputs['explainer_logit_d'][pt_idx].shap_interaction_values(self.df[self.mdlFeatures].iloc[[pt_idx]])
			if pt_idx == 0: 
				shap_interaction_values = append_arr
			else:
				shap_interaction_values = np.concatenate((shap_interaction_values, append_arr), axis=0)
		return shap_interaction_values

	def generate_shap_exposure_interaction_prob_df(self, exposure_var, n_perms=50):
		from ..binary_classification.logit2prob import convert_shapDF_logit_2_shapDF_prob
		first_order_shap_logit_df = pd.DataFrame(self.SHAP_outputs['shap_values_logit'], columns=self.mdlFeatures)
		exposure_interaction_df = pd.DataFrame(self.shap_interaction_values[:,self.mdlFeatures.index(exposure_var),:], 
											columns=self.mdlFeatures)
		first_order_exposure_interaction_logit_df=pd.concat([first_order_shap_logit_df.drop(columns=exposure_var), 
															exposure_interaction_df.add_prefix(exposure_var+':')], axis=1)
		first_order_exposure_interaction_logit_df['expectedValue'] = self.SHAP_outputs['meanExpLogit']
		first_order_exposure_interaction_prob_df = convert_shapDF_logit_2_shapDF_prob(first_order_exposure_interaction_logit_df, n_perms=n_perms)
		interaction_cols = [col for col in first_order_exposure_interaction_logit_df.columns if col.startswith(exposure_var+':')]
		shap_exposure_interaction_prob_df = first_order_exposure_interaction_prob_df[interaction_cols]
		shap_exposure_interaction_prob_df.columns = self.mdlFeatures
		setattr(self, 'shap_exposure_interaction_prob_df', shap_exposure_interaction_prob_df)
		print(f"Setting self.exposure_var to {exposure_var}")
		setattr(self, 'exposure_var', exposure_var)
		return shap_exposure_interaction_prob_df
	
	def cutoff_analysis(self, x, feature_thr=None, exposure_var = None, dose_var = None, 
		ylims =[0,.7], save_fig=False, return_fig=False, **kwargs):
		"""
		feature_thr defaults to median value
		kwargs: 
		dose_var = None, 
		trt_arm_name='Tx' 
        ms=10, mew=.5, 
		annotation_font_size=14, legend_font_size=8 
		y_label_offset_1 = .05, y_label_offset_2 = .05, 
		x_offset_1 = 0, x_offset_2 = 0, 
		figsize=(12,10)
		show_interaction=True, 
		"""
		if type(exposure_var) == type(None):
			exposure_var = self.exposure_var
		return cutoff_analysis(self, x=x, y=self.target, feature_thr = feature_thr, exposure_var = exposure_var, dose_var = dose_var, 
                    ylims =ylims , save_fig=save_fig, return_fig=return_fig, **kwargs)
	
	@LazyProperty
	def orderedFeatures(self):
		""" Order of importance based on SHAP analysis"""
		vals= np.abs(self.shap_values).mean(0) # Ordered by average |SHAP value| 
		ordered_Features = list(pd.DataFrame(zip(self.mdlFeatures,vals)).sort_values(1, ascending=False).reset_index(drop=True)[[0]].values.flatten())
		return ordered_Features

	@LazyProperty
	def bootsDF(self):
		"""If the models in each iteration are desired, 
		then run run_bootstrap_SHAP_analysis instead to avoid doubling computation time."""
		hyperparams = self.hyperparams
		bootsDF = parallel_bootstrap_shap_analysis(df=self.df, target=self.target, xgb_hyperparams=hyperparams, explainer_type = 'prob', 
			iterations=int(self.bootstrap_iterations), method='sample_with_replacement', train_size=.8, stratification_factor=None)
		return bootsDF

	def run_bootstrap_SHAP_analysis_local(self, bootstrap_iterations=None, method='sample_with_replacement'): 

		if bootstrap_iterations==None: 
			bootstrap_iterations = self.bootstrap_iterations

		bootsDF, out = bootstrap_SHAP_analysis(df=self.df, target=self.target, model=xgb.XGBClassifier(**self.hyperparams), explainer_type = 'raw', 
			iterations=bootstrap_iterations, method=method)
		setattr(self, 'bootstrap_SHAP_outputs', out['bootsDF'])
		setattr(self, 'bootsDF', bootsDF)
		return
	
	def run_bootstrap_SHAP_analysis_parallel(self, bootstrap_iterations=None, method='sample_with_replacement', train_size=0.8, stratification_factor=None): 
		"""
		This runs alternative bootstrap analyses
		method : 'train_test_split' or 'sample_with_replacement'
		running this saves the property 'bootsDF' to this class so
		it's available for plotting.
		"""
		hyperparams = self.hyperparams
		# xgb_hyperparams.update({'use_label_encoder':False})
		if bootstrap_iterations==None: 
			bootstrap_iterations = self.bootstrap_iterations
		bootsDF = parallel_bootstrap_shap_analysis(df=self.df, target=self.target, xgb_hyperparams=hyperparams, explainer_type = 'prob', 
			iterations=bootstrap_iterations, method=method, train_size=train_size, stratification_factor=stratification_factor)
		setattr(self, 'bootsDF', bootsDF)
		return bootsDF

	def plot_empirical_rates(self, summary_params=None, fig_width=10, plot_height=6, ymax=1, x_tick_label_size=12, save_fig=False,ms=4, fig_labels=True, nCols=3):
		if type(summary_params) == type(None):
			summary_params = self.summary_params
		# helper fcns: 
		def binomVec_yerr(vec):
			from statsmodels.stats.proportion import proportion_confint
			lwr, upr = np.abs(proportion_confint(vec.sum(), len(vec), method='wilson')-np.mean(vec))
			point_estimate = np.mean(vec)
			return np.round(point_estimate,4), np.round(lwr,4), np.round(upr,4)

		def binom_err(series):
			point_estimate, lwr, upr = binomVec_yerr(series)
			return [lwr,upr]

		def binom_CI(series):
			point_estimate, lwr, upr = binomVec_yerr(series)
			return [np.round(point_estimate-lwr,3), np.round(point_estimate+upr,3)]

		def prob_plot_advanced(x, y, df, nQuantiles=4, bins = None, ax=None, figsize=(6,4), ylims=None, return_errDF_only=False, categorical_mapping=None, x_tick_label_size=x_tick_label_size, ms=ms):
			""" plot binary probability vs. continuous variable discretized into q quantiles"""
			tmpDF = df[[x, y]]
			if type(bins) == str:
				if bins=='binary':
					tmpDF['feature_bin'] = df[x]
					classes = [str(0), str(1)]
				elif bins=='categorical':
					tmpDF['feature_bin'] = tmpDF[x].replace(categorical_mapping[x])
					tmpDF=tmpDF[tmpDF['feature_bin']!=-1]
					classes = list(categorical_mapping[x].values())
			else:
				if len(list(df[x].unique())) <7:
					classes = [str(v) for v in sorted(list(df[x].unique()))]
					tmpDF['feature_bin'] = pd.Categorical(df[x])
				else:
					if type(bins) == list:
						tmpDF['feature_bin'] = pd.cut(df[x], bins = bins, include_lowest=True, duplicates = 'drop')
					elif ((df[x] == df[x].min()).sum()/df[x].shape[0]) > .025: # If greater than 2.5% of the data is the lowest value: Make a bin for the LoQ/Zero value
						lowest_bin_zero = True
						lowest_binEdge = (df[x].min() == df[x]).sum()/df[x].notna().sum()
						bins = np.append(np.array([0]), np.linspace(lowest_binEdge,1, nQuantiles)) 
						tmpDF['feature_bin'] = pd.cut(df[x], bins = [df[x].quantile(q) for q in bins], include_lowest=True, duplicates = 'drop')
					else:
						bins=[df[x].quantile(q) for q in np.linspace(0,1, nQuantiles+1)]
						bins[0] = bins[0]-.1
						tmpDF['feature_bin'] = pd.cut(df[x], bins = bins, include_lowest=True, duplicates = 'drop')
					classes = tmpDF['feature_bin'].cat.categories.astype(str).to_list() # sorted list
			classes.append('nan')
			try:
				errDF = tmpDF.groupby('feature_bin').agg({y:['mean', binom_err, binom_CI, 'count']}).reset_index()
			except:
				import pdb; pdb.set_trace()
			errDF.columns = ['feature_bin', 'mean', 'binom_err', 'binom_CI','count']
			errDF['feature']=x
			errDF=errDF[['feature', 'feature_bin', 'mean', 'binom_err', 'binom_CI','count']]
			errDF['median_x_val'] = tmpDF.groupby('feature_bin').median()[x].values
			errDF.sort_values('median_x_val', inplace=True)
			if return_errDF_only:
				return errDF
			if ax == None: 
				fig,ax = plt.subplots(1,1, figsize=figsize)
			for idx in range(errDF.shape[0]):
				currDF = errDF.iloc[idx,:]
				x_val = currDF['median_x_val']#currDF[x+'_binned'].mid 
				y_val= currDF['mean']
				yerr = np.array(currDF['binom_err']).reshape(-1,1)
				ax.errorbar(x=x_val, y=y_val, yerr=yerr, marker='o', ms=ms, mew=4, color='gray')
			ax.set_xlabel(x+' (binned)')
			ax.set_ylabel(y+ ' rate')
			ax.set_title(y+ ' rate\nvs. '+ x)
			if type(ylims) != type(None):
				ax.set_ylim(ylims)
			if bins == 'categorical':
				ax.set_xticks(sorted(tmpDF[x].unique()))
				x_tick_labels = [categorical_mapping[x][n] for n in sorted(tmpDF[x].unique())]
				try:
					x_tick_labels=["\nOR ".join(l.split(" OR ")) for l in x_tick_labels]
				except:
					pass
				ax.set_xticklabels(x_tick_labels, rotation=30, ha='right', fontsize=x_tick_label_size)        
			else: 
				ax.set_xticks(sorted(errDF['median_x_val'].unique()))  
				ax.set_xticklabels(errDF['feature_bin'].cat.categories.astype(str), rotation=30, ha='right', fontsize=x_tick_label_size)
			
			return ax, errDF

		# Actual plotting:
		if type(summary_params) == type(None): 	
			summary_params = {v: {'nQuantiles': 4} for v in self.orderedFeatures}
			print('For now, genrating plots using summary_params=', summary_params)
		n_features = len(summary_params)
		summary_df = pd.DataFrame()
		fig=plt.figure(figsize=(fig_width,plot_height*np.ceil(n_features/float(nCols))))
		for a,(k,v) in enumerate(summary_params.items()):
			curr_ax = fig.add_subplot(int(np.ceil(n_features/nCols)),nCols,a+1)
			v=copy.deepcopy(v)
			if 'bins' in v:
				bins = copy.deepcopy(v['bins'])
				del v['bins']
			else:
				bins='categorical' if k in self.cat_cols else None
			
			ax, errDF= prob_plot_advanced(x=k, y=self.target, df=self.df,
					return_errDF_only=False, ax=curr_ax, **v, ylims=[0,ymax],
					bins=bins, categorical_mapping=self.categorical_mapping)
			
			summary_df = pd.concat([summary_df, errDF])
			ax.hlines(y=self.df[self.target].mean(), xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], ls='--')
			ax.set_yticks(np.arange(0,ymax+.05, .05))
			ax.grid(alpha=.5, axis='y')
		plt.tight_layout()
		if fig_labels:
			add_annotations(fig, fontsize=20)
		if save_fig:
			file_name= self.outputs_dir + f"Covariate associations with {self.target}.png"
			plt.savefig(file_name, bbox_inches='tight', dpi=300)
			print("Saved: ", file_name)
		return fig, summary_df

	def run_workflow(self):
		if 'learning_rate' in self.hyperparams: # if tuned it will have eta not learning rate
			self.tune_model()
		self.shap_values;
		self.bootsDF;
		if 'bootstrap_summaryDF' not in self.__dir__():
			self.generate_bootstrap_summaryDF();
		if ('exposure_var' in self.__dir__()) & ('shap_exposure_interaction_prob_df' not in self.__dir__()) & (self.hyperparams['objective']=='binary:logistic'):
			convert = input(f"Do you want to convert {self.exposure_var} interaction values to probability scale? (Y/N)")
			if convert.lower() == 'y':
				self.generate_shap_exposure_interaction_prob_df(self.exposure_var)
		return

	def interact(self):
		"""
		Launches streamlit app to explore ML-based insights 
		"""
		self.run_workflow()
		self.save(name='streamlit', include_time=False)
		try:
			assert 'shapml' in locals() # shapml is imported
		except:
			import shapml
		import os
		if self.hyperparams['objective'] == 'binary:logistic':
			app_path = os.path.join(os.path.dirname(shapml.__file__), 'utils/streamlit/binary_classification_app.py')
		elif self.hyperparams['objective'] == 'reg:squarederror': # TODO: modifiy app template 
			app_path = os.path.join(os.path.dirname(shapml.__file__), 'utils/streamlit/binary_classification_app.py')
		elif self.hyperparams['objective'] == 'survival:cox': # TODO: modifiy app template
			app_path = os.path.join(os.path.dirname(shapml.__file__), 'utils/streamlit/binary_classification_app.py')
		requirements_path = os.path.join(os.path.dirname(shapml.__file__), 'utils/streamlit/requirements.txt')
		helper_path = os.path.join(os.path.dirname(shapml.__file__), 'utils/streamlit/lib/*.py')
		cmd = "cp " + app_path + " " + os.path.join(os.getcwd(), 'app.py')
		assert os.system(cmd)==0 # was able to copy app.py
		cmd = "cp " + requirements_path + " " + os.path.join(os.getcwd(), 'requirements.txt')
		assert os.system(cmd)==0 # was able to copy requirements file

		try: # Create new lib folder with streamlit functions
			assert(os.system('mkdir lib')==0)
		except:
			assert(os.system('rm -rf lib')==0)
			assert(os.system('mkdir lib')==0)
		
		import yaml # Saving model's metadata to lib
		yaml.safe_dump(dict(outputs_dir = self.outputs_dir,target = self.target), open('./lib/app.yaml', 'w'))	

		cmd = "cp " + helper_path + " " +  os.path.join(os.path.curdir, 'lib/')
		assert os.system(cmd)==0 # was able to streamlit functions 

		print("""
		To share the app, navigate to the working directory and run: 
		rsconnect deploy streamlit --server https://connect.apollo.roche.com/ --api-key <YOUR_API_KEY (32 characters)> ./
		""")
		cmd = "streamlit run app.py"
		assert os.system(cmd)==0 # app was able to run

### Additional post hoc analyses
def shapModelComparison(dfs, target, hyperparams, analysisName='', save_fig=False, outputs_dir='./', n_folds_SHAP=20, verbose=False):
		"""
		dfs : list of dataframes
			  Features will be ordered by importance of first df

		analysisName: Give a unique name. This will be utilized to save files and load cached outputs.

		"""
		allVars = list(set(np.concatenate([list(dfs[m].columns) for m in range(len(dfs))]).flat)) # 
		allVars.pop(allVars.index(target))
		ModelComp_d = {}
		orderedFeatures_d = {}
		shapDF_prob_d = {}
		modelNames = []
		for m, _ in enumerate(dfs):
			currDF = dfs[m]
			ex_Feats = set(allVars).difference(currDF.columns)
			if len(ex_Feats)==0:
		#       exStr = "with all features"
				exStr=''
			else: 
				exStr ="wo "+",".join(sorted(list(ex_Feats)))
			print("\n", exStr)
			curr_analysis=xgb_shap(df=currDF, target=target, hyperparams=hyperparams, n_folds_SHAP=n_folds_SHAP, verbose=verbose)
			curr_analysis.shap_values;
			
			mdlName = exStr + "\nAUROC_CV: {:.3f}".format(curr_analysis.score_CV_SHAP)

			modelNames.append(mdlName)
			shapDF_prob_d[m] = curr_analysis.shapDF_prob.copy()
			orderedFeatures_d[m]  = copy.deepcopy(curr_analysis.orderedFeatures)

			if m == 0: 
				tmpDF = np.abs(shapDF_prob_d[0].drop(columns = 'meanExpValue')).mean().reset_index()
			else: 
				tmpDF = pd.merge(tmpDF, 
						 np.abs(shapDF_prob_d[m].drop(columns = 'meanExpValue')).mean().reset_index(), 
								on='index', how='outer')

		# Plot model comparison plots: 
		colNames = ['Feature']
		colNames.extend(modelNames)
		tmpDF.columns = colNames#['Feature', 'Optimized model', 'Model + additional features']
		featureOrder = orderedFeatures_d[0] # Order by importance in first model 
		featureOrder.extend(list(set(allVars).difference(featureOrder))) # And append any additional features
		tmpDF = tmpDF.set_index('Feature').loc[featureOrder].reset_index() 

		plt.style.use('seaborn-talk')
		tmpDF.plot(x= 'Feature', y=modelNames, kind='bar', figsize=(20,8))
		plt.title('Comparison of feature importance in different models')
		plt.ylabel('Mean |SHAP| value')
		if save_fig:
			fileName = outputs_dir + target + "SHAP model comparison {} {}.png".format(analysisName, time.asctime())
			print("Saved: " + fileName)
			plt.savefig(fileName, bbox_inches='tight')
		return


