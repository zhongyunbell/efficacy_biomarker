import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time, itertools, copy, tqdm

from collections import Counter

# import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import shap
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import KFold #StratifiedKFold

import hyperopt as hp
from hyperopt.pyll.base import scope
from hyperopt.pyll.stochastic import rng_from_seed
from hyperopt import STATUS_OK, fmin, hp, tpe, SparkTrials
from hyperopt.early_stop import no_progress_loss

from collections import defaultdict
import seaborn as sns

#### Imports in the package


from ..bootstrap_shap.bootstrap_shap_analysis import parallel_bootstrap_shap_analysis_LR, bootstrap_SHAP_analysis_LR
from ..utils.helpers import LazyProperty, nargout
from ..utils.superclasses import analysis, bootstrap_summary
from .bokeh_plots import binary_classificaiton_plots
from .logit2prob import convert_shapDF_logit_2_shapDF_prob
# from ..clustering import plot_pca, plot_tsne
from ..utils.misc import CustomScaler


def score_model(hyperparams, train_x, train_y, n_folds = 5, test_x = None, test_y=None, n_reps =1, verbose = True, return_val_predictions=False, return_train_scores=False, seed=0): # 
	val_scores = []
	train_scores = []
	y_vals = np.empty((len(train_y),n_folds*n_reps))
	y_vals[:] = np.nan
	y_trains = y_vals.copy() # Deep copies
	predictionsVal = y_vals.copy()
	predictionsTrain = y_vals.copy()
	mdlFeatures = list(train_x.columns)
	bool_cols = [col for col in mdlFeatures if train_x[col].dropna().value_counts().index.isin([0,1]).all()]
	bool_cols_index = [mdlFeatures.index(feat) for feat in bool_cols]

	if n_reps > 10000: 
		warnings.warn("n_reps is very high, you may start recycling random seeds now")
	for r in range(n_reps):
		kf = KFold(n_splits=n_folds, shuffle=True, random_state=r+seed*10000)
		i = 0
		for train_index, val_index in kf.split(train_x): 
			X_tr, X_val = train_x.iloc[train_index], train_x.iloc[val_index]
			y_tr, y_val = train_y.iloc[train_index], train_y.iloc[val_index]
			X_tr, X_val = CustomScaler(binary_columns=bool_cols_index).fit(X_tr).transform(X_tr), CustomScaler(binary_columns=bool_cols_index).fit(X_tr).transform(X_val)
			X_tr, X_val = X_tr.fillna(X_tr.median()), X_val.fillna(X_tr.median())
			
			if y_val.sum() < 3: 
				print("Warning less than 3 positive targets. Moving to next iteration")
				continue 
			model = LogisticRegression(**hyperparams).fit(X_tr,y_tr)

			currPredictions_val = model.predict_proba(X_val)[:,1]
			currPredictions_train = model.predict_proba(X_tr)[:,1]
			predictionsVal[val_index, i+r*n_folds] = currPredictions_val
			predictionsTrain[train_index, i+r*n_folds] = currPredictions_train
			y_vals[val_index, i+r*n_folds] = y_val
			y_trains[train_index, i+r*n_folds] = y_tr

			currROC_AUC_val = roc_auc_score(y_val.to_numpy(),currPredictions_val)
			currROC_AUC_train = roc_auc_score(y_tr.to_numpy(),currPredictions_train)
			val_scores.append(currROC_AUC_val)
			train_scores.append(currROC_AUC_train)
			i+=1
			
	if verbose: 
		print("-- Model performance:\n {}-fold cross validation repeated {} times".format(n_folds,n_reps))
		print(" AUROC (val)   : {:.3f}±{:.2f}".format(np.mean(val_scores), np.std(val_scores)))
		print(" AUROC (train) : {:.3f}±{:.3f}".format(np.mean(train_scores), np.std(train_scores)))
		
	if (type(test_x) != type(None)) & (type(test_y) != type(None)):
		model = LogisticRegression(**hyperparams).fit(train_x,train_y)
		predictions_test = model.predict_proba(test_x)[:,1]
		test_score = roc_auc_score(test_y.to_numpy(),predictions_test)
		if verbose: 
			print(" AUROC (Holdout)  : {:.3f}".format(test_score))
	
	if (type(test_x) != type(None)) & (type(test_y) != type(None)):
		return val_scores, test_score
	elif return_val_predictions ==True: 
		if return_train_scores:
			return val_scores, np.nanmean(predictionsVal, axis=1), train_scores
		else: 
			return val_scores, np.nanmean(predictionsVal, axis=1)
	elif return_train_scores:
		return val_scores, train_scores
	else:
		return val_scores

def reliability_diagram(y_true, y_prob, analysis_name, figsize=(6,2.5)):
	from scipy import stats
	fig = plt.figure(figsize=figsize)
	error_bar_alpha=.05
	c='red'   
	x = y_prob
	y = y_true
	set_name =analysis_name

	bins = [0]
	bins.extend(np.linspace(np.quantile(x[~np.isnan(x)], q =.05), np.quantile(x, q =.95), 9))
	bins.extend([1])
	digitized_x = np.digitize(x, bins)

	mean_count_array = np.array([[np.mean(y[digitized_x == i]),
								  len(y[digitized_x == i]),
								  np.mean(x[digitized_x==i])] 
								  for i in np.unique(digitized_x)])
	x_pts_to_graph = mean_count_array[:,2]
	y_pts_to_graph = mean_count_array[:,0]
	bin_counts = mean_count_array[:,1]
	plt.style.use('seaborn-talk')
	plt.subplot(1,2,1)
	plt.plot(np.linspace(0,1,100),(np.linspace(0,1,100)),'k--')
	plt.scatter(x_pts_to_graph,y_pts_to_graph, c=c)
	plt.axis([-0.1,1.1,-0.1,1.1])
	yerr_mat = stats.binom.interval(1-error_bar_alpha,bin_counts,x_pts_to_graph)/bin_counts - x_pts_to_graph
	yerr_mat[0,:] = -yerr_mat[0,:]
	plt.errorbar(x_pts_to_graph, x_pts_to_graph, yerr=yerr_mat, capsize=5)
	plt.xlabel('Predicted')
	plt.ylabel('Empirical')
	plt.title('Reliability diagram\n {}'.format(set_name))

	plt.subplot(1,2,2)
	plt.hist(x,bins=bins, ec='w')
	plt.title("Predicted probabilities\n{}".format(set_name))
	plt.tight_layout()
	return fig

# Hyperparameter tuning

SEED = 314159265
num_boost_round=100
n_folds_CV = 5 # for CV performance evaluation
n_folds_SHAP = 10 # for CV when doing SHAP analysis

def objective_model(hyperparams, train_x, train_y, b_coeff=0, n_folds=n_folds_CV):
	"""Objective function for Hyperparameter Optimization"""   
	aurocs, predictionCV = score_model(hyperparams, train_x, train_y, n_folds = n_folds_CV, 
										 verbose=False, return_val_predictions = True) 
	y_prob = np.array(predictionCV).flatten()
	score = np.mean(aurocs) 
	loss = 1 - score + b_coeff*brier_score_loss(train_y, y_prob)
	# Dictionary with information for evaluation
	return {'loss': loss,  'status': STATUS_OK}

def optimize(score, random_state=SEED, verbose=True, max_evals = 3):
	"""
	This is the optimization function that given a space (space here) of 
	hyperparameters and a scoring function (score here), finds the best hyperparameters.
	"""
	# Optimal grid search parameters: 
	space = {
		'C':            hp.loguniform('C', np.log(0.01), np.log(10000)),
		"penalty": "l2",
	}
	rstate = np.random.RandomState(SEED)
	best = fmin(score, space, algo=tpe.suggest, max_evals=max_evals, 
				verbose=verbose,rstate=rstate, 
				early_stop_fn=no_progress_loss(iteration_stop_count=50, percent_increase=0)) #, trials = SparkTrials(parallelism = 4))
	return best

def SHAP_CV_logistic(df, mdlFeatures, target, hyperparams, nFolds, verbose=True): 
	bool_cols = [col for col in mdlFeatures if df[mdlFeatures][col].dropna().value_counts().index.isin([0,1]).all()]
	bool_cols_index = [mdlFeatures.index(feat) for feat in bool_cols]
	if verbose: 
		print(f"Extracting SHAP values using {nFolds}-fold CV ...")
	start = time.time()
	val_scores = []
	train_scores = []
	models_d = {}
	explainer_logit_d = {}
	expectedLogit_d={}
	expectedProb_d={}
	# shap_interaction_val_d = {}

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
		X_tr, X_val = CustomScaler(binary_columns=bool_cols_index).fit(X_tr).transform(X_tr), CustomScaler(binary_columns=bool_cols_index).fit(X_tr).transform(X_val)
		X_tr, X_val = X_tr.fillna(X_tr.median()), X_val.fillna(X_tr.median())
		y_true.extend(y_val)

		model = LogisticRegression(**hyperparams).fit(X_tr,y_tr)
		currPredictions_val = model.predict_proba(X_val)[:,1]
		currPredictions_train = model.predict_proba(X_tr)[:,1]
		
		predictionsVal[val_index, i] = currPredictions_val
		predictionsTrain[train_index, i] = currPredictions_train
		y_vals[val_index, i] = y_val
		y_trains[train_index, i] = y_tr
		currROC_AUC_train = roc_auc_score(y_tr.to_numpy(),currPredictions_train)
		train_scores.append(currROC_AUC_train)

		explainer_logit = shap.LinearExplainer(model, X_tr)
		shap_values_logit_curr = explainer_logit.shap_values(X_val) #df[mdlFeatures].iloc[val_index]

		if i == 0: 
			shap_values_logit = shap_values_logit_curr
		else:    
			shap_values_logit = np.concatenate([shap_values_logit, shap_values_logit_curr])
		
		for v in val_index:
			models_d[v] = model
			explainer_logit_d[v] = explainer_logit
			expectedLogit_d[v] = explainer_logit.expected_value
		i+=1
	
	# Processing after For-loop
	meanExpLogit = np.mean(list(expectedLogit_d.values()))

	CV_AUROC = roc_auc_score(y_true,np.nanmean(predictionsVal, axis=1)) 
	if verbose:
		print("  CV AUROC: {:.3f}".format(CV_AUROC))
		print("  CV AUROC (train): {:.3f}±{:.3f}".format(np.mean(train_scores), np.std(train_scores)))
		end = time.time()
		print("   Execution time: {:.2f}s".format(np.round(end - start,5)))

	valPredictions_vec = np.nansum(predictionsVal, axis=1) # Only 1 value in each row
	outputs_d = dict(models_d=models_d,
					 shap_values_logit=shap_values_logit,
                     # shap_values_prob=shap_values_prob,
                     # explainer_prob_d=explainer_prob_d, 
                     # expectedProb_d = expectedProb_d, 
					 explainer_logit_d=explainer_logit_d,
					 expectedLogit_d=expectedLogit_d, 
                     # meanExpProb=meanExpProb,
					 meanExpLogit =meanExpLogit, 
					 predictionsVal=predictionsVal,
					 valPredictions_vec=valPredictions_vec,
					 predictionsTrain=predictionsTrain, 
					 y_vals=y_vals, 
					 y_trains=y_trains,
					 CV_score=CV_AUROC
					)
	return outputs_d

class logistic_shap(analysis, bootstrap_summary):
	def __init__(self, df, target, max_evals=100, hyperparams=None, n_folds_SHAP=20, bootstrap_iterations=int(np.round(1000*np.e)), verbose=True, remove_outliers=10, outputs_dir='./'): 
		"""
		Create an analysis instance for logistic regression
		remove outliers : True or False; if number supplied, it removes outliers > z standard deviations 
		"""
		analysis.__init__(self, df=df, target=target, remove_outliers=remove_outliers, outputs_dir=outputs_dir, categorical_encoding = 'ohe')
		bootstrap_summary.__init__()	
		self.max_evals = max_evals
		if hyperparams == None: 
			defaul_hyperparams = {"C": 1, 'penalty': 'l2'}
			self.hyperparams = defaul_hyperparams
		else:
			self.hyperparams = hyperparams
		
		self.AUROC_CV_SHAP = "Call 'shap_values_logit' first"
		self.predictionsCV_SHAP = "Call 'shap_values_logit' first"
		self.SHAP_outputs = "Call 'shap_values_logit' first"
		self.n_folds_SHAP=n_folds_SHAP
		self.bootstrap_SHAP_outputs = """Call 'run_bootstrap_SHAP_analysis_local' first. 
		run_bootstrap_SHAP_analysis_local only supports sample with replacement method currently.
		run_bootstrap_SHAP_analysis_parallel does not support outputting models yet.
		"""
		self.bootstrap_iterations = bootstrap_iterations
		self.verbose = verbose
		self.n_perms = 50 # For converting logit to shap values



	def tune_model(self, max_evals=None, verbose=True):
		if max_evals==None: 
			max_evals = self.max_evals
		start=time.time()
		def objective_model_curr(hyperparams, train_x=self.X, train_y=self.y):
			return objective_model(hyperparams, train_x=train_x, train_y=train_y)
		hyperparams = optimize(objective_model_curr, verbose=verbose, max_evals=max_evals)
		end=time.time()
		if verbose:
			print("Done: took", (end-start), "seconds")
			print("The best hyperparameters are: ", "\n")
			print(hyperparams)
			setattr(self, 'hyperparams', hyperparams)
		n_out = nargout()
		if n_out == 1:
			return hyperparams

	def model_performance(self, n_folds=n_folds_CV, n_reps=1, verbose=True):
		aurocs = score_model(self.hyperparams, self.X, self.y, n_folds = n_folds, n_reps=n_reps,
										 verbose=verbose, return_val_predictions = False)
		if verbose:
			print("AUROC from {}-fold cross-validation ({} reps): ".format(n_folds, n_reps) + "{:.3f}±{:.2f}".format(np.mean(aurocs), np.std(aurocs)))
		return aurocs

	def plot_reliability(self, show=True, n_folds = n_folds_CV): 
		aurocs, predictionsCV = score_model(self.hyperparams, self.X, self.y, n_folds = n_folds, 
										 verbose=verbose, return_val_predictions = True)
		fig = reliability_diagram(self.y, predictionsCV, analysis_name="Logistic Regression model", figsize=(10,4))
		fig.text(.1, 1, "AUROC from {}-fold cross-validation: ".format(n_folds) + "{:.3f}±{:.2f}".format(np.mean(aurocs), np.std(aurocs)), ha='left', fontsize = 18)
		if show: 
			pass
		else:
			plt.close()
			return fig

	def plot_reliability_SHAP(self, show=True):
		if self.predictionsCV_SHAP == "Call 'shap_values_logit' first":
			self.shap_values_prob;

		fig = reliability_diagram(self.y, self.predictionsCV_SHAP, analysis_name="Logistic Regression model", figsize=(10,4))
		fig.text(.1, 1, "AUROC (on validation sets in generation of SHAP values): {:.3f}".format(self.AUROC_CV_SHAP), ha='left', fontsize = 18)
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
			show(layout)

	@LazyProperty
	def shap_values_logit(self):
		SHAP_outputs = SHAP_CV_logistic(df=self.df, mdlFeatures=self.mdlFeatures, target=self.target, hyperparams=self.hyperparams, nFolds=self.n_folds_SHAP, verbose=self.verbose)
		setattr(self, 'AUROC_CV_SHAP', SHAP_outputs['CV_score'])
		setattr(self, 'predictionsCV_SHAP', SHAP_outputs['valPredictions_vec'])
		setattr(self, 'SHAP_outputs', SHAP_outputs)
		setattr(self, 'meanExpLogit', SHAP_outputs['meanExpLogit'])
		return SHAP_outputs['shap_values_logit']
	
	@LazyProperty
	def shap_values_prob(self):
		shap_values_prob = self.shapDF_prob[self.mdlFeatures].values
		return shap_values_prob

	@LazyProperty
	def shapDF_logit(self):
		shapDF_logit = pd.DataFrame(self.shap_values_logit, columns=self.mdlFeatures)
		shapDF_logit['expectedValue'] = list(self.SHAP_outputs['expectedLogit_d'].values())
		return shapDF_logit
	
	@LazyProperty
	def shapDF_prob(self):
		if len(self.mdlFeatures) >7:
			print("""This may take a while to convert logit to prob. 
			Try working only in the logit domain:
			self.shapDF_prob = self.shapDF_logit
			""")
		else:
			print("Converting from probability to logit may take a long time unless many CPUs are available")
		shapDF_prob = convert_shapDF_logit_2_shapDF_prob(self.shapDF_logit, n_perms=self.n_perms)
		setattr(self, 'meanExpProb', shapDF_prob['expectedValue'].mean())
		return shapDF_prob
	
	def shap_summary_plots_prob(self, selectionVec = None, show=True, figsize=(40,12), save_fig=False, outputs_dir=None, **kwargs): 
		"""
		kwargs are passed into shap.summary_plot()
		e.g. max_display
		"""
		if type(selectionVec) == type(None): 
			selectionVec = self.df.index.notna() # all rows
			selected = 'all'
		else: 
			selected = 'subset'
		fig = plt.figure(figsize=figsize)
		gs = fig.add_gridspec(1, 3)
		fig.add_subplot(gs[0, 0])
		shap.summary_plot(self.shap_values_prob[selectionVec,:], features=self.df.loc[selectionVec,:].drop(columns=self.target), show=False, plot_size=(figsize[0]/3,figsize[1]), plot_type='bar', **kwargs)
		fig.add_subplot(gs[0, 1:])
		shap.summary_plot(self.shap_values_prob[selectionVec,:], features=self.df.loc[selectionVec,:].drop(columns=self.target), show=False, plot_size=(2*figsize[0]/3,figsize[1]), **kwargs)
		plt.tight_layout()
		if selected == 'all':
			fig.text(.1, 1, "AUROC (CV): {:.3f}".format(self.AUROC_CV_SHAP), ha='left', fontsize = 18)
		if save_fig:
			if outputs_dir==None: 
				outputs_dir = self.outputs_dir
			else:
				pass
			file_name= outputs_dir + "SHAP summary plot (prob) {}.png".format(self.target)
			plt.savefig(file_name, bbox_inches='tight')
			print("Saved: ", file_name)
			return
		if show: 
			pass
		else:
			plt.close()
			return fig

	def shap_summary_plots_logit(self, selectionVec = None, show=True, figsize=(40,12), save_fig=False, outputs_dir=None, **kwargs): 
		if type(selectionVec) == type(None): 
			selectionVec = self.df.index.notna() # all rows
			selected = 'all'
		else: 
			selected = 'subset'
		fig = plt.figure(figsize=figsize)
		gs = fig.add_gridspec(1, 3)
		fig.add_subplot(gs[0, 0])
		shap.summary_plot(self.shap_values_logit[selectionVec,:], features=self.df.loc[selectionVec,:].drop(columns=self.target), max_display=30, show=False, plot_size=(figsize[0]/3,figsize[1]), plot_type='bar', **kwargs)
		fig.add_subplot(gs[0, 1:])
		shap.summary_plot(self.shap_values_logit[selectionVec,:], features=self.df.loc[selectionVec,:].drop(columns=self.target), max_display=30, show=False, plot_size=(2*figsize[0]/3,figsize[1]), **kwargs)
		plt.tight_layout()
		if selected == 'all':
			fig.text(.1, 1, "AUROC (CV): {:.3f}".format(self.AUROC_CV_SHAP), ha='left', fontsize = 18)
		if save_fig:
			if outputs_dir==None: 
				outputs_dir = self.outputs_dir
			else:
				pass
			file_name= outputs_dir + "SHAP summary plot (logit) {}.png".format(self.target)
			plt.savefig(file_name, bbox_inches='tight')
			print("Saved: ", file_name)
			return
		if show: 
			pass
		else:
			plt.close()
			return fig

	def shap_dependence_plots_prob(self, show=True, save_fig=False, outputs_dir=None):
		orderedFeatures_top_9 = self.orderedFeatures[:9]
		fig = plt.figure(figsize=(16,12))
		ylims = [self.shap_values_prob.min(), self.shap_values_prob.max()]
		for i,feature in enumerate(orderedFeatures_top_9): 
			ax = fig.add_subplot(331+i)
			shap.dependence_plot(self.mdlFeatures.index(feature), self.shap_values_prob, self.df[self.mdlFeatures],ax=ax, show=False)
			ax.set_ylim(ylims)
			xlims = ax.get_xlim()
			ax.hlines(y=0, xmin =xlims[0], xmax= xlims[1], ls='--')
			
		plt.tight_layout()
		if save_fig:
			if outputs_dir==None: 
				outputs_dir = self.outputs_dir
			else:
				pass
			file_name= outputs_dir + "SHAP dependence plots (prob) {}.png".format(self.target)
			plt.savefig(file_name)
			print("Saved: ", file_name)
			return
		if show: 
			pass
		else:
			plt.close()
			return fig

	def dependence_plot_prob(self,feature, interaction_feature='auto', show=True, save_fig=False, outputs_dir=None, **kwargs):
		if interaction_feature == 'auto':
			interaction_index = 'auto'
		else:
			interaction_index = self.mdlFeatures.index(interaction_feature)
		shap.dependence_plot(self.mdlFeatures.index(feature), self.shap_values_prob, self.df[self.mdlFeatures], interaction_index=interaction_index, show=show, **kwargs)
		return 
	
	def dependence_plot_logit(self,feature, interaction_feature='auto', show=True, save_fig=False, outputs_dir=None, **kwargs):
		if interaction_feature == 'auto':
			interaction_index = 'auto'
		else:
			interaction_index = self.mdlFeatures.index(interaction_feature)
		shap.dependence_plot(self.mdlFeatures.index(feature), self.shap_values_logit, self.df[self.mdlFeatures], interaction_index=interaction_index, show=show, **kwargs)
		return 

	@LazyProperty
	def orderedFeatures(self):
		""" Order of importance based on SHAP analysis"""
		vals= np.abs(self.shap_values_prob).mean(0) # Ordered by average |SHAP value| 
		ordered_Features = list(pd.DataFrame(zip(self.mdlFeatures,vals)).sort_values(1, ascending=False).reset_index(drop=True)[[0]].values.flatten())
		return ordered_Features


	@LazyProperty
	def bootsDF_logit(self):
		"""If the models in each iteration are desired, 
		then run run_bootstrap_SHAP_analysis instead to avoid doubling computation time."""
		bootsDF_logit = parallel_bootstrap_shap_analysis_LR(df=self.df, target=self.target, params=self.hyperparams, 
			iterations=self.bootstrap_iterations, method='sample_with_replacement', train_size=.8, stratification_factor=None)
		return bootsDF_logit

	@LazyProperty
	def bootsDF(self):
		if len(self.mdlFeatures) >7:
			print("""This may take a while to convert logit to prob. 
			Try working only in the logit domain:
			self.bootsDF = self.bootsDF_logit
			Trying to convert anyway:
			""")
		print("Converting from probability to logit may take a long time unless many CPUs are available")
		bootsDF = convert_shapDF_logit_2_shapDF_prob(self.bootsDF_logit, mdlFeatures=self.mdlFeatures, n_perms=self.n_perms)
		bootsDF['bootsIteration'] = self.bootsDF_logit['bootsIteration']
		bootsDF['index'] = self.bootsDF_logit['index']
		return bootsDF

	def run_bootstrap_SHAP_analysis_local(self, bootstrap_iterations=None, method='sample_with_replacement'): 
		if bootstrap_iterations==None: 
			bootstrap_iterations = self.bootstrap_iterations
		bootsDF_logit, out = bootstrap_SHAP_analysis_LR(df=self.df, target=self.target, model=LogisticRegression(**self.hyperparams), 
			iterations=bootstrap_iterations, method=method)
		setattr(self, 'bootstrap_SHAP_outputs', out)
		setattr(self, 'bootsDF_logit', bootsDF_logit)
		return
	
	def run_bootstrap_SHAP_analysis_parallel(self, bootstrap_iterations=None, method='sample_with_replacement', train_size=0.8, stratification_factor=None): 
		"""
		This runs alternative bootstrap analyses
		method : 'train_test_split' or 'sample_with_replacement'
		running this saves the property 'bootsDF' to this class so
		it's available for plotting.
		"""

		if bootstrap_iterations==None: 
			bootstrap_iterations = self.bootstrap_iterations
		bootsDF_logit = parallel_bootstrap_shap_analysis_LR(df=self.df, target=self.target, params=self.hyperparams,
			iterations=bootstrap_iterations, method=method, train_size=train_size, stratification_factor=stratification_factor)
		setattr(self, 'bootsDF_logit', bootsDF_logit)
		return bootsDF_logit

	# bootstrap_summary class contains functions like: 
	# mean_expected_value_bootstrap, plot_bootstrapped_feature_dependence, 
	# bootstrap_feature_summary_table, generate_bootstrap_summaryDF, plot_bootstrap_summary_table
		
	# def save(self, name="logisticAnalysis"): 
	# 	import time, pickle
	# 	fileName = self.outputs_dir + "_".join([self.target, name,time.asctime()]) +".p"
	# 	pickle.dump(self, open(fileName, "wb"))
	# 	print("Saving analysis as: ", fileName)

	def summary_stats(self, f=None, interaction_terms=""):
		""" 
		You can manually add interaction terms by setting argument interaction_terms:
		e.g. : interaction_terms = "+ BHBA1C:Cminsd"

		"""
		import statsmodels.api as sm
		import statsmodels.formula.api as smf
		print('Statistics of logistic regression (no regularization)')
		renamed_cols = [s.replace(' ', '_').replace('≥', '') for s in list(self.df.columns)]
		renamed_features = [s.replace(' ', '_') for s in list(self.mdlFeatures)]
		renamed_features = [s.replace('-', '_') for s in renamed_features]
		logitDF=self.df.rename(columns = dict(zip(list(self.df.columns), renamed_cols)))
		if type(f)==type(None):
			f = self.target.replace('≥', '') + "~" + " + ".join(renamed_features) + interaction_terms

		print(f)
		logitfit = smf.logit(formula = str(f), data = logitDF).fit(disp=False, maxiter=100)
		print(logitfit.summary())
		return logitfit.summary()

		
### Additional post hoc analyses
def shapModelComparison_logitstic(dfs, target, hyperparams, analysisName='', save_fig=False, outputs_dir='./', n_folds_SHAP=20, verbose=False):
		"""
		dfs : list of dataframes
			  Features will be ordered by importance of first df

		analysisName: Give a unique name. This will be utilized to save files and load cached outputs.

		"""
		allVars = list(set(np.concatenate([list(dfs[m].columns) for m in range(len(dfs))]).flat)) # 
		allVars.pop(allVars.index(target))
		ModelComp_d = {}
		orderedFeatures_d = {}
		shapDF_logit_d = {}
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
			curr_analysis=logistic_shap(df=currDF, target=target, hyperparams=hyperparams, n_folds_SHAP=n_folds_SHAP, verbose=verbose)
			curr_analysis.shap_values_logit;
			
			mdlName = exStr + "\nAUROC_CV: {:.3f}".format(curr_analysis.AUROC_CV_SHAP)

			modelNames.append(mdlName)
			shapDF_logit_d[m] = curr_analysis.shapDF_logit.copy()
			orderedFeatures_d[m]  = copy.deepcopy(curr_analysis.orderedFeatures)

			if m == 0: 
				tmpDF = np.abs(shapDF_logit_d[0].drop(columns = 'expectedValue')).mean().reset_index()
			else: 
				tmpDF = pd.merge(tmpDF, 
						 np.abs(shapDF_logit_d[m].drop(columns = 'expectedValue')).mean().reset_index(), 
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
			fileName = outputs_dir + target + "SHAP model comparison (logit) {} {}.png".format(analysisName, time.asctime())
			print("Saved: " + fileName)
			plt.savefig(fileName, bbox_inches='tight')
		return