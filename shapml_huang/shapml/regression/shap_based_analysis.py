import warnings
from bokeh.models.glyphs import AnnularWedge
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time, itertools, copy, tqdm

from collections import Counter

import xgboost as xgb
import shap
# shap.initjs()
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold #StratifiedKFold

import hyperopt as hp
from hyperopt.pyll.base import scope
from hyperopt.pyll.stochastic import rng_from_seed
from hyperopt import STATUS_OK, fmin, hp, tpe, SparkTrials
from hyperopt.early_stop import no_progress_loss


from ..utils.superclasses import analysis, bootstrap_summary, shap_analysis
from sklearn.metrics import r2_score
from ..bootstrap_shap.bootstrap_shap_analysis import parallel_bootstrap_shap_analysis, bootstrap_SHAP_analysis
from ..utils.helpers import LazyProperty, nargout


from collections import defaultdict
import seaborn as sns

def score_model(params, train_x, train_y, n_folds = 10, test_x = None, test_y=None, n_reps =1, verbose = True, return_val_predictions=False, return_train_scores=False, seed=0):
	val_scores = []
	train_scores = []
	y_vals = np.empty((len(train_y),n_folds*n_reps))
	y_vals[:] = np.nan
	y_trains = y_vals.copy() # Deep copies
	predictionsVal = y_vals.copy()
	predictionsTrain = y_vals.copy()
	if n_reps > 10000: 
		raise ValueError("n_reps is very high, you may start recycling random seeds now")
	for r in range(n_reps):
		kf = KFold(n_splits=n_folds, shuffle=True, random_state=r+seed*10000)
		i = 0
		for train_index, val_index in kf.split(train_x): 
			X_tr, X_val = train_x.iloc[train_index], train_x.iloc[val_index]
			y_tr, y_val = train_y.iloc[train_index], train_y.iloc[val_index]
			dtrain = xgb.DMatrix(X_tr, label=y_tr)
			dval   = xgb.DMatrix(X_val, label=y_val)
			watchlist = [(dval, 'eval'), (dtrain, 'train')]
			xgb_model = xgb.train(params, dtrain, num_boost_round)

			currPredictions_val = xgb_model.predict(dval)
			currPredictions_train = xgb_model.predict(dtrain)
			predictionsVal[val_index, i+r*n_folds] = currPredictions_val
			predictionsTrain[train_index, i+r*n_folds] = currPredictions_train
			y_vals[val_index, i+r*n_folds] = y_val
			y_trains[train_index, i+r*n_folds] = y_tr

			currR2_val = r2_score(y_val.to_numpy(),currPredictions_val)
			currR2_train = r2_score(y_tr.to_numpy(),currPredictions_train)
			val_scores.append(currR2_val)
			train_scores.append(currR2_train)
			i+=1
			
	if verbose: 
		print("-- Model performance:\n {}-fold cross validation repeated {} times".format(n_folds,n_reps))
		print(" R2 (val)   : {:.3f}±{:.2f}".format(np.mean(val_scores), np.std(val_scores)))
		print(" R2 (train) : {:.3f}±{:.3f}".format(np.mean(train_scores), np.std(train_scores)))
		
	if (type(test_x) != type(None)) & (type(test_y) != type(None)):
		dtrain = xgb.DMatrix(train_x, label=train_y)
		dval   = xgb.DMatrix(test_x, label=test_y)
		watchlist = [(dval, 'eval'), (dtrain, 'train')]
		xgb_model = xgb.train(params, dtrain, num_boost_round)
		predictions_all = xgb_model.predict(dval)
		# test_score = roc_auc_score(test_y.to_numpy(),predictions_all)
		test_score = r2_score(test_y.to_numpy(),predictions_all)
		if verbose: 
			print(" R2 (Holdout)  : {:.3f}".format(test_score))
	
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

def reliability_diagram(y_true, y_predict, analysis_name, figsize=(6,2.5), nBins = 10):
	from scipy import stats
	fig = plt.figure(figsize=figsize)
	error_bar_alpha=.05
	c='red'   
	x = y_predict
	y = y_true
	set_name =analysis_name

	bins = [np.min(y_predict)-.001]
	bins.extend(np.linspace(np.quantile(x[~np.isnan(x)], q =.05), np.quantile(x, q =.95), nBins-1))
	bins.extend([np.max(y_predict)+.001])
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
	plt.plot(np.linspace(np.min(bins),np.max(bins),20),(np.linspace(np.min(bins),np.max(bins),20)),'k--')
	plt.scatter(x_pts_to_graph,y_pts_to_graph, c=c)
	# plt.axis([-0.1,1.1,-0.1,1.1])
	std_x = np.array([np.std(x[digitized_x == i]) for i in np.unique(digitized_x)])

	yerr_mat = stats.binom.interval(1-error_bar_alpha,bin_counts,x_pts_to_graph)/bin_counts - x_pts_to_graph
	yerr_mat[0,:] = -yerr_mat[0,:]
	plt.errorbar(x_pts_to_graph, x_pts_to_graph, yerr=std_x, capsize=5)
	plt.xlabel('Predicted')
	plt.ylabel('Empirical')
	plt.title('Reliability diagram\n {}'.format(set_name))

	plt.subplot(1,2,2)
	plt.hist(x,bins=bins, ec='w')
	plt.title("Predicted Y\n{}".format(set_name))
	plt.tight_layout()
	return fig

SEED = 314159265
num_boost_round=100
n_folds_CV = 5 # for CV performance evaluation
n_folds_SHAP = 10 # for CV when doing SHAP analysis

def objective_model(params, train_x, train_y, b_coeff=0, n_folds=n_folds_CV):
	"""Objective function for XGBoost Hyperparameter Optimization"""   
	kf = KFold(n_splits=n_folds)
	# aurocs, predictionCV = ROC_AUC_model(params, train_x, train_y, n_folds = n_folds_CV, 
										 # verbose=False, return_val_predictions = True) 
	scores, predictionCV = score_model(params, train_x, train_y, n_folds = n_folds_CV, 
										 verbose=False, return_val_predictions = True) 
	y_prob = np.array(predictionCV).flatten()
	# score = np.mean(aurocs) 
	score = np.mean(scores)
	# loss = 1 - score + b_coeff*brier_score_loss(train_y, y_prob)
	loss = mean_squared_error(train_y, y_prob, squared=False)
	# Dictionary with information for evaluation
	return {'loss': loss,  'status': STATUS_OK}

def optimize_xgb(score, random_state=SEED, verbose=True, max_evals = 50):
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
		'reg_lambda':                  hp.loguniform('reg_lambda', np.log(0.001), np.log(100)),
		'subsample':                   hp.uniform('subsample', 0.6, 1),
		# "objective": "binary:logistic", 
		"objective": "reg:squarederror",         
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


def SHAP_CV(df, mdlFeatures, target, hyperparams, nFolds, verbose=True, background_data=None):  
	if verbose: 
		print(f"Extracting SHAP values using {nFolds}-fold CV ...")
	start = time.time()
	val_scores = []
	train_scores = []
	models_d = {}
	explainer_d = {}
	explainer_interventional_d = {}
	
	expected_d={}
	expectedInterventional_d={}
	
	shap_interaction_val_d = {}

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

		dtrain = xgb.DMatrix(X_tr, label=y_tr)
		dval   = xgb.DMatrix(X_val, label=y_val)
		watchlist = [(dval, 'eval'), (dtrain, 'train')]
		
		xgb_model = xgb.train(hyperparams, dtrain, num_boost_round)

		currPredictions_val = xgb_model.predict(dval)
		currPredictions_train = xgb_model.predict(dtrain)

	#     predictionsVal.extend(currPredictions_val)
		predictionsVal[val_index, i] = currPredictions_val
		predictionsTrain[train_index, i] = currPredictions_train
		y_vals[val_index, i] = y_val
		y_trains[train_index, i] = y_tr
		# currROC_AUC_train = roc_auc_score(y_tr.to_numpy(),currPredictions_train)
		# train_scores.append(currROC_AUC_train)
		currR2_train = r2_score(y_tr.to_numpy(),currPredictions_train)
		train_scores.append(currR2_train)

		# Calculate Shap values: 
		# explainer_prob = shap.TreeExplainer(xgb_model,data=df.drop(columns=target),
											  # feature_dependence="interventional", model_output="probability")
		explainer = shap.TreeExplainer(xgb_model, feature_perturbation="tree_path_dependent")
		shap_values_curr = explainer.shap_values(df[mdlFeatures].iloc[val_index])
		if type(background_data) == pd.DataFrame:
			explainer_interventional = shap.TreeExplainer(xgb_model,data=background_data,
											   feature_dependence="interventional", model_output="raw")
		else:
			explainer_interventional = shap.TreeExplainer(xgb_model,data=df.drop(columns=target),
											   feature_dependence="interventional", model_output="raw")

		shap_values_interventional_curr = explainer_interventional.shap_values(df[mdlFeatures].iloc[val_index])


		if i == 0: 
			shap_values = shap_values_curr
			shap_values_interventional = shap_values_interventional_curr
		else:    
			shap_values = np.concatenate([shap_values, shap_values_curr])
			shap_values_interventional = np.concatenate([shap_values_interventional, shap_values_interventional_curr])
		
		
		for v in val_index:
			models_d[v] = xgb_model
			explainer_d[v] = explainer
			expected_d[v] = explainer.expected_value

			explainer_interventional_d[v] = explainer_interventional
			expectedInterventional_d[v] = explainer_interventional.expected_value
		i+=1
	
	# Processing after For-loop
	meanExpValue = np.mean(list(expected_d.values()))
	meanExpInterventional = np.mean(list(expectedInterventional_d.values()))

	CV_R2 = r2_score(y_true,np.nanmean(predictionsVal, axis=1))
	if verbose:
		print("  CV R2: {:.3f}".format(CV_R2))
		print("  CV R2 (train): {:.3f}±{:.3f}".format(np.mean(train_scores), np.std(train_scores)))

		end = time.time()
		print("   Execution time: {:.2f}s".format(np.round(end - start,5)))
	valPredictions_vec = np.nansum(predictionsVal, axis=1) # Only 1 value in each row
	outputs_d = dict(models_d=models_d,
					 shap_values=shap_values,
					 shap_values_interventional=shap_values_interventional,
					 explainer_d = explainer_d,
					 expected_d = expected_d,
					 explainer_interventional_d = explainer_interventional_d,
					 expectedInterventional_d= expectedInterventional_d,
					 meanExpValue=meanExpValue,
					 meanExpInterventional =meanExpInterventional, 
					 predictionsVal=predictionsVal,
					 valPredictions_vec=valPredictions_vec,
					 predictionsTrain=predictionsTrain, 
					 y_vals=y_vals, 
					 y_trains=y_trains,
					 CV_R2=CV_R2
					)
	return outputs_d


class xgb_shap(analysis, bootstrap_summary, shap_analysis):
	def __init__(self, df, target, max_evals=50, hyperparams=None, n_folds_SHAP=20, scale_pos_weight=False, bootstrap_iterations=np.round(1000*np.e), verbose=True, outputs_dir= './', remove_outliers=False, **kwargs): 
		# import pdb; pdb.set_trace()
		analysis.__init__(self, df=df, target=target, remove_outliers=remove_outliers,outputs_dir=outputs_dir, verbose=verbose, **kwargs)
		bootstrap_summary.__init__()
		self.max_evals = max_evals
		if hyperparams == None: 
			default_hyperparams = xgb.XGBRegressor().get_xgb_params()
			default_hyperparams.update({'eval_metric': 'error'})
			self.hyperparams = default_hyperparams
		else:
			self.hyperparams = hyperparams
		self.score_name = 'R2'
		self.score_CV_SHAP = "Call 'shap_values' first"
		self.predictionsCV_SHAP = "Call 'shap_values' first"
		self.SHAP_outputs = "Call 'shap_values' first"
		self.n_folds_SHAP=n_folds_SHAP
		self.meanExpValue = None
		self.bootstrap_SHAP_outputs = """Call 'run_bootstrap_SHAP_analysis_local' first. 
		run_bootstrap_SHAP_analysis_local only supports sample with replacement method currently.
		run_bootstrap_SHAP_analysis_parallel does not support outputting models yet.
		"""
		
		self.bootstrap_iterations = bootstrap_iterations

	def tune_model(self, max_evals=None, verbose=True):
		if max_evals==None:
			max_evals=self.max_evals

		start=time.time()
		def objective_model_curr(params, train_x=self.X, train_y=self.y):
			return objective_model(params, train_x=train_x, train_y=train_y)
		hyperparams = optimize_xgb(objective_model_curr, verbose=verbose, max_evals=max_evals)
		hyperparams.update({"objective": "reg:squarederror", 'tree_method':"exact", "eval_metric":"error"})        
		end=time.time()
		if verbose:
			print("Done: took", (end-start), "seconds")
			print("The best hyperparameters are: ", "\n")
			print(hyperparams)
			setattr(self, 'hyperparams', hyperparams)
		n_out = nargout()
		if n_out == 1:
			return hyperparams

	def model_performance(self, n_folds=n_folds_CV, n_reps=1, verbose=True, seed=0):
		r2s, predictionsCV = score_model(self.hyperparams, self.X, self.y, n_folds = n_folds, n_reps=n_reps, verbose=verbose, return_val_predictions = True, seed=seed)
		if verbose:
			print("R2 from {}-fold cross-validation ({} reps): ".format(n_folds, n_reps) + "{:.3f}±{:.2f}".format(np.mean(r2s), np.std(r2s)))
		return r2s
		

	def plot_reliability(self, show=True, n_folds = n_folds_CV): 
		r2s, predictionsCV = score_model(self.hyperparams, self.X, self.y, n_folds = n_folds, verbose=False, return_val_predictions = True)
		fig = reliability_diagram(self.y, predictionsCV, analysis_name="XGBoost model", figsize=(10,4))
		fig.text(.1, 1, "R2 from {}-fold cross-validation: ".format(n_folds) + "{:.3f}±{:.2f}".format(np.mean(r2s), np.std(r2s)), ha='left', fontsize = 18)        
		if show: 
			pass
		else:
			plt.close()
			return fig

	def plot_reliability_SHAP(self, show=True):
		if self.predictionsCV_SHAP == "Call 'shap_values' first":
			self.shap_values
		fig = reliability_diagram(self.y, self.predictionsCV_SHAP, analysis_name="XGBoost model", figsize=(10,4))
		fig.text(.1, 1, "{} (CV used to generate SHAP values): {:.3f}".format(self.score_name, self.score_CV_SHAP), ha='left', fontsize = 18)   
		if show: 
			pass
		else:
			plt.close()
			return fig

	@LazyProperty
	def shap_values(self):
		SHAP_outputs = SHAP_CV(df=self.df, mdlFeatures=self.mdlFeatures, target=self.target, hyperparams=self.hyperparams, nFolds=self.n_folds_SHAP, verbose=self.verbose)
		setattr(self, 'score_CV_SHAP', SHAP_outputs['CV_R2'])
		setattr(self, 'predictionsCV_SHAP', SHAP_outputs['valPredictions_vec'])
		setattr(self, 'SHAP_outputs', SHAP_outputs)
		setattr(self, 'meanExpValue', SHAP_outputs['meanExpValue'])
		return SHAP_outputs['shap_values']

	@LazyProperty
	def shapDF(self):
		shapDF = pd.DataFrame(self.shap_values, columns=self.mdlFeatures)
		shapDF['meanExpValue'] = self.meanExpValue
		return shapDF

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
		# hyperparams.update({'use_label_encoder':False})
		bootsDF = parallel_bootstrap_shap_analysis(df=self.df, target=self.target, xgb_hyperparams=hyperparams, explainer_type = 'raw', 
			iterations=self.bootstrap_iterations, method='sample_with_replacement', train_size=.8, stratification_factor=None)
		return bootsDF

	def run_bootstrap_SHAP_analysis_local(self, bootstrap_iterations=None, method='sample_with_replacement', **kwargs): 

		if bootstrap_iterations==None: 
			bootstrap_iterations = self.bootstrap_iterations

		bootsDF, out = bootstrap_SHAP_analysis(df=self.df, target=self.target, model=xgb.XGBRegressor(**self.hyperparams), explainer_type = 'raw', 
			iterations=bootstrap_iterations, method=method, **kwargs)
		setattr(self, 'bootstrap_SHAP_outputs', out)
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
		if bootstrap_iterations==None: 
			bootstrap_iterations = self.bootstrap_iterations

		bootsDF = parallel_bootstrap_shap_analysis(df=self.df, target=self.target, xgb_hyperparams=hyperparams, explainer_type = 'raw', 
			iterations=bootstrap_iterations, method=method, train_size=train_size, stratification_factor=stratification_factor)
		setattr(self, 'bootsDF', bootsDF)
		return bootsDF

	# def save(self, name="analysis"): 
	# 	import time, pickle
	# 	fileName = self.outputs_dir + "_".join([self.target, name,time.asctime()]) +".p"
	# 	pickle.dump(self, open(fileName, "wb"))
	# 	print("Saving analysis as: ", fileName)
		
### Additional post hoc analyses
def shapModelComparison(dfs, target, hyperparams, analysisName='', save_fig=False, outputs_dir='./', n_folds_SHAP=20, verbose=False):
		"""
		dfs : list of dataframes
			  Features will be ordered by importance of first df

		analysisName: Give a unique name. This will be utilized to save files and load cached outputs.

		"""
		allVars = list(set(np.concatenate([list(dfs[m].columns) for m in range(len(dfs))]).flat)) # 
		allVars.pop(allVars.index(target))
		orderedFeatures_d = {}
		shapDF_d = {}
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
			
			mdlName = exStr + "\nR2_CV: {:.3f}".format(curr_analysis.score_CV_SHAP)
			modelNames.append(mdlName)
			shapDF_d[m] = curr_analysis.shapDF.copy()
			orderedFeatures_d[m]  = copy.deepcopy(curr_analysis.orderedFeatures)

			if m == 0: 
				tmpDF = np.abs(shapDF_d[0].drop(columns = 'expectedValue')).mean().reset_index()
			else: 
				tmpDF = pd.merge(tmpDF, 
						 np.abs(shapDF_d[m].drop(columns = 'expectedValue')).mean().reset_index(), 
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
			fileName = outputs_dir + target + " SHAP model comparison {} {}.png".format(analysisName, time.asctime())
			print("Saved: " + fileName)
			plt.savefig(fileName, bbox_inches='tight')
		return

