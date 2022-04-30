import copy
import tqdm
import sys, time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import time, itertools, copy, tqdm
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from .shap_based_analysis import xgb_shap
from sklearn.metrics import roc_auc_score, r2_score
from lifelines.utils import concordance_index

import xgboost as xgb

prediction_type_d = {'binary_classification': {'score_fcn': roc_auc_score, 'score_name': 'AUROC'},
'regression': {'score_fcn': r2_score, 'score_name': 'R_squared'},
'survival': {'score_fcn': concordance_index, 'score_name': 'C-index'}}


# TODO: Ensure this works for survival

class hyperparameter_selection:
	def __init__(self, df, target, prediction_type: "'binary_classification', 'regression', or 'survival'", max_evals=25, verbose=True, outputs_dir='./'): 
		self.df = df
		self.target = target
		self.mdlFeatures = [feat for feat in df.columns if feat != target]
		self.max_evals=max_evals
		self.verbose = verbose
		self.hyp_perf_df = "Run run_models() first"
		self.mean_abs_shap_vals_df = "Run run_models() first"
		self.iters = 50 #Number of hyperparameters to test to select the most appropriate hyperparameters
		self.outputs_dir = outputs_dir
		self.prediction_type=prediction_type
		self.score_name = prediction_type_d[prediction_type]['score_name']

	def run_models(self, iters=None): 
		if iters == None:
			iters=self.iters
		else:
			iters = iters
		allMeanAbsShapVals = np.array([])
		hyp_perf_df =pd.DataFrame([])
		for r in tqdm.notebook.tqdm(range(iters), desc='Running models', disable=False):
			train_df, test_df = train_test_split(self.df, test_size = 0.2, random_state=r, stratify=self.df[self.target])
			curr_analysis = xgb_shap(df=train_df, target=self.target, n_folds_SHAP=20, max_evals=self.max_evals, verbose=False)
			curr_hyperparams = curr_analysis.tune_model(verbose=self.verbose)
			curr_score_CV= np.mean(curr_analysis.model_performance(n_folds=5, verbose=self.verbose))
			allMeanAbsShapVals = np.append(allMeanAbsShapVals, np.abs(curr_analysis.shap_values).mean(axis=0))
			val_predictions = xgb.XGBModel(**curr_hyperparams).fit(train_df[self.mdlFeatures], train_df[self.target]).predict(test_df[self.mdlFeatures])
			curr_score_val = prediction_type_d[self.prediction_type]['score_fcn'](test_df[self.target], val_predictions)
			hyp_perf_df = pd.concat([hyp_perf_df, 
										 pd.DataFrame(dict(iteration = [r], score_CV=curr_score_CV,
											score_val=curr_score_val, **curr_hyperparams))])
		allMeanAbsShapVals_reshaped = allMeanAbsShapVals.reshape(-1,len(self.mdlFeatures))
		mean_abs_shap_vals_df = pd.DataFrame(allMeanAbsShapVals_reshaped, columns=self.mdlFeatures).reset_index().rename(columns={'index':'iteration'})
		setattr(self, 'mean_abs_shap_vals_df', mean_abs_shap_vals_df)
		setattr(self, 'hyp_perf_df', hyp_perf_df)
		return
	
	def plot_importance_variability(self):
		try:
			assert type(self.hyp_perf_df) == pd.DataFrame
			assert type(self.mean_abs_shap_vals_df) == pd.DataFrame
		except: 
			print("Run 'run_models()' first")
			return

	def plot_SHAP_importance_variability(self):
		try:
			assert type(self.hyp_perf_df) == pd.DataFrame
			assert type(self.mean_abs_shap_vals_df) == pd.DataFrame
		except: 
			print("Running 'run_models()' first")
			self.run_models()
		meanAbsShapValsDF = self.mean_abs_shap_vals_df.copy()
		plt.style.use('seaborn-poster')
		meanAbsShapValsDF.boxplot(column = list(self.mdlFeatures), rot=90, grid=False)
		plt.ylabel('Mean |SHAP| value')
		plt.title('Variability in feature importance estimates\n(using various hyperparameters)')
		
	def plot_SHAP_importance_clusters(self, n_clusters=4):
		try:
			assert type(self.hyp_perf_df) == pd.DataFrame
			assert type(self.mean_abs_shap_vals_df) == pd.DataFrame
		except: 
			print("Running 'run_models()' first")
			self.run_models()
			
		meanAbsShapValsDF = self.mean_abs_shap_vals_df[self.mdlFeatures].copy()
		# hyp_perf_DF = self.hyp_perf_df.copy()
		clusters = KMeans(n_clusters=n_clusters, random_state=1).fit_predict(meanAbsShapValsDF[self.mdlFeatures])
		# hyp_perf_DF['Cluster'] = clusters
		meanAbsShapValsDF['Cluster'] = clusters
		clusteredShapValuesDF = meanAbsShapValsDF.melt(id_vars='Cluster',value_name='SHAP value', var_name='Feature').sort_values(['Feature', 'SHAP value'], ascending=False)
		
		fig=sns.catplot(x='Feature', y='SHAP value', hue='Cluster', data=clusteredShapValuesDF,alpha=1, legend=False, s=10)
		fig.fig.set_size_inches(16,6)
		plt.xticks(rotation=90);
		sns.lineplot(x='Feature', y='SHAP value', hue='Cluster', markers=True, data=clusteredShapValuesDF,alpha=1, ax= fig.ax, palette=sns.color_palette("tab10")[:n_clusters])
		plt.legend(bbox_to_anchor=(1.0, 1), title = 'Cluster', ncol=2)
		plt.ylabel('Mean |SHAP| value')
		plt.setp(fig.ax.get_legend().get_title(), fontsize='14')
	
	def plot_hyperparameter_clusters(self, n_clusters=4):
		try:
			assert type(self.hyp_perf_df) == pd.DataFrame
			assert type(self.mean_abs_shap_vals_df) == pd.DataFrame
		except: 
			print("Running 'run_models()' first")
			self.run_models()
			
		meanAbsShapValsDF = self.mean_abs_shap_vals_df.copy()
		hyp_perf_DF = self.hyp_perf_df.copy()
		clusters = KMeans(n_clusters=n_clusters, random_state=1).fit_predict(meanAbsShapValsDF[self.mdlFeatures])
		hyp_perf_DF['Cluster'] = clusters
		
		tmpDF = hyp_perf_DF[['score_CV', 'eta', 'max_depth', 'min_child_weight', 'reg_alpha', 'reg_lambda', 'subsample']]
		tmpDF = pd.DataFrame(StandardScaler().fit_transform(tmpDF), columns = tmpDF.columns)
		tmpDF['Cluster'] = clusters
		tmpDF= tmpDF.melt(id_vars='Cluster', value_name='value (z-score)').sort_values(['variable', 'value (z-score)'], ascending=[True, False])
		
		fig=sns.catplot(x='variable', y='value (z-score)', hue='Cluster', data=tmpDF,alpha=1, legend=False, s=10)
		fig.fig.set_size_inches(16,6)
		plt.xticks(rotation=90);
		sns.lineplot(x='variable', y='value (z-score)', hue='Cluster', markers=True, data=tmpDF,alpha=1, ax= fig.ax, palette=sns.color_palette("tab10")[:n_clusters])
		plt.legend(bbox_to_anchor=(1.0, 1), title = 'Cluster', ncol=2)
		plt.setp(fig.ax.get_legend().get_title(), fontsize='14')
	
	def get_grouped_hyperparmeter_clusters_df(self, n_clusters=4):
		try:
			assert type(self.hyp_perf_df) == pd.DataFrame
			assert type(self.mean_abs_shap_vals_df) == pd.DataFrame
		except: 
			print("Running 'run_models()' first")
			self.run_models()
			
		meanAbsShapValsDF = self.mean_abs_shap_vals_df.copy()
		hyp_perf_DF = self.hyp_perf_df.copy()
		clusters = KMeans(n_clusters=n_clusters, random_state=1).fit_predict(meanAbsShapValsDF[self.mdlFeatures])
		hyp_perf_DF['Cluster'] = clusters
		hyp_perf_DF.drop(columns='iteration').groupby(['Cluster', 'max_depth']).agg(['mean', 'std', 'count'])
		return hyp_perf_DF.drop(columns='iteration').groupby(['Cluster', 'max_depth']).agg(['mean', 'std', 'count'])
	
	def get_optimal_hyperparameters(self, n_clusters=4, min_viable_hyperparams = 5, verbose=False):
		try:
			assert type(self.hyp_perf_df) == pd.DataFrame
			assert type(self.mean_abs_shap_vals_df) == pd.DataFrame
		except: 
			print("Running 'run_models()' first")
			self.run_models()
			
		meanAbsShapValsDF = self.mean_abs_shap_vals_df.copy()
		hyp_perf_DF = self.hyp_perf_df.copy()
		clusters = KMeans(n_clusters=n_clusters, random_state=1).fit_predict(meanAbsShapValsDF[self.mdlFeatures])
		hyp_perf_DF['Cluster'] = clusters
		viable_clusters = list(hyp_perf_DF.groupby('Cluster').count()[(hyp_perf_DF.groupby('Cluster').count()['iteration']>=min_viable_hyperparams)].reset_index()['Cluster'].to_dict().keys())
		viable_clusters_df = hyp_perf_DF[hyp_perf_DF.Cluster.apply(lambda x: x in viable_clusters)]
		# if self.verbose:
		#     print("Viable hyperparameters")
		#     display(viable_clusters_df.drop(columns='iteration'))
		# import pdb; pdb.set_trace()	
		viable_grouped_df = viable_clusters_df.groupby(['Cluster', 'max_depth']).mean()[viable_clusters_df.groupby(['Cluster', 'max_depth']).count()['iteration']>=min_viable_hyperparams].drop(columns='iteration').reset_index()
		# viable_grouped_df = viable_clusters_df.drop(columns='iteration').groupby(['Cluster', 'max_depth']).mean().reset_index()
		

		max_score = viable_grouped_df.groupby(['Cluster', 'max_depth']).mean()['score_CV'].max()
		try:
			best_cluster_max_depth = viable_grouped_df[viable_grouped_df.score_CV == max_score][['Cluster', 'max_depth']].iloc[0].to_dict()
			if verbose:
				from IPython.display import display
				display(viable_grouped_df[viable_grouped_df.score_CV == max_score][['Cluster', 'max_depth']].iloc[0])
		except:
			print ("Error encountered: Try decreasing n_clusters")
			return
		if self.verbose:
			print("Best cluster & max_depth:", best_cluster_max_depth)
		optimal_hyperparams = (viable_clusters_df[viable_clusters_df.apply(lambda x: (x.Cluster == best_cluster_max_depth['Cluster']) and (x.max_depth==best_cluster_max_depth['max_depth']), axis=1)]
		 .drop(columns='iteration').groupby(['Cluster', 'max_depth'])
		 .mean()).iloc[0,:].to_dict()
		optimal_hyperparams.pop('score_CV')
		optimal_hyperparams.pop('score_val')
		optimal_hyperparams.update({'objective': 'binary:logistic', 'tree_method': 'exact', 'eval_metric': 'error'})
		optimal_hyperparams.update({'max_depth': best_cluster_max_depth['max_depth']})
		return optimal_hyperparams

	def plot_CV_holdout_performance(self, save_fig=True, ax=None, figsize=(6,4), outputs_dir=None, **kwargs):
		import seaborn as sns
		if ax == None:
			plt.figure(figsize=figsize)
		sns.boxplot(x='variable', y='value', data=self.hyp_perf_df.melt(id_vars='iteration', value_vars = ['score_CV','score_val']), **kwargs)
		plt.title('Model performance on cross validation and holdout sets')
		plt.ylabel(self.score_name)
		if save_fig:
			if outputs_dir==None: 
				outputs_dir = self.outputs_dir
			else:
				pass
			file_name= outputs_dir + "Hyperparamerter generalization {}.png".format(self.target)
			plt.savefig(file_name, bbox_inches='tight')
			print("Saved: ", file_name)
			return
		else:
			return ax

	def save(self, name="hyp_search"): 
		import time, pickle
		fileName = self.outputs_dir + "_".join([self.target, name,time.asctime()]) +".p"
		pickle.dump(self, open(fileName, "wb"))
		print("Saving analysis as: ", fileName)
