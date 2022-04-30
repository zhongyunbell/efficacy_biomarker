import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ..clustering import plot_pca, plot_tsne, plot_pacmap
from ..clustering.dsv import plot_pca_3d
from .misc import msnoMatrix, removeDF_outliers
import seaborn as sns
import copy
from .helpers import LazyProperty, nargout
from ..bootstrap_shap.bootstrap_shap_analysis import plot_bootstrapped_feature_dependence, bootstrap_feature_summary_table, generate_bootstrap_summaryDF, plot_bootstrap_summary_table, plot_bootstrapped_feature_dependence_lowess
from ..utils.misc import standard_name, display_name
from collections import Counter
import category_encoders as ce
import shap

class analysis:
	"""
	This object can be useful for general data exploration
	# THis supports clustering, plotting missingness, and variance inflation factor analysis
	"""
	def __init__(self, df, target, meta_df=pd.DataFrame(), remove_outliers=False, outputs_dir = './', categorical_encoding='oridinal', min_category_size=10, verbose=True, **kwargs):
		if 'USUBJID' in df.columns:
			print("Removing 'USUBJID'")
			df = df.drop(columns='USUBJID')
		if remove_outliers:
			if (type(remove_outliers) == int) | (type(remove_outliers) == float):
				z=remove_outliers
			else: 
				z=10
			df = removeDF_outliers(df, z=z)
			self.df = df.copy()
		else:
			self.df = df.copy()
		self.target = target
		X = df[[feat for feat in df.columns if feat != target]]
		y = df[target]
		self.X = X
		self.y = y
		self.verbose=verbose

		for k,v in kwargs.items(): 
			setattr(self, k, v)

		if np.any(df.dtypes==object): 
			cat_cols = list(df.dtypes[df.dtypes==object].index)
			self.cat_cols = cat_cols
			for cat in cat_cols: 
				counts_d = dict(Counter(df[cat]))
				for sub_cat in counts_d:
					if counts_d[sub_cat] < min_category_size:
						if self.verbose:
							print(f"Removing {sub_cat} from {cat}")
						df.loc[df[cat] == sub_cat,cat] = np.nan
						X.loc[X[cat] == sub_cat,cat] = np.nan

			if categorical_encoding=='oridinal':
				print('Encoding categorical columns in order of prevalance: ', cat_cols)   
				def get_count_oridinal_mapping(cat_cols : "e.g: ['RACE', 'REGION']"):
					# cat_cols : list of categorical columns
					# min_cat_size: minimum size of a group remaining is grouped into an 'OTHER' category
					mapping = []   
					for col in cat_cols:
						curr_count_d = dict(Counter(df[col]).most_common(50))
						i=1
						curr_cats = list(curr_count_d.keys())
						# curr_order_d = {}
						for k in curr_cats:
							# if curr_count_d[k] < min_cat_size:
							# 	print("Removing {} : {} due to low counts ({:.0f})".format(col,k, curr_count_d[k]))
							# 	del curr_count_d[k]
							# else:
							curr_count_d[k]=i #count is being converted to order
							i+=1
							# curr_count_d['OTHER']=i
						curr_count_d[None]=0
						mapping.append({'col': col, 'mapping':curr_count_d})
					return mapping

				categorical_mapping = get_count_oridinal_mapping(cat_cols=cat_cols)
				encoder = ce.OrdinalEncoder(cols=cat_cols, mapping=categorical_mapping)
				cat_mapping2 = {} # reprocess this so it's easier to use later
				for i in categorical_mapping:
					cat_mapping2[i['col']]={}
					for k2,v2 in i['mapping'].items():
						cat_mapping2[i['col']][v2]=k2
				print(cat_mapping2)
				self.categorical_mapping=cat_mapping2
				encoder.fit(X, y)
				X = encoder.transform(X)
				self.X = X
				for col in cat_cols:
					df[col] = X[col]
				self.df=df
			else: #if categorical_encoding == 'ohe':
				# cat_cols = list(df.dtypes[df.dtypes==object].index)
				encoder = ce.OneHotEncoder(cols=cat_cols)
				# X = df[[feat for feat in df.columns if feat != target]]
				# y = df[target]
				encoder.fit(X, y)
				X = encoder.transform(X, y)
				cat_mapping = {} # reprocess this so it's easier to use later
				for i in encoder.category_mapping:
					cat_mapping[i['col']]={}
					for k2,v2 in i['mapping'].items():
						cat_mapping[i['col']][v2]=k2
				category_baselines ={}
				
				for cat in cat_cols:
					category_baselines[cat] = standard_name(cat+ " " + Counter(df[cat]).most_common(1)[0][0])
				print("Dropping baselines (most common) for OHE of categorical variables: ", category_baselines)

				# Rename categorical columns to something sensible
				for cat in cat_mapping: 
					cols2rename = [col for col in X.columns if col.startswith(cat+"_")]
					for col in cols2rename:
						if type(cat_mapping[cat][int(col.split("_")[-1])]) == float:
							# if np.isnan(cat_mapping[cat][int(col.split("_")[-1])]):
							X.drop(columns = col, inplace=True)
						else: 
							X.rename(columns={col : standard_name(cat+ " " +cat_mapping[cat][int(col.split("_")[-1])])},inplace=True)

						# Error could arise if there's something odd with column names (i.e. other variables start with the same category name) 
				# Drop "baseline" subgroup (i.e. the most frequent)
				X.drop(columns=list(category_baselines.values()), inplace=True)
				df = X.copy()
				df[target]=y.copy()
				self.X = X
				self.category_baselines = category_baselines
				self.df=df
		else:
			self.cat_cols = []
			self.categorical_mapping={}
		
		self.target = target
		self.mdlFeatures = [feat for feat in df.columns if feat != target]
		import os, glob
		if len(glob.glob(outputs_dir))==0:
			os.system('mkdir '+ outputs_dir)

		meta_columns = list(set(meta_df.columns).difference(self.df.columns))
		self.meta_columns = meta_columns
		self.meta_df = meta_df[meta_columns]
		self.outputs_dir = outputs_dir

	def add_meta_df(self, meta_df):
		meta_columns = list(set(meta_df.columns).difference(self.df.columns))
		setattr(self, 'meta_columns', meta_columns)
		setattr(self, 'meta_df', meta_df[meta_columns])
		return 
		
	def redefine_df(self, df):
		self.X = df[[feat for feat in df.columns if feat != self.target]]
		self.y = df[self.target]
		self.df = df
		self.mdlFeatures = [feat for feat in df.columns if feat != self.target]
		try: 
			del self.orderedFeatures
		except:
			pass
		try:
			del self.shap_values
		except:
			pass
		try:
			del self.SHAP_outputs
		except:
			pass
		try:
			del self.shapDF
		except:
			pass

	def plot_missingness(self, show=True, colorBy=None, figsize=(30,8), save_fig=False, outputs_dir=None, **kwargs):
		fig = msnoMatrix(self.df, colorBy = colorBy, figsize=figsize, **kwargs)
		if save_fig:
			if type(outputs_dir)==type(None): 
				outputs_dir = self.outputs_dir
			else:
				pass
			file_name= outputs_dir + "SHAP summary plot {}.png".format(self.target)
			plt.savefig(file_name, bbox_inches='tight')
			print("Saved: ", file_name)	
		if show: 
			pass
		else:
			plt.close()
			return fig

	def clustermap(self, type='hvplot', height=600, width=600, use_dendogram=True):
		""" 
		type : 'seaborn' or 'hvplot'
		"""
		if type =="seaborn":
			cm = sns.clustermap(self.df.corr(), cmap='coolwarm', linewidths=.1)
		elif type=="hvplot":
			import hvplot.pandas
			if use_dendogram:
				cm = sns.clustermap(self.df.corr(), cmap='coolwarm', linewidths=.1);
				col_order=cm.data2d.columns
				plt.close()
			else:
				col_order = self.mdlFeatures
			return self.df[col_order].corr().hvplot.heatmap(cmap="coolwarm", rot=90, height=height, width = width, tools=['crosshair'])
	
	def correlation_martix(self, feature=None, features=None, return_styled = True):
		"""
		Optional arguments can be passed: 
		features : list of features to see correlation matrix (using seaborn dendogram)
		feature  : feature-centric correlation vector is returned
		"""
		#  numeric_cols = [feat for feat,typ in self.df.dtypes.to_dict().items() if typ in [np.dtype('float64'), np.dtype('int64')]]
		tmpDF = self.df.copy()
		if type(features)==type(None):
			pass
		else:
			tmpDF = tmpDF[features]
		corrDF = tmpDF[tmpDF.columns[tmpDF.nunique()>1]].corr()
		
		if type(feature)==type(None):
			pass
		else:
			if return_styled:
				return corrDF[[feature]].sort_values(feature).style.background_gradient(low=1, high=.1,vmax=1, cmap='RdBu_r')
			else:
				return corrDF[[feature]] 
		
		cm = sns.clustermap(corrDF, cmap='coolwarm', linewidths=.1);
		print(list(cm.data2d.columns))
		plt.close()
		if return_styled:
			return tmpDF[cm.data2d.columns].corr().style.background_gradient(low=1, high=.1,vmax=1, cmap='RdBu_r')
		else:
			return tmpDF[cm.data2d.columns].corr()

	def plot_pacmap(self, table_vars=None, n_clusters=5, index_labels=None, index_name='index', color_by='prediction', return_outputs=False, pacmap_params=dict(), **kwargs):
		""" 
		index_labels : column of dataframe or list of strings for each instance 
		index_name   : Name for the rows in the dataset (default data_label)
		table_vars: None (defaults to the top 5 features), 
					if table_vars is an int: show the top {int} features
					pass [] to forgo the table
		color_by : Can also try 'Cluster' (from Kmeans) or the target variable 
		default is 'prediction'
		pacmap_params:
		n_dims=2,
		n_neighbors=10,
		MN_ratio=0.5,
		FP_ratio=2.0,
		pair_neighbors=None,
		pair_MN=None,
		pair_FP=None,
		distance='euclidean',
		lr=1.0,
		num_iters=450,
		verbose=False,
		apply_pca=True,
		intermediate=False,
		random_state=None,
		kwargs:
		plot_width=750, plot_height=plot_height, table_width=725, table_height=table_height, n_clusters=5, ms=10
		
		"""
		df = self.df.copy()
		df['index']=list(self.df.index)
		try: 
			features = self.orderedFeatures
		except: 
			features = self.mdlFeatures
		if color_by == 'prediction':
			df['prediction'] = np.round(self.predictionsCV_SHAP, 2)
		if type(table_vars) == type(None):
			table_vars = [color_by]
			table_vars.extend(features[:5])
		elif type(table_vars)==int:
			n_features = table_vars
			table_vars = [color_by]
			table_vars.extend(features[:n_features])
		elif len(table_vars)==0:
			table_vars = None

		if type(index_labels) != type(None):
			if index_name == 'index': 
				index_labels=df['index']
			else: 
				df[index_name] = index_labels
				table_vars.insert(0, index_name)
		if table_vars: 
			table_vars.insert(0, 'index')
		# features refers to features used to cluster instances
		return plot_pacmap(df=df, features=features, table_vars=table_vars, data_label=index_name, color_by=color_by, return_outputs=return_outputs, pacmap_params=pacmap_params, **kwargs)
			
	def plot_tsne(self, features=None, table_vars=None, n_clusters=5, index_labels=None, index_name='index', color_by='prediction', return_outputs=False, tsne_params=dict(init = 'pca', n_iter=1000, perplexity=30, learning_rate=200), **kwargs):
		""" 
		index_labels : column of dataframe or list of strings for each instance 
		index_name   : Name for the rows in the dataset (default data_label)
		table_vars: None (defaults to the top 5 features), 
					if table_vars is an int: show the top {int} features
					pass [] to forgo the table
		color_by : Can also try 'Cluster' (from Kmeans) or the target variable 
		default is 'prediction'
		tsne_params :   dictionary of tSNE params: default=dict() , which yields dict(init = 'pca', n_iter=1000, perplexity=30)
						perplexity: float, default=30.0
						learning_rate: float, default=200.0
						n_iter: int, default=1000
						init : {‘random’, ‘pca’}
						random_state : int, default=1
						More details: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html?highlight=tsne#sklearn.manifold.TSNE
		kwargs:
		plot_width=750, plot_height=plot_height, table_width=725, table_height=table_height, n_clusters=5, ms=10
		
		"""
		df = self.df.copy()
		df['index']=list(self.df.index)
		if type(features) == type(None):
			try: 
				features = self.mdlFeatures
			except: 
				features = self.orderedFeatures
		if color_by == 'prediction':
			df['prediction'] = np.round(self.predictionsCV_SHAP, 2)
		if type(table_vars) == type(None):
			table_vars = [color_by]
			table_vars.extend(features[:5])
		elif type(table_vars)==int:
			n_features = table_vars
			table_vars = [color_by]
			table_vars.extend(features[:n_features])
		elif len(table_vars)==0:
			table_vars = None

		if type(index_labels) != type(None):
			if index_name == 'index': 
				index_labels=df['index']
			else: 
				df[index_name] = index_labels
				table_vars.insert(0, index_name)
		if table_vars: 
			table_vars.insert(0, 'index')
		# features refers to features used to cluster instances
		return plot_tsne(df=df, features=features, table_vars=table_vars, data_label=index_name, color_by=color_by, return_outputs=return_outputs, tsne_params=tsne_params, **kwargs)
	
	def plot_pca(self, features=None, table_vars=None, n_clusters=5, index_labels=None, index_name='index', color_by='prediction', return_outputs=False, **kwargs):
		""" 
		index_labels : column of dataframe or list of strings for each instance 
		index_name   : Name for the rows in the dataset (default data_label)
		table_vars: None (defaults to the top 5 features)
		color_by : Can also try 'Cluster' (from Kmeans) or the target variable 
		default is 'prediction'
		kwargs:
		plot_width=750, plot_height=plot_height, 
		table_width=725, table_height=table_height, biplot_coeff=- 6, n_clusters=5, ms=10
		"""
		df = self.df.copy()
		df['index']=list(self.df.index)
		if type(features) == type(None):
			try: 
				features = self.mdlFeatures
			except: 
				features = self.orderedFeatures

		if color_by == 'prediction':
			df['prediction'] = np.round(self.predictionsCV_SHAP, 2)
		if type(table_vars) == type(None):
			table_vars = [color_by]
			table_vars.extend(features[:5])
		elif type(table_vars)==int:
			n_features = table_vars
			table_vars = [color_by]
			table_vars.extend(features[:n_features])
		elif len(table_vars)==0:
			table_vars = None

		if type(index_labels) != type(None):
			if index_name == 'index': 
				index_labels=df['index']
			else: 
				df[index_name] = index_labels
				table_vars.insert(0, index_name)
		if table_vars: 
			table_vars.insert(0, 'index')
		# features refers to features used to cluster instances
		return plot_pca(df=df, features=features, table_vars=table_vars, data_label=index_name, color_by=color_by, return_outputs=return_outputs, **kwargs)

	def plot_pca_3d(self, n_clusters=5, color_by='prediction', features=None, **kwargs):
		""" 
		index_labels : column of dataframe or list of strings for each instance 
		index_name   : Name for the rows in the dataset (default data_label)
		table_vars: None (defaults to the top 5 features)
		color_by : Can also try 'Cluster' (from Kmeans) or the target variable 
		default is 'prediction'
		kwargs:
		biplot_coeff=-6, n_clusters=5, ms=10
		"""
		df = self.df.copy()
		df['index']=list(self.df.index)
		if type(features)==type(None):
			try: 
				features = self.orderedFeatures
			except: 
				features = self.mdlFeatures
		if color_by == 'prediction':
			df['prediction'] = np.round(self.predictionsCV_SHAP, 2)
		# features refers to features used to cluster instances
		return plot_pca_3d(df=df, features=features, color_by=color_by, **kwargs)

	def variance_inflation_factor_analysis(self, return_styled=True, feature_order='importance'):
		"""
		feature_order: 'correlation' - clustered by correlation or Default: 'importance' (i.e. SHAP importance)
		"""
		#import pandas as pd
		#import numpy as np
		from patsy import dmatrices
		import statsmodels.api as sm
		from statsmodels.stats.outliers_influence import variance_inflation_factor
		def order_features_by_correlation(df): 
			cm = sns.clustermap(df.corr(), cmap='coolwarm', linewidths=.1);
			plt.close()
			return list(cm.data2d.columns)
		
		if feature_order == 'correlation':
			features = order_features_by_correlation(self.df)
		elif feature_order == 'importance': 
			try: 
				features = self.orderedFeatures
			except: 
				print("Cannot order features by importance. Sorting by correlation clusters")
				features = order_features_by_correlation(self.df)
		tmpDF = self.df.copy()
		renamed_features = [s.replace(' ', '_') for s in features]
		renamed_features = [s.replace('-', '_') for s in renamed_features]
		tmpDF=tmpDF.rename(columns = dict(zip(features, renamed_features)))
		y, X = dmatrices(self.target+' ~' + " + ".join(renamed_features), tmpDF.fillna(tmpDF.median()), return_type='dataframe')
		vif = pd.DataFrame()
		vif["feature"] = X.columns
		vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
		if return_styled:
			return vif.iloc[1::,:].style.background_gradient(low=0, high=.1,vmax=20, cmap='Reds')
		else:
			print("Style using: .style.background_gradient(low=0, high=.1,vmax=20, cmap='Reds').highlight_null('gray')")
			return vif.iloc[1::,:]	

	def copy(self):
		import copy
		return copy.deepcopy(self)

	def save(self, name=None, save_type='lean', include_time=True): 
		if type(name)==type(None): 
			name = str(self.__class__).split("'")[1].split(".")[-1]
		import time, pickle
		if include_time:
			fileName = self.outputs_dir + "_".join([self.target, name,time.asctime()]) +".p"
		else:
			fileName = self.outputs_dir + "_".join([self.target, name]) +".p"
		if save_type == 'lean':
			from ..utils.misc import remove_explainers_models
			lean_copy = remove_explainers_models(self)
			pickle.dump(lean_copy, open(fileName, "wb"))
		else:
			pickle.dump(self, open(fileName, "wb"))
		print("Saved as: ", fileName)

class shap_analysis: 
	def __init__():
		pass
	
	def shap_summary_plots(self, selectionVec = None, show=True, figsize=(40,12), save_fig=False, outputs_dir=None, **kwargs): 
		if type(selectionVec) == type(None): 
			selectionVec = self.df.index.notna() # all rows
			selected = 'all'
		else: 
			selected = 'subset'
		fig = plt.figure(figsize=figsize)
		#fig.subplot(1,2,1)
		gs = fig.add_gridspec(1, 3)
		fig.add_subplot(gs[0, 0])
		shap.summary_plot(self.shap_values[selectionVec,:], features=self.df.loc[selectionVec,:].drop(columns=self.target), show=False, plot_size=(figsize[0]/3,figsize[1]), plot_type='bar', **kwargs)
		#plt.subplot(1,2,2)
		fig.add_subplot(gs[0, 1:])
		shap.summary_plot(self.shap_values[selectionVec,:], features=self.df.loc[selectionVec,:].drop(columns=self.target), show=False, plot_size=(2*figsize[0]/3,figsize[1]), **kwargs)
		plt.tight_layout()
		if selected == 'all':
			fig.text(.1, 1, "{} (CV): {:.3f}".format(self.score_name, self.score_CV_SHAP), ha='left', fontsize = 18)
		if save_fig:
			if type(outputs_dir)==type(None): 
				outputs_dir = self.outputs_dir
			else:
				pass
			file_name= outputs_dir + "SHAP summary plot {}.png".format(self.target)
			plt.savefig(file_name, bbox_inches='tight')
			print("Saved: ", file_name)
			return
		if show: 
			pass
		else:
			plt.close()
			return fig

	def shap_dependence_plots(self, show=True, save_fig=False, outputs_dir=None):
		orderedFeatures_top_9 = self.orderedFeatures[:9]
		fig = plt.figure(figsize=(16,12))
		ylims = [self.shap_values.min(), self.shap_values.max()]
		for i,feature in enumerate(orderedFeatures_top_9): 
			ax = fig.add_subplot(331+i)
			shap.dependence_plot(self.mdlFeatures.index(feature), self.shap_values, self.df[self.mdlFeatures],ax=ax, show=False)
			ax.set_ylim(ylims)
			xlims = ax.get_xlim()
			ax.hlines(y=0, xmin =xlims[0], xmax= xlims[1], ls='--')
			
		plt.tight_layout()
		if save_fig:
			if outputs_dir==None: 
				outputs_dir = self.outputs_dir
			else:
				pass
			file_name= outputs_dir + "SHAP dependence plots {}.png".format(self.target)
			plt.savefig(file_name)
			print("Saved: ", file_name)
			return
		if show: 
			pass
		else:
			plt.close()
			return fig

	def dependence_plot(self,feature, interaction_feature='auto', show=True, save_fig=False, outputs_dir=None, **kwargs):
		if interaction_feature == 'auto':
			interaction_index = 'auto'
		else:
			interaction_index = self.mdlFeatures.index(interaction_feature)
		shap.dependence_plot(self.mdlFeatures.index(feature), self.shap_values, self.df[self.mdlFeatures], interaction_index=interaction_index, show=show, **kwargs)
		return 
				
	def force_plot(self, Idx):
		pass
		return shap.force_plot(base_value=self.SHAP_outputs['explainer_d'][Idx].expected_value, 
                shap_values = self.shap_values[Idx,:], 
                features = self.df.drop(columns=self.target).iloc[Idx,:])		

	def shap_interaction_summary(self, feature, selectionVec = None, show=True, drop_cols=[], max_display= 15, figsize=(10,6), save_fig=False, outputs_dir=None, **kwargs):
		# feature = 'CTR1'
		if not hasattr(self, 'shap_interaction_values'):
			print('Calculating interaction values first')
			self.shap_interaction_values
		# figsize=(10,6)

		fig = plt.figure(figsize=figsize)
		if type(selectionVec) == type(None): 
			selectionVec = self.df.index.notna() # all rows
		gs = fig.add_gridspec(1, 1)
		fig.add_subplot(gs[0, 0])
		tmpDF = pd.DataFrame(self.shap_interaction_values[selectionVec, self.mdlFeatures.index(feature)], columns=self.mdlFeatures).drop(columns=drop_cols)
		drop_cols.append(self.target)
		shap.summary_plot(shap_values = tmpDF.values, 
						  features=self.df.loc[selectionVec,:].drop(columns=drop_cols), 
						  max_display=max_display, show=False, plot_size=figsize, **kwargs)
		fig.text(.1, 1, "Interactions with {}".format(feature), ha='left', fontsize = 18)
		plt.tight_layout()
		if save_fig:
			if outputs_dir==None: 
				outputs_dir = self.outputs_dir
			else:
				pass
			file_name= outputs_dir + "SHAP {} interatction summary plot {}.png".format(feature, self.target)
			plt.savefig(file_name)
			print("Saved: ", file_name)
			return
		if show: 
			pass
		else:
			plt.close()
			return fig

	def shap_exposure_impacts(self, exposure_var=None, figsize=(6,4), save_fig=False, return_fig=False, selectionVec=None, show_std=True):
		if type(exposure_var) == type(None):
			exposure_var = self.exposure_var
		if type(selectionVec) == type(None): 
			selectionVec = self.df.index.notna()
		if ('shap_exposure_interaction_prob_df' in self.__dir__()) & (exposure_var == self.exposure_var):
			y_scale = '∆ probability' 
			cols2use = self.shap_exposure_interaction_prob_df.drop(columns=exposure_var).columns
			# shap.summary_plot(union_mdl.shap_exposure_interaction_prob_df.drop(columns=dropCols).values, union_mdl.df[cols2use], plot_type='bar')
			mean_2nd = self.shap_exposure_interaction_prob_df.loc[selectionVec,:].drop(columns=exposure_var).agg(lambda x : np.nanmean(np.abs(x)))
			sort_order = mean_2nd.sort_values(ascending=False).index
			mean_2nd = mean_2nd.loc[sort_order]
			std_2nd =self.shap_exposure_interaction_prob_df.loc[selectionVec,:].drop(columns=exposure_var).agg(lambda x : np.nanstd(np.abs(x))).loc[sort_order]
			second_order_effects_df = pd.concat([mean_2nd, std_2nd], axis=1).reset_index()
			second_order_effects_df.columns=['covariate','mean impact', 'std']
			second_order_effects_df['covariate impact on ER'] = second_order_effects_df.apply(lambda x: "{:.4f}±{:.4f}".format(x['mean impact'],x['std']), axis=1)
		else: 
			if self.hyperparams['objective'] == 'binary:logistic':
				print('Interaction values in probability scale were not found; using log-odds scale')
				print('Hint: self.generate_shap_exposure_interaction_prob_df(exposure_var)')
				y_scale='log-odds'
			else: 
				y_scale='∆ prediction'
			shap_interaction_df = pd.DataFrame(self.shap_interaction_values[selectionVec, self.mdlFeatures.index(exposure_var)], 
											columns=self.mdlFeatures)
			cols2use = shap_interaction_df.drop(columns=exposure_var).columns
			mean_2nd = shap_interaction_df.drop(columns=exposure_var).agg(lambda x : np.nanmean(np.abs(x)))
			sort_order = mean_2nd.sort_values(ascending=False).index
			mean_2nd = mean_2nd.loc[sort_order]
			std_2nd =shap_interaction_df.drop(columns=exposure_var).agg(lambda x : np.nanstd(np.abs(x))).loc[sort_order]
			second_order_effects_df = pd.concat([mean_2nd, std_2nd], axis=1).reset_index()
			second_order_effects_df.columns=['covariate','mean impact', 'std']
			second_order_effects_df['covariate impact on ER'] = second_order_effects_df.apply(lambda x: "{:.4f}±{:.4f}".format(x['mean impact'],x['std']), axis=1)

		fig = plt.figure(figsize=figsize)
		plt.errorbar(x=range(len(mean_2nd)), y=mean_2nd, yerr=std_2nd if show_std else 0, fmt='ok')
		plt.xticks(range(len(mean_2nd)), sort_order, ha='right', rotation=45)
		plt.ylabel(f"Covariate impact on {exposure_var}-{self.target} relationship\n(mean {'± std ' if show_std else ''}|SHAP interaction value|)\n({y_scale})")
		plt.title(f"Covariate impacts on {exposure_var}-{self.target} relationship")
		plt.grid(alpha=.5)
		
		if save_fig:
			file_name= self.outputs_dir + f"Covariate impacts on {exposure_var} {self.target} relationship.png"
			plt.savefig(file_name, bbox_inches='tight', dpi=300)
			print("Saved: ", file_name)
		if return_fig:
			n_out = nargout()
			if n_out == 2:
				return fig, second_order_effects_df
			else:
				return fig
		else:
			return second_order_effects_df
			
class bootstrap_summary:
	def __init__():
		pass

	@LazyProperty
	def mean_expected_value_bootstrap(self):
		out = self.bootsDF.groupby('bootsIteration').agg({'expectedValue':np.mean}).mean()[0]
		return out

	def plot_bootstrapped_feature_dependence(self, x_feature, units='', shap_features=[], color_by=None, bins=None, level_type='categorical', nQuantiles = 10, yaxis_label='∆ prediction', figsize = (16,5), save_fig=False, outputs_dir=None, return_summary_table=False, return_fig=False, ylims=None, xlims=None, **kwargs):
		""" 
		This function flexibly plots bootstrapped feature dependence plots:

		plot_feature_dependence(bootsDF=bootsDF, df=df, x_feature='CTR1', shap_feature=['CTR1', 'MAINT_TRT'], color_by='MAINT_TRT',level_type='categorical')

		color_by : must be a feature in df; default is None, which colors by quantiles
		level_type : defines how to color the feature 'sequential' or 'categorical'; default: 'categorical'
		x_feature: The feature value to be plotted on the x-axis
		shap_features: defines what to shap values to combine on the y-axis (e.g. ['Cmin', 'Cmax', 'AUC']), 
			by default, it is simply the SHAP values of x_feature
		
		y-axis: Label for y-axis
		Note: if the y-axis starts with 'Adjusted', the CIs will also incorporate variability of expected value

		"""
		if outputs_dir == None: 
			outputs_dir = self.outputs_dir
		else:
			pass
		categorical_mapping = None
		if 'cat_cols' in dir(self):
			if x_feature in self.cat_cols:
				categorical_mapping=self.categorical_mapping
		return plot_bootstrapped_feature_dependence(bootsDF=self.bootsDF, df=self.df, x_feature=x_feature, shap_features=shap_features, color_by=color_by, level_type=level_type, save_fig=save_fig, units=units, nQuantiles = nQuantiles, yaxis_label=yaxis_label, figsize=figsize, outputs_dir=outputs_dir, return_summary_table=return_summary_table, return_fig=return_fig, bins=bins, ylims=ylims, xlims=xlims, categorical_mapping=categorical_mapping, **kwargs)
	
	def plot_bootstrapped_feature_dependence_lowess(self, x_feature, yaxis_label='∆ prediction', save_fig=False, outputs_dir=None, **kwargs):
		"""
		kwargs options: 
		figsize=(3.5,5),color='g', conf=.05, show_points=True, fill_alpha=.33, marker_alpha=.1, marker_size=10, grid_alpha=.5, label=None, ax=None, ylims=None, xlims=None
		"""
		if outputs_dir == None: 
			outputs_dir = self.outputs_dir
		else:
			pass
		return plot_bootstrapped_feature_dependence_lowess(bootsDF=self.bootsDF, df=self.df, x_feature=x_feature, yaxis_label=yaxis_label, save_fig=save_fig, outputs_dir=outputs_dir, **kwargs)
	
	def bootstrap_feature_summary_table(self, feature, shap_features=[], bins=None, nQuantiles=10, metric='∆ prediction', ci = .95, target=None):
		"""
		bins : list or 'binary' (default None : quantile bins selected using nQuantiles)
		"""
		categorical_mapping = None
		if type(bins) ==type(None):
			if len(set(self.df[feature].unique()).difference([0,1]))==0:
				bins = 'binary'
			elif 'cat_cols' in dir(self):
				if feature in self.cat_cols:
					bins = 'categorical'
					categorical_mapping=self.categorical_mapping
		return bootstrap_feature_summary_table(feature=feature, bootsDF=self.bootsDF, df=self.df, shap_features=shap_features, bins=bins, nQuantiles=nQuantiles, metric=metric, ci=ci, categorical_mapping=categorical_mapping, target=target)

	def generate_bootstrap_summaryDF(self, summary_params=None, return_styled_df=False, metric = "∆Prediction", n_features=10, nQuantiles=4):
		"""
		Generates an overall summary of analysis given manually selected parameters:
		e.g.: 
		if metric.starstwith('adjusted): shows the adjusted prediction rather than ∆ prediction
		summary_params = {'Cminsd': {'nQuantiles': 4}, 
						'BHBA1C': {'bins':[2.5,5.7,6.4,12.2]},
						'BGLUC': {'bins': [60,100,120, 130, 350]},
						'BHDL': {'bins': [0,40,60, 210]},
						'Region_EUROPE': {},# binary
						'Race_BLACK_OR_AFRICAN_AMERICAN' : {}
						}
		"""
		if type(summary_params) == type(None):
			if 'bootstrap_summaryDF' in self.__dir__():
				bootstrap_summary = self.bootstrap_summaryDF
				print('No summary params were found so using cached value')
			else:
				bootstrap_summary = generate_bootstrap_summaryDF(self, summary_params=summary_params, metric=metric, target=self.target, n_features=n_features, nQuantiles=nQuantiles)
		else:
			bootstrap_summary = generate_bootstrap_summaryDF(self, summary_params=summary_params, metric=metric, target=self.target, n_features=n_features, nQuantiles=nQuantiles)
		if type(bootstrap_summary) == pd.core.frame.DataFrame: # May not be needed anymore
			setattr(self, 'bootstrap_summaryDF', bootstrap_summary) # store the latest bootstrap_summaryDF
			setattr(self, 'summary_params', summary_params) # store the latest bootstrap_summaryDF
			print("Saved self.bootstrap_summaryDF. To remove it, del self.bootstrap_summaryDF")
			print("Saved self.summary_params. To remove it, del self.bootstrap_summaryDF")
		if not return_styled_df:
			return bootstrap_summary
		else:
			cols2use = ['Variable', 'Value', "N", "mean "+metric, metric]
			if self.target in bootstrap_summary.columns:
				cols2use.extend(['mean '+ self.target, self.target])
			curr_summary_styled = bootstrap_summary[cols2use]
			indecies = pd.MultiIndex.from_arrays([curr_summary_styled["Variable"].values,curr_summary_styled["Value"].values],
									names=['Variable', 'Value'])
			curr_summary_styled.index = indecies
			cols2use = ["N", 'mean '+metric, metric]
			if self.target in bootstrap_summary.columns:
				cols2use.extend(['mean '+ self.target, self.target])
			curr_summary_styled = (curr_summary_styled[cols2use]
								.style.background_gradient(axis=None, cmap='RdBu_r', subset=["mean "+metric], vmin=-.15,vmax=.15)
								.set_table_styles([{'selector': 'th', 'props': [('font-size', '8pt'),('border-style','solid'),('border-width','1px')]}]))
			if self.target in bootstrap_summary.columns:
				target_min= self.df[self.target].mean() - self.df[self.target].std()
				target_max= self.df[self.target].mean() + self.df[self.target].std()
				curr_summary_styled = curr_summary_styled.background_gradient(axis=None, cmap='RdBu_r', subset=["mean "+self.target], vmin=target_min,vmax=target_max)

			return curr_summary_styled

	def plot_bootstrap_summary_table(self, summary_params=None, metric = "∆Prediction", n_features=5, nQuantiles=4, **kwargs):
		"""
		nQuantiles: only utilized if summary_params are not supplied and there's no cached bootstrap_summaryDF
		kwargs: kwargs for plot_bootstrap_summary_table fcn: 
						vline=0, v_line_label = '',
						   ax=None,color='k', show=True, title="Impact of covariates on Y",
								 x_axis='∆Prediction', label=None, offset=0, figsize=None, xlims=None
		"""
		try: #Use existing bootstrap_summaryDF if available
			bootstrap_summaryDF = self.bootstrap_summaryDF 
			print('Genrating figure using cached bootstrap_summaryDF')
		except: 
			bootstrap_summaryDF = self.generate_bootstrap_summaryDF(summary_params, return_styled_df=False, metric = metric, n_features=n_features, nQuantiles=nQuantiles)
		return plot_bootstrap_summary_table(bootstrap_summaryDF, title="Impact of covariates on predictions\nexpected value: {:.2f}".format(self.mean_expected_value_bootstrap), **kwargs)

	
