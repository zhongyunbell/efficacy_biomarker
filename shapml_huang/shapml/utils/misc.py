from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
import seaborn as sns

def msnoMatrix(df, colorBy, sortBy = [], title='', x_offset = -1, figsize=(30,8), y_labelpad=100, dropCols=[], annotations=None, y_annotation_offset=10, bins=None):
	""" 
	Show dataset missingness
	Syntax: 
	msnoMatrix(df_orig[cols2use], colorBy='IPADOSE',sortBy=['PHASE'], title='IPAT dataset')
	annotations can be supplied using this syntax: 
	annotations = {'trail design': ['IPADOSE'],
	'demographics': ['REGION', 'RACE',  'TOBHX'],
	'biometrics': ['BAGE', 'BHT', 'BWT', 'BBMI'],
	'physiology': ['CRCL', 'EGFR'], 
	'biomarkers': ['BALB', 'BALP', 'BALT', 'BAMYLASE', 'BAST', 'BBICARB', 'BCA', 'BCHOL', 'BCREAT', 'BGLUC', 'BHBA1C','BHCT', 'BHDL', 'BHGB', 'BLDL', 'BLIPASE', 'BMG', 'BPLAT', 'BK', 'BPSA', 'BINR', 'BAPTT', 'BRBC', 'BSODIUM', 'BBILI', 'BPROT', 'BTRIG', 'BWBC', 'BBASO', 'BEOS', 'BLYM', 'BMONO', 'TPROT', 'PSA_Cycle2'], 
	'disease history':['BECOG', 'PTENGSFL', 'HINCI', 'METSITES', 'PCHMSD', 'PCHISD', 'PCHMETD', 'PRGENRL', 'BPIBL', 'ONSURG', 'ONRADIO', 'FUSURG', 'FURADIO', 'PRPROST', 'PRORCH', 'PRRADIO', 'PRSTX', 'PRDOX', 'PRSTMHS', 'PRSTNMS', 'PRSTMCR', 'PRSTOTH', 'BPSG3FL', 'GLEASON'] , 
	'target': ['OS']}
	"""
	import missingno as msno
	import seaborn as sns
	from collections import Counter
	import copy
	import matplotlib.pyplot as plt
	import numpy as np

	sortBy = copy.deepcopy(sortBy)

	if type(colorBy)==type(None):
		sortBy.insert(0,colorBy)
		df = df.sort_values(sortBy)
	else:
		sortBy.insert(0,colorBy)
		df = df.sort_values(sortBy)

	if type(annotations) == dict:
		import itertools 
		cols2use = list(itertools.chain(*list(annotations.values())))
		if colorBy not in cols2use:
			cols2use.append(colorBy)
	else: 
		cols2use = list(df.columns)
		
	if type(colorBy)!=type(None):
		if df[colorBy].nunique()>8:
			if type(bins) == type(None):
				nQuantiles=5
				bins=[df[colorBy].quantile(q) for q in np.round(np.linspace(0,1, nQuantiles+1),2)]
				bins[0] = bins[0]-.1
			df[colorBy+"_binned"] = pd.cut(df[colorBy], bins = bins, include_lowest=True, duplicates = 'drop')
			colorBy= colorBy+"_binned"
	fig,ax =plt.subplots(1,1, figsize=figsize)
	msno.matrix(df[cols2use].drop(columns=dropCols), ax=ax, labels=True, sparkline=False)

	if type(colorBy)!=type(None):
		colorBy_d = Counter(df[colorBy])
		colorBys = [c for c in df[colorBy].unique()]
		hs = [colorBy_d[c] for c in colorBys]
		ys = np.cumsum(hs)
		xs=[0]*len(ys)
		ws= [df.shape[1]]*len(ys)
		colors = sns.palettes.color_palette(n_colors=len(colorBys))
		for x,y,h,w,c,feat in zip(xs,ys,hs,ws,colors,colorBys):
			ax.add_patch(plt.Rectangle((x-.5,y), width=w, height=-h, alpha=.3, fc=c))
			ax.text(x+x_offset,y-h/2,feat, size=18, ha='right', color='k')
		ax.set_title(title+"\nsorted by "+ ", ".join(sortBy), size=18)
		ax.set_ylabel(colorBy, size=24, color='k', labelpad=y_labelpad)
		xlims = ax.get_xlim()

	if type(annotations) == dict:
		annotation_colors= sns.color_palette('tab10', n_colors=len(annotations))
		next_x_pos = 0
		for idx,topic in enumerate(annotations): 
			print(topic)
			print(annotations[topic])
			x_pos=next_x_pos
			next_x_pos+=len(annotations[topic])
			ax.hlines(y=df.shape[0] + y_annotation_offset+2, xmin=x_pos-.5, xmax=next_x_pos-.55, color=annotation_colors[idx], linewidth=5)
			ax.vlines(x=next_x_pos-.5, ymin=0, ymax=df.shape[0], color=annotation_colors[idx], linewidth=5, linestyle=':')
			ax.text(s=topic,x=(x_pos-.5+next_x_pos-.51)/2, y=df.shape[0]+y_annotation_offset, 
					ha='right', va='top', fontsize=22, rotation=45, color=annotation_colors[idx])
			ax.set_xlim(xlims)
	return fig

def plot_lowess(x, y, color='g', fill_alpha=.33, marker_alpha=.1, marker_size=10, grid_alpha=.5, ax=None, conf=.05, label=None, show_points=True, figsize=(3.5,5)):
	import numpy as np
	import matplotlib.pyplot as plt
	y = y[~np.isnan(x)]
	x = x[~np.isnan(x)]

	selection_vec = x < np.quantile(x,.95)
	y = y[selection_vec]
	x = x[selection_vec]

	x_sort = np.argsort(x)
	x=x[x_sort]
	y=y[x_sort]
	
	from skmisc.loess import loess
	l = loess(x,y)
	l.fit()
	pred = l.predict(x, stderror=True)
	conf = pred.confidence(alpha=conf)
	lowess = pred.values
	ll = conf.lower
	ul = conf.upper
	if ax == None: 
		_, ax = plt.subplots(1,1, figsize=figsize)
	if show_points: 
		ax.scatter(x, y, color=color, s=marker_size, alpha=marker_alpha)
	ax.plot(x, lowess,color=color, label=label)
	ax.fill_between(x,ll,ul,color=color, alpha=fill_alpha)
	plt.grid(alpha=grid_alpha)
	if type(label)==type(None):
		pass
	else:
		plt.legend()
	return ax

def show_seaborn_palettes():
	from IPython.display import display
	import seaborn as sns
	palettes = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r']
	for pal in palettes:
		print(pal)
		display(sns.palettes.color_palette(palette=pal, n_colors=10))

import re
standard_name = lambda x: re.sub('[@#\-.*/]', ' ', x).replace(' ', '_').lower()
display_name = lambda x: x.replace('_', ' ').title()

def removeDF_outliers(df, cols2remove_outliers = None, z=10):
	print("Removing outliers that are >{:.2f} std from the mean".format(z))
	outDF = df.copy()
	if type(cols2remove_outliers) == type(None):
		# Remove outliers from all columns
		cols2remove_outliers = list(df.columns)
	df_zscore = (df[cols2remove_outliers] - df[cols2remove_outliers].mean())/df[cols2remove_outliers].std()
	for feature in cols2remove_outliers:
		n2remove = len(outDF.loc[(np.abs(df_zscore[feature])>z), feature])
		if n2remove >0:
			print("Removing", str(n2remove), feature, "outlier(s)")
			print(outDF.loc[(np.abs(df_zscore[feature])>z), feature])
			outDF.loc[(np.abs(df_zscore[feature])>z), feature] = np.nan 
	return outDF

class CustomScaler(TransformerMixin): 
	def __init__(self, binary_columns=[]):		
		self.scaler = StandardScaler()
		self.binary_columns = binary_columns

	def fit(self, X,y=None):
		non_binary_columns = list(set(range(X.shape[1])).difference(self.binary_columns))
		setattr(self,'non_binary_columns',  non_binary_columns)
		if type(X)==pd.DataFrame:
			self.scaler.fit(X.iloc[:, non_binary_columns],y)
		else:
			self.scaler.fit(X[:, non_binary_columns],y)
		return self

	def transform(self, X):
		X_out=copy.deepcopy(X)
		if type(X)==pd.DataFrame:
			X_transformed = self.scaler.transform(X.iloc[:, self.non_binary_columns])
			X_out.iloc[:, self.non_binary_columns] = X_transformed
		else:
			X_transformed = self.scaler.transform(X[:, self.non_binary_columns])
			X_out[:, self.non_binary_columns] = X_transformed
		return X_out		


summary_params = {'Cminsd': {'nQuantiles': 4}, 
 'BHBA1C': {'bins':[0,5.7,6.4,12.2]},
 'BGLUC': {'bins': [60,100,120, 130, 350]},
 'BHDL': {'bins': [0,40,60, 210]},
 'Region_EUROPE': {'bins': [0,.5,1]},
 'Race_BLACK_OR_AFRICAN_AMERICAN' : {'bins': [0,.5,1]}
}

def generate_bootstrap_summaryDF(summary_params, currAnalysis, return_styled_df=False): 
	metric = "∆Prob"
	bootstrap_summary = pd.DataFrame([])
	for key in summary_params:
		print(f"Adding {key}")
		tmpDF= currAnalysis.bootstrap_summary_table(feature=key, metric=metric, **summary_params[key])
		if key == 'Cminsd': 
			tmpDF.loc[0, "Value"] = 0
		elif key=='Race_BLACK_OR_AFRICAN_AMERICAN':
			tmpDF.loc[0, "Value"] = 0
			tmpDF.loc[1, "Value"] = 1
		elif key=='Region_EUROPE':
			tmpDF.loc[0, "Value"] = 0
			tmpDF.loc[1, "Value"] = 1
		bootstrap_summary = pd.concat([bootstrap_summary, tmpDF])
	if not return_styled_df: 
		return bootstrap_summary.reset_index(drop=True)
	curr_summary_styled = bootstrap_summary[['Variable', 'Value', "mean ∆Prob", '∆Prob']].reset_index(drop=True)
	indecies = pd.MultiIndex.from_arrays([curr_summary_styled["Variable"].values,curr_summary_styled["Value"].values],
							  names=['Variable', 'Value'])
	curr_summary_styled.index = indecies
	curr_summary_styled = (curr_summary_styled[['mean ∆Prob', '∆Prob']]
						  .style.background_gradient(axis=None, cmap='RdBu_r', subset=["mean ∆Prob"], vmin=-.15,vmax=.15)
						  .set_table_styles([{'selector': 'th', 'props': [('font-size', '8pt'),('border-style','solid'),('border-width','1px')]}]))
	return curr_summary_styled

def shap_interaction_plot(analysis, feature, exposure_var, exposure_thr_vals=None, feature_thr=None, selectionVec=None, return_fig=False, figsize=(6,4), s=20, ylims=None):
    import matplotlib
    if type(exposure_thr_vals) == type(None):
        exposure_thr_vals=[float(analysis.df[exposure_var].min()), float(analysis.df[exposure_var].max())]
    if type(feature_thr) == type(None):
        feature_thr=analysis.df[feature].min()
    if type(selectionVec) == type(None):
        selectionVec=analysis.df[exposure_var].notna()
    fig,ax = plt.subplots(1,1, figsize=figsize)
    # ax = fig.add_subplot(111)
    tmpDF = pd.DataFrame({feature: analysis.df.drop(columns = analysis.target).loc[selectionVec, :].iloc[:,analysis.mdlFeatures.index(feature)].values, 
    'SHAP value': analysis.shap_interaction_values[selectionVec, analysis.mdlFeatures.index(exposure_var)][:,analysis.mdlFeatures.index(feature)],
    exposure_var: analysis.df[[exposure_var]].loc[selectionVec, :].values.flatten()})
    sns.scatterplot(x=feature, y='SHAP value', hue = exposure_var, data = tmpDF[(tmpDF[exposure_var]>=exposure_thr_vals[0]) & (tmpDF[exposure_var]<=exposure_thr_vals[1]) ], 
                    palette=sns.color_palette(palette='RdYlBu_r', as_cmap=True), s=s, ax = ax, 
                    hue_norm=matplotlib.colors.Normalize(vmin=analysis.df[exposure_var].min(), vmax=analysis.df[exposure_var].max()-2*analysis.df[exposure_var].std()),
                    edgecolor='gray')
    ax.set_title(f'Impact of {feature} on ER relationship\n (ER relationship: Effect of {exposure_var} on {analysis.target})')
    xlims= [tmpDF[feature].quantile(.01)-tmpDF[feature].std(), tmpDF[feature].quantile(.99)+tmpDF[feature].std()]
    ax.set_xlim(xlims)
    if type(ylims) == type(None):
        ylims = [tmpDF['SHAP value'].min()-tmpDF['SHAP value'].std(), tmpDF['SHAP value'].max()+tmpDF['SHAP value'].std()]
    ax.set_ylim(ylims)
    ax.plot([feature_thr, feature_thr], ylims, '--k',alpha=.5, lw=1.5)
    ax.plot(xlims, [0, 0], '--k',alpha=.5)
    ax.legend(loc=1, bbox_to_anchor=(1.15, .7), title=exposure_var)
    ax.grid(alpha=.5, ls='--')
    if return_fig:
        return fig


def get_combinedFeatName(shap_features):
	""" Helper function that generates the combinedFeat_name which is utilized in various functions"""
	tmp = sorted([feat for feat in shap_features if feat != 'expectedValue'])
	if 'expectedValue' in shap_features:
		tmp.append('expectedValue')
	shap_features=tmp
	combinedFeat_name= ", ".join(shap_features)
	combinedFeat_name = f'shap_sum({combinedFeat_name})'
	return combinedFeat_name

def shap_interaction_plot_combined(analysis, features, exposure_var, exposure_thr_vals=None, feature_thr=None, selectionVec=None, return_fig=False, s=20, fontsize=8, hue=None, figsize=(6,4), color_norm='linear'):
    """ This generates a combined interaction plot where we can examine the impact of multiple features on an ER relationship"""
    x_feature = features[0]
    import matplotlib
    if type(exposure_thr_vals) == type(None):
        exposure_thr_vals=[float(analysis.df[exposure_var].min()), float(analysis.df[exposure_var].max())]
    if type(feature_thr) == type(None):
        feature_thr=analysis.df[x_feature].min()
    if type(selectionVec) == type(None):
        selectionVec=analysis.df[x_feature].notna()
    fig,ax = plt.subplots(1,1, figsize=figsize)
    # ax = fig.add_subplot(111)
    # Caluculate ∑shap_interaction_values for features
    for f, feat in enumerate(features):
        curr_shap_arr = analysis.shap_interaction_values[selectionVec, analysis.mdlFeatures.index(exposure_var)][:,analysis.mdlFeatures.index(feat)].reshape(-1,1)
        if f == 0:
            shap_sum_arr = curr_shap_arr
        else:
            shap_sum_arr = np.concatenate((shap_sum_arr, curr_shap_arr), axis=1)
    shap_sum_arr = shap_sum_arr.sum(axis=1)
    
    if type(hue)==type(None):
        hue = exposure_var
    else:
        pass
    
    tmpDF = pd.DataFrame({x_feature: analysis.df.drop(columns = analysis.target).loc[selectionVec, :].iloc[:,analysis.mdlFeatures.index(x_feature)].values, 
    'shap sum': shap_sum_arr,
    exposure_var: analysis.df[[exposure_var]].loc[selectionVec, :].values.flatten(), 
                          hue: analysis.df[[hue]].loc[selectionVec, :].values.flatten()})
    
    color_norm_d=dict(linear=matplotlib.colors.Normalize,
    	log=matplotlib.colors.LogNorm)
    sns.scatterplot(x=x_feature, y='shap sum', hue = hue, data = tmpDF[(tmpDF[exposure_var]>=exposure_thr_vals[0]) & (tmpDF[exposure_var]<=exposure_thr_vals[1]) ], 
                    palette=sns.color_palette(palette='RdYlBu_r', as_cmap=True), s=s, ax = ax, 
                    hue_norm=color_norm_d[color_norm](vmin=analysis.df[hue].min(), vmax=analysis.df[hue].quantile(.99)),
                    edgecolor='gray')
    ax.set_title(f'Impact of {get_combinedFeatName(features)[9:-1]} on ER relationship\n(where ER relationship: Effect of {exposure_var} on {analysis.target})', fontsize=fontsize)
    xlims= [tmpDF[x_feature].quantile(.01)-tmpDF[x_feature].std(), tmpDF[x_feature].quantile(.99)+tmpDF[x_feature].std()]
    ax.set_xlim(xlims)
    ylims = [tmpDF['shap sum'].min()-tmpDF['shap sum'].std(), tmpDF['shap sum'].max()+tmpDF['shap sum'].std()]
    ax.set_ylim(ylims)
    ax.plot([feature_thr, feature_thr], ylims, '--k',alpha=.5)
    ax.plot(xlims, [0, 0], '--k',alpha=.5)
    ax.legend(loc=1, bbox_to_anchor=(1.3, .7), title=hue)
    ax.grid(alpha=.5, ls='--')
    ax.set_ylabel(get_combinedFeatName(features)+'\n(interaction values)\n(log-odds)', fontsize=fontsize)
    if return_fig:
        return fig
		
def remove_explainers_models(obj):
	""" 
	Removes explainers and models from a deep copy of an object and potential obj.feature_elimination (if exists) 
	It's looking in obj.SHAP_outputs, obj.bootstrap_SHAP_outputs, and obj.feature_selection.SHAP_outputs
	"""
	model_copy = copy.deepcopy(obj)
	if 'SHAP_outputs' in model_copy.__dict__.keys():
		if type(model_copy.SHAP_outputs) == dict:
			dict_keys = list(model_copy.SHAP_outputs.keys())
			for k in dict_keys:
				if k.startswith('explainer') | k.startswith('model'):
					print(f"Deleting: {k} from SHAP_outputs")
					del model_copy.SHAP_outputs[k]

	if 'feature_selection' in model_copy.__dict__.keys():
		FS_copy = model_copy.feature_selection # not a deep copy
		if 'SHAP_outputs' in FS_copy.__dict__.keys():
			if type(FS_copy.SHAP_outputs) == dict:
				dict_keys = list(FS_copy.SHAP_outputs.keys())
				for k in dict_keys:
					if k.startswith('explainer') | k.startswith('model'):
						print(f"Deleting: {k} from SHAP_outputs in nested feature_elimination object")
						del FS_copy.SHAP_outputs[k]

	if 'bootstrap_SHAP_outputs' in model_copy.__dict__.keys():       
		if type(model_copy.bootstrap_SHAP_outputs) == dict:
			dict_keys = list(model_copy.bootstrap_SHAP_outputs.keys())
			for k in dict_keys:
				if k.startswith('explainer') | k.startswith('model'):
					print(f"Deleting: {k} from bootstrap_SHAP_outputs")
					del model_copy.bootstrap_SHAP_outputs[k]
	return model_copy


from numpy import floor
from math import factorial
def kthperm(S, k):  
	P = []
	while S != []:
		f = factorial(len(S)-1)
		i = int(floor(k/f))
		x = S[i]
		k = k%f
		P.append(x)
		S = S[:i] + S[i+1:]
	return P

def uniform_permutation_sampling(features:"['A', 'B', 'C', 'D', 'E']", n_permutations=50, seed=None):
	perms=[]
	if type(seed) == type(None):
		for k in list(np.linspace(0,factorial(len(features))-1,n_permutations).astype(int)):
			perms.append(kthperm(S=features, k=k))
	else: 
		import random
		random.seed(seed)
		perm_numbers = random.sample(range(factorial(len(features))),n_permutations)
		for k in perm_numbers:
			perms.append(kthperm(S=features, k=k))
	return perms

def gen_annotated_variable_defsDF(annotations, varDefs, return_styled=True):
	"""
	Returns an annotated (MultiIndex) dataframe with categories of variables
	Syntax: 
	annotations = {'trail design': ['IPADOSE'],
 	'demographics': ['REGION', 'RACE',  'TOBHX'],
 	'biometrics': ['BAGE', 'BHT', 'BWT', 'BBMI'],
 	'physiology': ['CRCL', 'EGFR'], 
 	'disease history':['BECOG', 'PTENGSFL', 'HINCI', 'METSITES', 'PCHMSD', 'PCHISD', 'PCHMETD', 'PRGENRL', 'BPIBL', 'ONSURG', 'ONRADIO', 'FUSURG', 'FURADIO', 'PRPROST', 'PRORCH', 'PRRADIO', 'PRSTX', 'PRDOX', 'PRSTMHS', 'PRSTNMS', 'PRSTMCR', 'PRSTOTH', 'BPSG3FL', 'GLEASON'] , 
 	'biomarkers': ['BALB', 'BALP', 'BALT', 'BAMYLASE', 'BAST', 'BBICARB', 'BCA', 'BCHOL', 'BCREAT', 'BGLUC', 'BHBA1C','BHCT', 'BHDL', 'BHGB', 'BLDL', 'BLIPASE', 'BMG', 'BPLAT', 'BK', 'BPSA', 'BINR', 'BAPTT', 'BRBC', 'BSODIUM', 'BBILI', 'BPROT', 'BTRIG', 'BWBC', 'BBASO', 'BEOS', 'BLYM', 'BMONO', 'TPROT', 'PSA_Cycle2'], 
 	'target': ['OS']}
	varDefs = {'PSA_Cycle2': 'PSA @ beginning of cycle 2', 'OS': 'Overall survival(months)', 'TPROT': 'Total protein', 'PTENGSFL': 'PTEN-loss tumors by NGS Patient Flag', 'BAGE': 'Age', 'BBILI': 'Bilirubin', 'BBUN': 'Blood urea nitrogen', 'BCA': 'Calcium', 'BCHOL': 'Cholesterol', 'BCREAT': 'Creatinine', 'BEOS': 'Eosinophils', 'BHCT': 'Hematocrit', 'BHDL': 'HDL', 'BHGB': 'Hemoglobin', 'BINR': 'Prothrombin Intl.', 'BK': 'Potassium', 'BLDL': 'Low-density lipoprotein', 'BLIPASE': 'Triacylglycerol Lipase', 'BLYM': 'Lymphocytes', 'BMG': 'Magnesium', 'BMONO': 'Monocytes', 'BPLAT': 'Platelets', 'BPROT': 'Protein', 'BRBC': 'red blood cells', 'BSODIUM': 'Sodium', 'BTRIG': 'Triglycerides', 'BWBC': 'White blood cells', 'BPSA': 'Prostate specific antigen', 'BBMI': 'BMI', 'BECOG': 'ECOG status', 'BHT': 'Height', 'BWT': 'Weight', 'CRCL': 'Creatinine Clearance', 'EGFR': 'Estimated glomerular filtration rate', 'HINCI': 'NCI hepatic impairment criteria', 'IPADOSE': 'IPAT dose', 'PHASE': 'Trial phase', 'AGE': 'Age', 'RACE': 'Race', 'BGLUC': 'Baseline Glucose', 'BHBA1C': 'HBA1c', 'AUCsd': 'AUC single dose', 'Cmaxsd': 'Cmax single dose', 'Cminsd': 'C-trough (single dose)', 'BALB': 'Albumin', 'BALP': 'Alkaline Phosphatase', 'BALT': 'Alanine Aminotransferase', 'BAMYLASE': 'Amylase', 'BAPTT': 'Activated Partial Thromboplastin', 'BAST': 'Aspartate Aminotransferase', 'BBASO': 'Basophils', 'BBICARB': 'Bicarbonate', 'REGION': 'Region', 'USUBJID': 'Unique Subject Identifier', 'METSITES': 'Number of Metastases Sites', 'PCHGSD': 'Gleason Score at Diagnostic', 'PCHMSD': 'Months Since Diagnosis', 'PCHISD': 'Initial Stage at Diagnostic', 'PCHMETD': 'Time to Metastasis Since Diagnostic', 'TOBHX': 'Tobacco History', 'ONSURG': 'On-study Surgical Procedure (Yes/No)', 'ONRADIO': 'On-study Radiotherapy (Yes/No)', 'FUSURG': 'Follow-up Surgical Procedure (Yes/No)', 'FURADIO': 'Follow-up Radiotherapy (Yes/No)', 'PRGENRL': 'Progression at Enrollment', 'PRPROST': 'Prior Prostatectomy (Yes/No)', 'PRORCH': 'Prior Orchiectomy (Yes/No)', 'PRRADIO': 'Prior Prostate Radiotherapy (Yes/No)', 'PRSTX': 'Prior Systemic Therapy (Yes/No)', 'PRDOX': 'Prior Docetaxel (Yes/No)', 'PRSTMHS': 'Prior Sys Therapy Setting MHS (Yes/No)', 'PRSTNMS': 'Prior Sys Therapy Setting NMS (Yes/No)', 'PRSTMCR': 'Prior Sys Therapy Setting MCR (Yes/No)', 'PRSTOTH': 'Prior Sys Therapy Setting Other (Yes/No)', 'BPSG3FL': 'Baseline Pain Severity Score > 3 Flag', 'BPIBL': 'Pain Severity Assessment at Baseline', 'GLEASON': 'Gleason Score at Screening'}
	"""
	data = []
	for cat in annotations:
		currVars=sorted(copy.deepcopy(annotations[cat]))
		for el in currVars:
			if el in varDefs:
				data.append((cat, el, varDefs[el]))
			else: 
				print(el, "not in dictionary")
	varDefDF= pd.DataFrame(data, columns=['Category', 'Variable', 'Definition'])
	indecies = pd.MultiIndex.from_arrays([varDefDF["Category"].values, varDefDF["Variable"].values],
											 names=['Category', 'Variable'])
	varDefDF.index = indecies
	if return_styled:
		return varDefDF[['Definition']].style.set_table_styles([{'selector': 'th', 'props': [('font-size', '8pt'),('border-style','solid'),('border-width','1px')]}])
	else:
		return varDefDF[['Definition']]

def load_analysis(outputs_dir, target, analysis_name='parsMDL'):
    import glob, os, pickle
    fileName = max(glob.glob(outputs_dir + target+ f"_{analysis_name}*.p"), key=os.path.getctime)
    print("Loaded:", fileName)
    analysis= pickle.load(open(fileName, 'rb'))
    return analysis

def run_pca(df, n_clusters=0):
    from shapml.utils.misc import CustomScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    df=df.copy()
    features=list(df.columns)
    df2fit = CustomScaler().fit_transform(df[features]) # StandardScaling and ignoring binary columns
    df2fit = df2fit.fillna(df2fit.median())
    pca = PCA(n_components=3).fit(df2fit)
    df['PC1'] = pca.transform(df2fit)[:,0]
    df['PC2'] = pca.transform(df2fit)[:,1]
    df['PC3'] = pca.transform(df2fit)[:,2]
    
    pc1s = []
    pc2s = []
    pc3s = []
    for feature in features: 
        coeff = 1
        pc1 = np.dot((np.array(features)==feature).astype(int), pca.components_[0])*coeff
        pc2 = np.dot((np.array(features)==feature).astype(int), pca.components_[1])*coeff
        pc3 = np.dot((np.array(features)==feature).astype(int), pca.components_[1])*coeff
        pc1s.append(pc1)
        pc2s.append(pc2)
        pc3s.append(pc3)
    biplot_df = pd.DataFrame(zip(features, pc1s, pc2s, pc3s), columns=['feature', 'PC1', 'PC2', 'PC3'])
    
    if n_clusters > 1:
        clusters = KMeans(n_clusters=n_clusters, random_state=1).fit_predict(df[['PC1', 'PC2', 'PC3']])
        df['Cluster'] = list(map(str, clusters))
    return df, biplot_df

def add_annotations(fig, x_offset=-40, y_offset=25, fontsize=30, weight='bold'):
    import string
    axes = fig.get_axes()
    texts = [string.ascii_uppercase[s] for s in range(len(axes))]
    for a,l in zip(axes, texts):
        a.annotate(l, xy=(x_offset, y_offset + a.bbox.height), xycoords="axes pixels", fontsize=fontsize, weight = weight)
    return