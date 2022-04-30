# from collections import Counter
# from itertools import permutations
import numpy as np 
import pandas as pd 
from string import ascii_uppercase
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-poster')
from collections import defaultdict
import math
f_inv_logit = lambda x: np.exp(x)/(1+np.exp(x))
import time
import swifter
import copy
from ..utils.misc import uniform_permutation_sampling
from collections import Counter, defaultdict
from ..utils.helpers import nargout


def sequential_probability_contributions(impact_values, feature_names, exp_value:"in log-odds scale", colors_d={}, plot=True):
	colors_d = defaultdict(lambda : 'gray', colors_d)
	x = np.linspace(-6+exp_value, 6+exp_value, 10000)
	y=f_inv_logit(x)

	prob_values = [f_inv_logit(i+exp_value) for i in np.cumsum(impact_values)]
	tmp=[f_inv_logit(exp_value)];tmp.extend(prob_values) 
	delta_p = np.diff(tmp)
	probability_contributions = dict(zip(feature_names,np.round(delta_p,4)))
	if plot:
		fs=12
		plt.scatter(x,y, marker='.', color='k', s=1)
		plt.plot([exp_value, exp_value], [0, f_inv_logit(exp_value)], '--k')
		plt.annotate("Expected\n probability:\n{:.2f}".format(f_inv_logit(exp_value)), (exp_value,f_inv_logit(exp_value)), ha='right')
		plt.ylabel("Probability")
		plt.xlabel("Log-odds")
		r = 3 
		i = 0 
		xpos = copy.deepcopy(impact_values)
		xpos[0] = xpos[0]+exp_value
		for x,v,dp in zip(np.cumsum(xpos), np.round(prob_values,r), np.round(delta_p, r)):  
			s = f"{feature_names[i]}\nOrder: {i+1}\np={v}\nΔp={dp}\nlog-odds: {impact_values[i]}"
			plt.annotate(s, (x, v),xytext=(x-2 if (i%2==0) else x+2, v+.15), ha='center', va='center', 
						 size=fs, fontweight='bold', color=colors_d[feature_names[i]],
						arrowprops=dict(arrowstyle='-|>', color=colors_d[feature_names[i]]))
			plt.scatter(x,v, color=colors_d[feature_names[i]])
			i+=1
	order_of_consideration = {f:c+1 for c,f in enumerate(feature_names)}
	return probability_contributions, order_of_consideration
	
import cachetools
@cachetools.cached(cache={})
def generate_random_permutations(features, n_perms, seed=99):
	i = 0
	np.random.seed(seed)
	while i<n_perms:
		j = 0
		if i == 0:
			curr_perm = tuple(np.random.permutation(features))
			arr= [curr_perm]
		else: 
			curr_perm = tuple(np.random.permutation(features))
			if (curr_perm not in arr):
				arr.append(curr_perm)
			else:
				j=0
				while (curr_perm in arr):
					curr_perm = tuple(np.random.permutation(features))
					if curr_perm not in arr: 
						arr.append(curr_perm)
						break
					j+=1
					if j > 1000: 
						print(f"Couldn't find a novel permuations after {j} iterations")
						print(f"Terminating early with {len(arr)} permuations")
						return arr
		i+=1
	return arr


def find_opt_seed(n_features, n_perms, seed_max=10000, show=False, non_uniformity_penalty=8):
	"""
	non_uniformity_penalty: this should be a multiple of 2 (Chi-squared is 2) 
	larger penalty penalizes outlier occurences to a great extent.
	"""
	def calc_chi_powers(n_features, n_perms, seed):
		features = ["x"+str(col) for col in range(1, n_features+1)]
		perms = generate_random_permutations(features=tuple(features), n_perms=n_perms, seed=seed); perms=np.array([list(arr) for arr in perms])
		expected = n_perms/n_features
		chi_power= np.array([])
		for feature in features:
			possible_positions=range(1,n_features+1)
			positions = np.where(perms==feature)[1]+1
			position_counts = defaultdict(lambda : 0, Counter(positions))
			height = np.array([position_counts[p] for p in possible_positions])
			chi_power = np.concatenate([chi_power, (height-expected)**non_uniformity_penalty/expected])
		return np.mean(chi_power)

	seeds = range(seed_max)
	chiPowers = [calc_chi_powers(n_features=n_features, n_perms=n_perms, seed=s) for s in seeds]
	seed_idx = np.where(chiPowers==min(chiPowers))[0][0]
	
	if show:
		plt.plot(seeds, chiPowers, '-o')
		plt.vlines(x=seed_idx, ymin=0, ymax=2, ls='--')
		plt.ylabel(f'Chi^{non_uniformity_penalty}')
		plt.xlabel('Seed')
		plt.title(f'Optimal seed {seed_idx}, Chi^{non_uniformity_penalty}: {np.round(chiPowers[seed_idx],3)}')
	n_out = nargout()
	if n_out == 2:
		return seed_idx, chiPowers[seed_idx]
	else:
		return seed_idx	

def calc_mean_probability_impacts(feature_impacts, exp_value, n_perms, features = None, plot=False, save=False, approach=None, seed=0):
	""" 
	approach : 'approx' - will return an approximation based on a sampling of the permuation space with specified seed
			   'exact' - uses all possible permutations
				How to convert logit to shap (default: None- 'approx' if nFeatures<=4 else 'exact')
	"""
	if type(approach) == type(None):
		if (len(feature_impacts) <= 4):
			approach = 'exact'
		else: 
			approach = 'approx'
			seed=seed
	elif approach == 'exact':
		if (len(feature_impacts) > 6):
			raise ValueError("This will probably take a while. Try using 'approx' approach instead.")
	if (len(feature_impacts) > 4) & plot:
		raise ValueError("This may take too long, and maybe too complex to plot. (Set plot=False)")
	
	if features == None: 
		features = [feat for feat in ascii_uppercase[:len(feature_impacts)]] 
		assert len(features) == len(feature_impacts)
	
	feature_d = dict(zip(features, feature_impacts))
	colors = sns.palettes.color_palette('tab10')[:len(feature_impacts)]
	all_perms_df = pd.DataFrame([])
	colors_d = dict(zip(features, colors))
	
	if plot:
		n_permutations = math.factorial(len(features))
		fig = plt.figure(figsize=(6*n_permutations,6))

	if approach == 'approx': 
		iterator = generate_random_permutations(tuple(features), n_perms=n_perms, seed=seed)
	elif approach == 'approx_uniform':
		iterator = uniform_permutation_sampling(features=features, n_permutations=n_perms)
		# iterator=generate_random_permutations(tuple(features), n_perms=n_perms)
	elif approach == 'approx_random':
		iterator=generate_random_permutations(tuple(features), n_perms=n_perms)
	elif approach == 'exact':
		iterator = uniform_permutation_sampling(features=features, n_permutations=math.factorial(len(features)))
		# iterator= generate_random_permutations(tuple(features), n_perms=math.factorial(len(features)))

	for ii, curr_perm in enumerate(iterator):
		impact_values = [feature_d[p] for p in curr_perm]
		feature_names = [f for f in curr_perm]
		if plot:
			fig.add_subplot(n_permutations//6, 6, ii+1)
	#     print(feature_names)
		prob_impact, order = sequential_probability_contributions(impact_values, feature_names, exp_value, colors_d, plot=plot)
		currDF = pd.DataFrame({'logit impact': feature_d,
		'probability impact' : prob_impact,
		'order of consideration': order,
						  'permuation': ii+1}) 
		all_perms_df = pd.concat([all_perms_df, currDF])
	
	if plot: 
		plt.tight_layout()
		if save:
			plt.savefig('permutaions.png',  bbox_inches='tight')
		plt.show()
		from IPython.display import display
		display(all_perms_df)
	mean_probability_impact = all_perms_df.reset_index().groupby('index').mean()[['logit impact','probability impact']]
	# mean_probability_impact['expected_value', 'logit impact'] = exp_value
	return mean_probability_impact

def convert_to_prob(x, mdlFeatures, approach='approx', n_perms=50, seed=0):
	""" function to apply across rows 
	n_perms ignored if approach == 'approx'
	"""
	out = calc_mean_probability_impacts(feature_impacts=x[mdlFeatures], exp_value = x['expectedValue'], plot=False, features = mdlFeatures, approach=approach, n_perms=n_perms,seed=seed)['probability impact']
	out['expectedValue'] = f_inv_logit(x['expectedValue'])
	return out

def convert_shapDF_logit_2_shapDF_prob(shapDF_logit, mdlFeatures=None, approach='approx', seed_max=10000, seed=None, n_perms=50, expectedValueCol='expectedValue'):
	""" 
	Converts logit SHAP values to probablity estimates. maximum number of features is 8 (otherwise it's too computationally expensive)
	shapDF_logit : a dataframe containing shap values in logit scale with at least one column entitled "expectedValue" 
	approach: Options: 'approx', 'approx_uniform', 'exact'
			  'approx' - will return an approximation based on a sampling of the permuation space that has a roughly uniform distribution (best out of a number of iterations) 
	"""
	assert expectedValueCol in shapDF_logit.columns
	if type(mdlFeatures) == type(None): 
		# Best practice would be to suppy the model features, if not supplied, it will use all the columns not entitled "expected_value"
		mdlFeatures = list(shapDF_logit.columns)
		mdlFeatures.pop(mdlFeatures.index(expectedValueCol))

	if (approach =='approx') & (type(seed) == type(None)):
		print("Determining seed that yields an approximate uniform sampling of the permuation space")
		seed=find_opt_seed(n_features=len(mdlFeatures), n_perms=n_perms, seed_max=seed_max)
		print(f"Using seed: {seed}")


	print("Converting logit values to probability scale")
	start = time.time()
#     print(mdlFeatures)
#     out = shapDF_logit.apply(convert_to_prob, mdlFeatures=mdlFeatures, approach=approach, n_perms=n_perms, axis=1)
	
	out = shapDF_logit.swifter.apply(convert_to_prob, mdlFeatures=mdlFeatures, approach=approach, n_perms=n_perms, seed=seed, axis=1)
	end = time.time()
	print("Execution time: {:.2f}s".format(np.round(end - start,5)))
	return out