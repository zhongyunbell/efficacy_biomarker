import pandas as pd
import numpy as np
import streamlit as st
import copy
import matplotlib.pyplot as plt
import seaborn as sns

f_inv_logit = lambda x: np.exp(x)/(1+np.exp(x))
def genrate_potential_mdl_terms(df, target):
    potential_mdl_vars=[col for col in df.columns if col != target]
    potential_interaction_terms =[]
    for feat1 in potential_mdl_vars:
        for feat2 in potential_mdl_vars:
            if feat1 != feat2:
                potential_interaction_terms.append(f'{feat1}:{feat2}')            
    potential_mdl_terms = copy.deepcopy(potential_mdl_vars)
    potential_mdl_terms.extend(potential_interaction_terms)
    return sorted(potential_mdl_terms)

@st.cache(hash_funcs={pd.DataFrame: lambda _: None}, suppress_st_warning=True)
def generate_df_for_LR_synthetic(df, target, oversampler_name, outputs_dir, random_state, desired_rows, mdl_terms=None, formula_like=None):
    from patsy import dmatrices
    if type(mdl_terms) != type(None):
        formula_like = target + " ~ " + " + ".join(mdl_terms)
        print(formula_like)
    elif type(formula_like) != type(None):
        pass
    else:
        ValueError("Must supply mdl_terms or formula_like")
    y,X = dmatrices(formula_like=formula_like, data=df.fillna(df.median()), return_type='dataframe')
    return pd.concat([X,df[target]],axis=1)

@st.cache(hash_funcs={pd.DataFrame: lambda _: None}, suppress_st_warning=True)
def generate_df_for_LR_real(df, target, outputs_dir, desired_rows, mdl_terms=None, formula_like=None):
    from patsy import dmatrices
    if type(mdl_terms) != type(None):
        formula_like = target + " ~ " + " + ".join(mdl_terms)
        print(formula_like)
    elif type(formula_like) != type(None):
        pass
    else:
        ValueError("Must supply mdl_terms or formula_like")
    y,X = dmatrices(formula_like=formula_like, data=df.fillna(df.median()), return_type='dataframe')
    return pd.concat([X,df[target]],axis=1)

def synthesize_outcomes(df_for_LR, coeffs_d, random_state=0):
    """
    df: design matrix that contains all the terms in the model defined by coeff_d
    returns: (df_for_LR, logit_df)
            df_for_LR: now contains synthetic_probability and synthetic targets
            logit_df: contains columns columns only for primary model terms (interaction effects are evenly distributed to primary terms)
    """
    np.random.seed(random_state)
    df = df_for_LR.copy()
    logit_df = pd.DataFrame()
    for col in coeffs_d:
        logit_df[col] = coeffs_d[col]*df[col]    
    df['synthetic_probability'] = f_inv_logit(logit_df.sum(axis=1))
    df['synthetic_target'] = np.random.binomial(n=1, p=df['synthetic_probability'])
    # Mutate logit_df so that interaction effects are distributed evenly to primary terms:
    # Note: This removes interaction terms
    interaction_cols = [term for term in logit_df.columns if term.__contains__(':')]
    if len(interaction_cols)>0:
        from collections import Counter, defaultdict
        import itertools
        split_var_list = [col.split(':') for col in logit_df.columns]
        var_counts_d = Counter(list(itertools.chain(*list(split_var_list)))) # This defines how to split contributions

        for full_mdl_term in logit_df.columns:
            split_mdl_terms = list(full_mdl_term.split(':'))
            if len(split_mdl_terms)==1:
                pass
            else:
                for split_mdl_term in split_mdl_terms:
                    logit_df[split_mdl_term] += logit_df[full_mdl_term]/(var_counts_d[split_mdl_term])
        logit_df.drop(columns = interaction_cols, inplace=True)        
    return df, logit_df


def binomVec_yerr(vec):
    from statsmodels.stats.proportion import proportion_confint
    lwr, upr = np.abs(proportion_confint(vec.sum(), len(vec), method='wilson')-np.mean(vec))
    point_estimate = np.mean(vec)
    return point_estimate, lwr, upr

def binom_err(series):
    point_estimate, lwr, upr = binomVec_yerr(series)
    return [lwr,upr]

def prob_plot(x, y, q, df, hl=None, ax=None, figsize=(6,4), ylims=None, return_fig=False, color='gray', legend_label=None, ms=2):
    """ plot binary probability vs. continuous variable discretized into q quantiles"""
    if ax == None: 
        fig,ax = plt.subplots(1,1, figsize=figsize)
    tmpDF = df[[x, y]]
    tmpDF[x+'_binned'] = pd.qcut(tmpDF[x],q, duplicates='drop')
    errDF = tmpDF.groupby(x+'_binned').agg({y:['mean', binom_err, 'count']}).reset_index()
    errDF.columns = [x+'_binned', 'mean', 'binom_err', 'count']

    for idx in range(len(tmpDF[x+'_binned'].unique().categories)):
        currDF = errDF.iloc[idx,:]
        x_val = currDF[x+'_binned'].mid #had this at left
        y_val= currDF['mean']
        yerr = np.array(currDF['binom_err']).reshape(-1,1)
        ax.errorbar(x=x_val, y=y_val, yerr=yerr, marker='o', ms=ms, mew=4, color=color, label=legend_label)
        ax.set_xlabel(x+' (binned)')
        ax.set_ylabel(y+ ' rate')
        ax.set_title(y+ ' rate\nvs. '+ x)
    if type(hl) != type(None):
        ax.hlines(hl, ax.get_xlim()[0], ax.get_xlim()[1], 'k', '--' )
    if type(ylims) != type(None):
        ax.set_ylim(ylims)
    if return_fig:
        return fig
    return ax

def stratified_prob_plot(x, y, q, df, stratify=None, figsize=(6,4), stratify_thr=None, **kwargs):
    fig,ax = plt.subplots(1,1, figsize=figsize)
    if type(stratify) != type(None):
        m = 0
        legend_labels = []
        if stratify in df.columns: 
            print(f"Stratified ROC by {stratify}")
            nUnique = len(df[stratify][df[stratify].notna()].unique())
            if nUnique <= 2: 
                unique_values = df[stratify].unique()
                colors = ['b', 'r']
                for u in set(unique_values):
                    if np.isnan(u):
                        continue
                    selection_vec = (df[stratify]==u)
                    analysis_str = f"{stratify}=={u}"
                    ax = prob_plot(x, y, q, df[selection_vec], color=colors[m], ax=ax, legend_label=analysis_str, **kwargs)
                    # legend_labels.append(analysis_str)
                    m+=1
            else:
                colors = ['b', 'r']
                if type(stratify_thr)==type(None):
                    med_val = np.round(df[stratify].median(),2)
                else:
                    med_val = stratify_thr
                for ii in [0,1]:
                    if ii ==0: 
                        selection_vec = (df[stratify]<=med_val)
                        comparator = "<="
                        analysis_str = f"{stratify}{comparator}{med_val}"
                    elif ii==1:
                        selection_vec = (df[stratify]>med_val)
                        comparator = ">"
                        analysis_str = f"{stratify}{comparator}{med_val}"
                    ax = prob_plot(x, y, q, df[selection_vec], color=colors[m], ax=ax, legend_label=analysis_str, **kwargs)
                    # legend_labels.append(analysis_str)
                    m+=1
        else:
            NameError("{stratify} is not in dataframe")
        
        # h, l = fig.ax.get_legend_handles_labels()
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        
        lgnd = plt.legend(by_label.values(), by_label.keys(), fontsize=10, markerscale=.2)
        for handle in lgnd.legendHandles:
            handle.set_lw([1])
        return fig
    else:
        return prob_plot(x, y, q, df, return_fig=True, **kwargs)



### cutoff analysis


import seaborn as sns
from statsmodels.stats.proportion import proportion_confint, proportions_ztest 
import statsmodels.api as sm
import statsmodels.formula.api as smf
from shapml.bootstrap_shap.bootstrap_shap_analysis import getCombined_BootsCI_DF, get_combinedFeatName
from collections import defaultdict

my_round_lwr = lambda x: np.floor(x*20)/20 # round down to nearest .05
my_round_upr = lambda x: np.ceil(x*20)/20 # round up to nearest .05



alpha = .8 
ci = .95 
figsize = (18,6)

# Prepare dataset for cutoff-analysis: 
# df_st = FE.df.copy()
    
# if prediction_time == 'Induction': 
#     shap_features = [exposure_var]
#     valid_idxs = list(df_st[df_st['Adalimumab']==0].index)
# elif prediction_time == 'Maintenance': 
#     shap_features = [exposure_var, 'MAINT_TRT']
#     valid_idxs = df_st.query("INDUCT_TRT ==1").index

def cutoff_plot_st(analysis, shap_features, df_st, prediction_time, exposure_var, valid_idxs, outputs_dir, Treatment_Flag='Treatment_Flag', yaxis="Estimated Tx effect\n(∆probability)", thr=200, x_feature='BFECAL', plot_type = 'Estimated Tx effect', savefig=False, show_interaction=True, figsize=(18,6), colorBy = 'Treatment group', bootstrap_analysis=True):
    target=analysis.target
    
    shapDF_prob = analysis.shapDF_prob.copy()
    meanExpProb= analysis.shapDF_prob['meanExpValue'].mean()

    st.write('Median: ', np.round(df_st[x_feature].median(), 2))
    # st.write(df_st.columns)
    fig = plt.figure(figsize=figsize)
    plt.style.use(['seaborn-white','seaborn-colorblind','seaborn-talk'])
    if bootstrap_analysis: 
        bootsDF_cutoff=analysis.bootsDF.copy()
        ax = fig.add_subplot(1,2,1)
        if ('MAINT_TRT' not in bootsDF_cutoff.columns) & ('MAINT_TRT' in shap_features):
            shap_features.pop(shap_features.index('MAINT_TRT'))
        if 'CTR1_MAINT_TRT' in analysis.df.columns:
            shap_features.append('CTR1_MAINT_TRT')
        feature=x_feature
        combinedFeat_name=get_combinedFeatName(shap_features)
        feature_bootsDF = (getCombined_BootsCI_DF(bootsDF=bootsDF_cutoff, df=df_st, 
                            x_feature=x_feature, shap_features=shap_features)
                            .rename(columns = {combinedFeat_name+"_lwr": 'CI_lwr', 
                                                combinedFeat_name+"_median": 'median_SHAP',
                                                combinedFeat_name+"_mean": 'mean_SHAP',
                                                combinedFeat_name+"_upr": 'CI_upr'})).iloc[valid_idxs,:]
        if prediction_time == 'Induction': 
            feature_bootsDF['Treatment group'] = df_st['TRT'].replace({0:'Placebo', 1:'TRT'})
        elif prediction_time == 'Maintenance': 
            feature_bootsDF['INDUCT_TRT'] = df_st.INDUCT_TRT.replace({0:'Placebo', 1:'TRT'})
            feature_bootsDF['MAINT_TRT'] = df_st.MAINT_TRT.replace({0:'Placebo', 1:'TRT'})
            feature_bootsDF['Treatment group'] = feature_bootsDF.apply(lambda x: "".join([x['INDUCT_TRT'], ', ', x['MAINT_TRT']]), axis=1)
        
        nLevels = len(feature_bootsDF['Treatment group'].unique())
        colors0 = sns.color_palette(palette='hsv', n_colors=nLevels).as_hex()
        # colors0 = ('#2ca02c','#f7f7f7')
        if yaxis.startswith('Adjusted'): 
            feature_bootsDF[['CI_lwr', 'mean_SHAP', 'CI_upr']] = feature_bootsDF[['CI_lwr', 'mean_SHAP', 'CI_upr']] + bootsDF_cutoff['expectedValue'].mean()
        else:
            feature_bootsDF[['CI_lwr', 'mean_SHAP', 'CI_upr']] = feature_bootsDF[['CI_lwr', 'mean_SHAP', 'CI_upr']]

        lwr = feature_bootsDF.apply(lambda x: x.mean_SHAP - x.CI_lwr, axis=1)
        upr = feature_bootsDF.apply(lambda x: x.CI_upr - x.mean_SHAP, axis=1)
        
        if colorBy == 'Treatment group':
            levels_d0 = dict(enumerate(feature_bootsDF["Treatment group"].unique()))
            levels_d0 = defaultdict(lambda : 12, {v:k for k,v in levels_d0.items()})
            feature_bootsDF['color0']=  feature_bootsDF["Treatment group"].apply(lambda x: colors0[levels_d0[x]] if levels_d0[x]<= 10 else '#808080').astype(str)
            
        else:
            feature_bootsDF['color0'] = 'gray'

            
        classes = list(feature_bootsDF["Treatment group"].unique()) #['TRT, TRT'];
        for cl in classes:
            currDF = feature_bootsDF[feature_bootsDF["Treatment group"] == cl]
            i = currDF.index
            ax.errorbar(currDF[feature].loc[i], 
                            feature_bootsDF.mean_SHAP.loc[i], 
                            np.array([lwr.loc[i],upr.loc[i]]), 
                            fmt='o', 
                            color=feature_bootsDF['color0'].loc[i[0]],
                            ecolor='lightgray', elinewidth=2, capsize=2, 
                        alpha=.8, markeredgecolor='k', markeredgewidth=.5,markersize=8,
                            label = feature_bootsDF["Treatment group"].loc[i[0]])

        if feature.startswith('CTR'): 
            units = ' (μg/mL)'
        else:
            units = ''
        ax.set_xlabel(feature + " value" + units)
        ax.set_ylabel(yaxis)
        ax.set_title("Treatment effect" + " vs {}\nw/ bootstrapped {:.0f}% CI".format(feature, ci*100))
        ax.legend()

        if yaxis.startswith('Adjusted'): 
            ylim_lwr = my_round_lwr(shapDF_prob.min()[0:-1].min()-0+meanExpProb)
            ylim_upr = my_round_upr(shapDF_prob.max()[0:-1].max()+.05+meanExpProb)
        else: 
            ylim_lwr = my_round_lwr(shapDF_prob.min()[0:-1].min()-0)
            ylim_upr = my_round_upr(shapDF_prob.max()[0:-1].max()+.05)

        ax.set_yticks(np.arange(ylim_lwr, ylim_upr, .05))
        ax.grid(True, alpha=.8)
        ax.set_facecolor((.96, .96, .96))
        ax.set_ylim(ylim_lwr, ylim_upr)
        ax.vlines(x=thr, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], ls='--', color='r')

        if x_feature == 'FECAL_week10':
            ax.set_xscale('log')
            from matplotlib.ticker import ScalarFormatter
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.set_xlim(left=0)
        elif x_feature == 'CREATN_week10':
            ax.set_xlim(left=20)
            ax.set_xlim(right=feature_bootsDF[feature].std()+feature_bootsDF[feature].quantile(.97))
        else:
            ax.set_xlim(left=0)
            ax.set_xlim(right=feature_bootsDF[feature].std()+feature_bootsDF[feature].quantile(.97))

        if yaxis.startswith('Adjusted'):
            ax.plot(ax.get_xlim(), [meanExpProb, meanExpProb], '--k')
        else:
            ax.plot(ax.get_xlim(), [0, 0], '--k')

        ### 

    
    cols = [target] 
    cols.append(x_feature)
    df = df_st[cols]
    pLims = np.round(df[target].mean(), 2)
    pLims = [pLims-.3, pLims + .3]
    y_offset = .05
    allFeats=[]
    allNs = []
    allPs = []
    allYerrs=[]
    allYerrs_delta = []
    n_placebo = []
    n_TRT = []
    nFeats =1

    f=0; feat=x_feature;

    df[feat+"_bin"] = df[feat] < thr         
    ps = [np.round(df[target][~df[feat+"_bin"]].mean(),3),  np.round(df[target][df[feat+"_bin"]].mean(),3)]
    ns= [(~df[feat+"_bin"]).sum(),  df[feat+"_bin"].sum()]
    labels = [feat +"\n<{:.1f}\nn={}".format(thr,ns[1]), feat +"\n>={:.1f}\nn={}".format(thr,ns[0])]
    
    yerrs_abs= [np.round(proportion_confint(int(ps[0]*ns[0]), ns[0], alpha=0.05, method='normal'),2), 
                np.round(proportion_confint(int(ps[1]*ns[1]), ns[1], alpha=0.05, method='normal'),2)]
    yerrs= [np.round(np.abs(yerrs_abs[0]-ps[0]),2),np.round(np.abs(yerrs_abs[1]-ps[1]),2)]
    allYerrs_delta.extend([list(yerrs[0]),list(yerrs[1])])


    ax3 = fig.add_subplot(nFeats,4,f*4+3)
    grp_names =[feat+" ≤ {:.2f}".format(thr), feat+" > {:.2f}".format(thr)]
    exposure_groups=False
    if plot_type == 'Empirical probability':
        tmpDF = df_st.iloc[valid_idxs,:]
        tmpDF = tmpDF[tmpDF[feat].notna()]
        
        # tmpDF['TRT'] = tmpDF['Treatment group'].replace({"Placebo":0, "TRT":1})
        tmpDF[feat+"_binned"] = tmpDF[feat].apply(lambda x: feat+" ≤ {:.2f}".format(thr) if x <= thr else feat+" > {:.2f}".format(thr))
        if prediction_time == "Induction":
            trt_var = 'TRT'
        elif prediction_time == 'Maintenance':
            trt_var = 'MAINT_TRT'
            tmpDF['TRT'] = tmpDF['MAINT_TRT']
        elif prediction_time == 'Semisynthetic':
            tmpDF['TRT'] = tmpDF[Treatment_Flag]

        errDF = tmpDF.groupby([feat+"_binned", 'TRT']).agg({target:['mean', binom_err, 'count']})[target].reset_index()

        if feat != 'CTR2':
            ax3.plot([0,1], errDF[errDF[feat+"_binned"]==grp_names[0]].sort_values('TRT', ascending=True).reset_index(drop=True)['mean'].values, color=sns.palettes.color_palette('muted')[0], label=grp_names[0])
            ax3.plot([0,1], errDF[errDF[feat+"_binned"]==grp_names[1]].sort_values('TRT', ascending=True).reset_index(drop=True)['mean'].values, color=sns.palettes.color_palette('muted')[1], label=grp_names[1])

        for idx in range(4):
            currDF = errDF.iloc[idx,:]
            trt= currDF['TRT']
            trt = 'Placebo' if trt == 0 else 'TRT'
            if trt =='Placebo':
                n_placebo.append(currDF['count'])  
            else: 
                n_TRT.append(currDF['count'])
            grp = 0 if (currDF[feat+"_binned"]==feat+" ≤ {:.2f}".format(thr)) else 1
            label = currDF[feat+"_binned"] + f"; n={currDF['count']} " + str(trt) 
            color = sns.palettes.color_palette('muted')[0] if (grp == 0) else sns.palettes.color_palette('deep')[1]
            x = currDF['TRT']+(.02 if grp == 0 else 0)
            y= currDF['mean']
            yerr = np.array(currDF['binom_err']).reshape(-1,1)

            if feat !='CTR2':
                ax3.errorbar(x=x, y=y, yerr=yerr, marker='o', ms=15, markeredgecolor=color, mew=4, color=color if trt=='TRT' else 'w', ecolor=color)
                if trt == 'TRT':
                    errDF['n_responders'] = errDF['mean']*errDF['count']
                    tx_effect = errDF[errDF[feat+'_binned'] == grp_names[grp]].iloc[1]['mean']- errDF[errDF[feat+'_binned'] == grp_names[grp]].iloc[0]['mean']

                    count = [errDF[errDF[feat+'_binned'] == grp_names[grp]].iloc[0]['n_responders'], errDF[errDF[feat+'_binned'] == grp_names[grp]].iloc[1]['n_responders']]
                    nobs = [errDF[errDF[feat+'_binned'] == grp_names[grp]].iloc[0]['count'], errDF[errDF[feat+'_binned'] == grp_names[grp]].iloc[1]['count']]
                    _, pval = proportions_ztest(count, nobs)
                    ann_str = "∆{:.2f}, p={:.4f}".format(tx_effect, pval)
                    ax3.annotate(xy=(x,y+.02), text=ann_str, color=color, ha='right',va='bottom', size=14)
        
        if feat !='CTR2':
            # Determine interaction significance

            logitDF = tmpDF[[feat, feat+"_binned", 'TRT', target]].dropna()
            logitDF[feat+"_binned"] = logitDF[feat+"_binned"].apply(lambda x: 1 if x == feat+" ≤ {:.2f}".format(thr) else 2)
            form = f'{target} ~ {x_feature}_binned + TRT + TRT:{x_feature}_binned'
            logitfit = smf.logit(formula = str(form), data = logitDF).fit(disp=False)

            ax3.set_xticks(ticks=[0,1])
            ax3.set_xticklabels(labels=['Placebo', 'TRT'])
            ax3.set_ylim([0, 1])
            ax3.set_ylabel(plot_type)
            ax3.set_title(feat+"*treatment\ninteraction: p={:.4f}".format(logitfit.pvalues['TRT:'+x_feature+'_binned']))
            ax3.legend(bbox_to_anchor=(1, -.3))
        else:
            ax3.axis('off')
            pass
        if show_interaction:
            tmpDF = pd.concat([logitfit.params, logitfit.pvalues], axis=1).rename(columns={0:'Coefficients', 1:'pvalues'})
            st.write(tmpDF)
            
    elif plot_type == 'Estimated Tx effect':
        tmpDF = feature_bootsDF[feature_bootsDF[feat].notna()]
        tmpDF['TRT'] = tmpDF['Treatment group'].replace({"Placebo":0, "TRT":1})
        tmpDF[feat+"_binned"] = tmpDF[feat].apply(lambda x: feat+" < {:.2f}".format(thr) if x < thr else feat+" >= {:.2f}".format(thr))
        errDF = tmpDF.groupby([feat+"_binned", 'Treatment group']).agg({'mean_SHAP':['mean', 'std', 'count']})['mean_SHAP'].reset_index()

        for idx in range(4):
            currDF = errDF.iloc[idx,:]
            trt= currDF['Treatment group']
            if (trt =='Placebo') | (trt== "TRT, Placebo"): 
                n_placebo.append(currDF['count'])  
            elif (trt == 'TRT') | (trt== "TRT, TRT"): 
                n_TRT.append(currDF['count'])
            grp = 0 if (currDF[feat+"_binned"]==feat+" < {:.2f}".format(thr)) else 1
            label = currDF[feat+"_binned"] + f"; n={currDF['count']} " + str(trt) 
            color = sns.palettes.color_palette('muted')[0] if (grp == 0) else sns.palettes.color_palette('deep')[1]
            if prediction_time == 'Induction':
                curr_d = {"Placebo":0, "TRT":1}
            elif prediction_time == 'Maintenance':
                curr_d = {"TRT, Placebo":0, "TRT, TRT":1}

            x = curr_d[currDF['Treatment group']]+(.02 if grp == 0 else 0)
            y= currDF['mean']
            yerr = np.array(currDF['std']).flatten()
            if feat !='CTR2':
                ax3.errorbar(x=x, y=y, yerr=yerr, marker='o', ms=15, markeredgecolor=color, mew=4, color=color if trt=='TRT' else 'w', ecolor=color)
                if trt == 'TRT':
                    tx_effect = errDF[errDF[feat+'_binned'] == grp_names[grp]].iloc[0]['mean']- errDF[errDF[feat+'_binned'] == grp_names[grp]].iloc[1]['mean']
                    ann_str = "∆{:.2f}".format(tx_effect)
                    ax3.annotate(xy=(x,y+.01), s=ann_str, color=color, ha='right',va='bottom', size=14)
        if feat !='CTR2':
            ax3.plot([0,1], errDF[errDF[feat+"_binned"]==grp_names[0]].sort_values('Treatment group', ascending=False).reset_index(drop=True)['mean'].values, color=sns.palettes.color_palette('muted')[0], label=grp_names[0])
            ax3.plot([0,1], errDF[errDF[feat+"_binned"]==grp_names[1]].sort_values('Treatment group', ascending=False).reset_index(drop=True)['mean'].values, color=sns.palettes.color_palette('muted')[1], label=grp_names[1])
            ax3.set_xticks(ticks=[0,1])
            ax3.set_xticklabels(labels=['Placebo', 'TRT'])
            ax3.set_ylabel(plot_type)
            ax3.set_title(feat+"*treatment\ninteraction plot")

            ax3.legend(bbox_to_anchor=(1, -.3))

        else:
            ax3.axis('off')
            pass

    elif plot_type == 'Estimated Tx effect (empirical exposure)':
        plot_type == 'Estimated Tx effect'
        exposure_groups=True
        tmpDF = feature_bootsDF[feature_bootsDF[feat].notna()]
        med_val = np.round((df_st[exposure_var][df_st[exposure_var]>0]).median(),2)
        tmpDF['TRT'] = (df_st[exposure_var] > med_val).astype(int) # Here TRT is actaully high TRT
        # tmpDF['TRT'] = tmpDF['Treatment group'].replace({"Placebo":0, "TRT":1})
        tmpDF[feat+"_binned"] = tmpDF[feat].apply(lambda x: feat+" < {:.2f}".format(thr) if x < thr else feat+" >= {:.2f}".format(thr))
        errDF = tmpDF.groupby([feat+"_binned", 'TRT']).agg({'mean_SHAP':['mean', 'std', 'count']})['mean_SHAP'].reset_index()
        for idx in range(4):
            currDF = errDF.iloc[idx,:]
            trt= currDF['TRT']
            if (trt == 0): 
                n_placebo.append(currDF['count'])  
            elif (trt == 1): 
                n_TRT.append(currDF['count'])
            grp = 0 if (currDF[feat+"_binned"]==feat+" < {:.2f}".format(thr)) else 1
            label = currDF[feat+"_binned"] + f"; n={currDF['count']} " + str(trt) 
            color = sns.palettes.color_palette('muted')[0] if (grp == 0) else sns.palettes.color_palette('deep')[1]
            if prediction_time == 'Induction':
                curr_d = {"Placebo":0, "TRT":1}
            elif prediction_time == 'Maintenance':
                curr_d = {"TRT":0, "TRT, TRT":1}

            x = currDF['TRT']+(.02 if grp == 0 else 0)
            y= currDF['mean']
            yerr = np.array(currDF['std']).flatten()
            if feat !='CTR2':
                ax3.errorbar(x=x, y=y, yerr=yerr, marker='o', ms=15, markeredgecolor=color, mew=4, color=color if trt==1 else 'w', ecolor=color)
                if trt == 1:
                    tx_effect = errDF[errDF[feat+'_binned'] == grp_names[grp]].iloc[0]['mean']- errDF[errDF[feat+'_binned'] == grp_names[grp]].iloc[1]['mean']
                    ann_str = "∆{:.2f}".format(tx_effect)
                    ax3.annotate(xy=(x,y+.01), text=ann_str, color=color, ha='right',va='bottom', size=14)
        if feat !='CTR2':
            ax3.plot([1,0], errDF[errDF[feat+"_binned"]==grp_names[0]].sort_values('TRT', ascending=False).reset_index(drop=True)['mean'].values, color=sns.palettes.color_palette('muted')[0], label=grp_names[0])
            ax3.plot([1,0], errDF[errDF[feat+"_binned"]==grp_names[1]].sort_values('TRT', ascending=False).reset_index(drop=True)['mean'].values, color=sns.palettes.color_palette('muted')[1], label=grp_names[1])
            ax3.set_xticks(ticks=[0,1])
            ax3.set_xticklabels(labels=[f'Placebo/\nlow exposure(≤{med_val})', f'High exposure\n>{med_val}'])
            ax3.set_ylabel(plot_type)
            ax3.set_title(feat+"*exposure\ninteraction plot")

            ax3.legend(bbox_to_anchor=(1, -.3))

        else:
            ax3.axis('off')
            pass

    if plot_type == 'Empirical probability (empirical exposure)':
        plot_type='Empirical probability'
        exposure_groups=True
        tmpDF = df_st.iloc[valid_idxs,:]
        tmpDF = tmpDF[tmpDF[feat].notna()]
        # tmpDF['TRT'] = tmpDF['Treatment group'].replace({"Placebo":0, "TRT":1})
        tmpDF[feat+"_binned"] = tmpDF[feat].apply(lambda x: feat+" < {:.2f}".format(thr) if x < thr else feat+" >= {:.2f}".format(thr))

        med_val = np.round((tmpDF[exposure_var][tmpDF[exposure_var]>0]).median(),2)
        tmpDF['TRT'] = (tmpDF[exposure_var] > med_val).astype(int) # Here TRT is actaully high TRT
        st.write(f"Here 'High exposure' is defined as {exposure_var} >median value ({med_val})")
        errDF = tmpDF.groupby([feat+"_binned", 'TRT']).agg({target:['mean', binom_err, 'count']})[target].reset_index()
        if feat != 'CTR2':
            ax3.plot([0,1], errDF[errDF[feat+"_binned"]==grp_names[0]].sort_values('TRT', ascending=True).reset_index(drop=True)['mean'].values, color=sns.palettes.color_palette('muted')[0], label=grp_names[0])
            ax3.plot([0,1], errDF[errDF[feat+"_binned"]==grp_names[1]].sort_values('TRT', ascending=True).reset_index(drop=True)['mean'].values, color=sns.palettes.color_palette('muted')[1], label=grp_names[1])

        for idx in range(4):
            currDF = errDF.iloc[idx,:]
            trt= currDF['TRT']
            trt = 'Placebo' if trt == 0 else 'TRT'
            if trt =='Placebo':
                n_placebo.append(currDF['count'])  
            else: 
                n_TRT.append(currDF['count'])
            grp = 0 if (currDF[feat+"_binned"]==feat+" < {:.2f}".format(thr)) else 1
            label = currDF[feat+"_binned"] + f"; n={currDF['count']} " + str(trt) 
            color = sns.palettes.color_palette('muted')[0] if (grp == 0) else sns.palettes.color_palette('deep')[1]
            x = currDF['TRT']+(.02 if grp == 0 else 0)
            y= currDF['mean']
            yerr = np.array(currDF['binom_err']).reshape(-1,1)

            if feat !='CTR2':
                ax3.errorbar(x=x, y=y, yerr=yerr, marker='o', ms=15, markeredgecolor=color, mew=4, color=color if trt=='TRT' else 'w', ecolor=color)
                if trt == 'TRT':
                    errDF['n_responders'] = errDF['mean']*errDF['count']
                    tx_effect = errDF[errDF[feat+'_binned'] == grp_names[grp]].iloc[1]['mean']- errDF[errDF[feat+'_binned'] == grp_names[grp]].iloc[0]['mean']

                    count = [errDF[errDF[feat+'_binned'] == grp_names[grp]].iloc[0]['n_responders'], errDF[errDF[feat+'_binned'] == grp_names[grp]].iloc[1]['n_responders']]
                    nobs = [errDF[errDF[feat+'_binned'] == grp_names[grp]].iloc[0]['count'], errDF[errDF[feat+'_binned'] == grp_names[grp]].iloc[1]['count']]
                    _, pval = proportions_ztest(count, nobs)
                    ann_str = "∆{:.2f}, p={:.4f}".format(tx_effect, pval)
                    # st.write(ann_str)
                    ax3.annotate(xy=(x,y+.02), text=ann_str, color=color, ha='right',va='bottom', size=14)
        
        if feat !='CTR2':
            # Determine interaction significance

            logitDF = tmpDF[[feat, feat+"_binned", 'TRT', target]].dropna()
            logitDF[feat+"_binned"] = logitDF[feat+"_binned"].apply(lambda x: 1 if x == feat+" < {:.2f}".format(thr) else 2)
            form = f'{target} ~ {x_feature}_binned + TRT + TRT:{x_feature}_binned'
            logitfit = smf.logit(formula = str(form), data = logitDF).fit(disp=False)

            ax3.set_xticks(ticks=[0,1])
            ax3.set_xticklabels(labels=[f'Placebo/\nlow exposure(≤{med_val})', f'High exposure\n>{med_val}'])
            ax3.set_ylim([0, 1])
            ax3.set_ylabel(plot_type)
            ax3.set_title(feat+"*exposure\ninteraction: p={:.4f}".format(logitfit.pvalues['TRT:'+x_feature+'_binned']))
            ax3.legend(bbox_to_anchor=(1, -.3))
        else:
            ax3.axis('off')
            pass
        if show_interaction:
            tmpDF = pd.concat([logitfit.params, logitfit.pvalues], axis=1).rename(columns={0:'Coefficients', 1:'pvalues'})
            # st.write(tmpDF)


    ax3.set_xlim([-.1, 1.1])
    ax4 = fig.add_subplot(nFeats,4,f*4+4)
    facecolors = [sns.palettes.color_palette('muted')[c] for c in [0,0,1,1]]
    facecolors[0] = (.99,.99,.99) 
    facecolors[2] = (.99,.99,.99)
    errDF.plot.bar(x=feat+"_binned", y='count', 
                        ec=[sns.palettes.color_palette('muted')[c] for c in [0,0,1,1]], lw=4,
                        color = facecolors, legend=False, rot=45, ax = ax4)

    plt.setp(ax4.get_xticklabels(), ha='right')
    ax4.set_title('Count of subgroups')
    ax4.set_xlabel('')


    allPs.extend([ps[1],ps[0]])
    allNs.extend([ns[1], ns[0]])
    allYerrs.extend([list(yerrs_abs[1]),list(yerrs_abs[0])])
    allFeats.extend(labels)
    plt.tight_layout()
    fileName = outputs_dir + "{}-cutoff_plots threshold {:.3f}_version1.png".format(x_feature, thr)
    if savefig:
        plt.savefig(fileName, bbox_inches='tight')
        print("Saved: ", fileName)

    empProbDF = pd.DataFrame({'patient group' : allFeats, 
                'probability': allPs, 
                '95% CI': allYerrs, 
                '95% CI lwr & upr':allYerrs_delta, 
                                "n_Placebo":n_placebo,
                                "n_TRT":n_TRT}).reset_index()
    empProbDF['patient group'] = empProbDF['patient group'].apply(lambda x: x.replace('\n', '', 1))
    empProbDF['patient group'] = empProbDF['patient group'].apply(lambda x: x.replace('\n',' '))
    if exposure_groups:
        st.write(empProbDF[['patient group', 'probability', '95% CI', 'n_Placebo', 'n_TRT']].rename(columns={'n_Placebo':'n low_TRT' , 'n_TRT':'n high_TRT'}))
    else:
        st.write(empProbDF[['patient group', 'probability', '95% CI', 'n_Placebo', 'n_TRT']])
    
    return fig