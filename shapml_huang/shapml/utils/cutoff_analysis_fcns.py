from logging import warning
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ..binary_classification.shap_based_analysis_logistic import logistic_shap
import statsmodels.api as sm
from ..utils.misc import add_annotations

def binomVec_yerr(vec):
    from statsmodels.stats.proportion import proportion_confint
    lwr, upr = np.abs(proportion_confint(vec.sum(), len(vec), method='wilson')-np.mean(vec))
    point_estimate = np.mean(vec)
    return point_estimate, lwr, upr

def binom_err(series):
    point_estimate, lwr, upr = binomVec_yerr(series)
    return [lwr,upr]

def additive_interaction(self, x='BHDL', feature_thr=55.0):
    try:
        df=pd.concat([self.df, self.meta_df.copy()],axis=1)                                        
    except:
        df=self.df.copy()
    df=df[df[x].notna()]
    # feature_bins=[df[x].min(),feature_thr, df[x].max()]
    df['GROUP'] = (df[x]>feature_thr).astype(int)
    df['TRT'] = (df[self.exposure_var].isna() | df[self.exposure_var]>0).astype(int)
    df=df[['GROUP', 'TRT', self.target]] 

    risk_difference_0 = df[(df['GROUP']==0) & (df['TRT']==1)].mean()[self.target] - df[(df['GROUP']==0) & (df['TRT']==0)].mean()[self.target]
    risk_difference_1 = df[(df['GROUP']==1) & (df['TRT']==1)].mean()[self.target] - df[(df['GROUP']==1) & (df['TRT']==0)].mean()[self.target]
    group_names = [x+f"≤{feature_thr}", x+f">{feature_thr}"]
    print('Tx-related risk difference for {} : {:.3f}'.format(group_names[0], risk_difference_0))
    print('Tx-related risk difference for {} : {:.3f}'.format(group_names[1], risk_difference_1))
    if risk_difference_0<=risk_difference_1:
        low_risk_group=0
        high_risk_group=1
    else: 
        low_risk_group=1
        high_risk_group=0
        
    print("Group with greater risk difference: ", group_names[high_risk_group])
    p00 = df[(df['GROUP']==low_risk_group) & (df['TRT']==0)].mean()[self.target]
    print(f"{self.target} rate for placebo {group_names[low_risk_group]} (low-risk group): {np.round(p00, 3)}")
    p01 = df[(df['GROUP']==high_risk_group) & (df['TRT']==0)].mean()[self.target] 
    print(f"{self.target} rate for placebo {group_names[high_risk_group]} (high-risk group): {np.round(p01, 3)}")

    p10 = df[(df['GROUP']==low_risk_group) & (df['TRT']==1)].mean()[self.target]
    print(f"{self.target} rate for Tx {group_names[low_risk_group]} (low-risk group): {np.round(p10, 3)}")
    p11 = df[(df['GROUP']==high_risk_group) & (df['TRT']==1)].mean()[self.target]
    print(f"{self.target} rate for Tx {group_names[high_risk_group]} (high-risk group): {np.round(p11, 3)}")
    t = (p11-p00) - ((p10-p00) + (p01-p00)) 

    if low_risk_group==1:
        df['GROUP'] = df['GROUP'].apply(lambda x: int(not x))
    df['GROUP_TRT'] = df.apply(lambda x: x['GROUP']*x['TRT'], axis=1)# This df will be used to test for additive interaction effects
    X = sm.add_constant(df[['GROUP', 'TRT', 'GROUP_TRT']].values, prepend=True)
    y= df[self.target].values
    lm = pd.read_html(sm.OLS(y, X).fit().summary().tables[1].as_html(), header=0, index_col=0)[0]
    lm.index=['Intercept', 'GROUP', 'TRT', 'GROUP:TRT']
    
    print(""" 
    Note: Group 1 is the high risk group
    Asessing the extent to which the effect of the two factors together exceeds the effect of each considered individually 
    t = (p11-p00) - ((p10-p00) + (p01-p00)), where p11 is indicative of rate related to treatment and the high risk group, respectively.
    if t<0 sub-additive, while if t>0 super-additive meaning the effect of two factors together exceeds the effect of each considered indiviually
    Note: t = GROUP:TRT in the OLS model
    Analysis detailed in: https://www.degruyter.com/document/doi/10.1515/em-2013-0005/html
    """)
    if t>0:
        print("t = {:.3f}".format(t), "; therfore, interaction effect is super-additive (i.e. effect of GROUP:TRT exceeds effect of GROUP + effect of TRT)")
    else:
        print("t = {:.3f}".format(t), "; therfore, interaction effect is sub-additive (i.e. effect of GROUP:TRT does not exceed the effect of GROUP + effect of TRT)")
    return lm, t

def shap_interaction_plot_cutoff(self, features, exposure_var, x_feature=None, exposure_thr_vals=None, feature_thr=None, selectionVec=None, return_fig=False, figsize=(6,4), s=20, ylims=None, ylabel=None, ax=None, title_str=None, show_legend=True,
                                loc=1, bbox_to_anchor=(1.15, .7)):
    feature=features[0]
    if type(exposure_thr_vals) == type(None):
        exposure_thr_vals=[float(self.df[exposure_var].min()), float(self.df[exposure_var].max())]
    if type(x_feature)==type(None):
        x_feature=feature
    if type(feature_thr) == type(None):
        feature_thr=self.df[feature].min()
    if type(selectionVec) == type(None):
        selectionVec=range(self.df.shape[0])

    if type(ax) == type(None): 
        fig,ax = plt.subplots(1,1, figsize=figsize)
    # ax = fig.add_subplot(111)
    tmpDF = pd.DataFrame({x_feature: self.df.drop(columns = self.target).iloc[:,self.mdlFeatures.index(x_feature)].values, 
    exposure_var: self.df[[exposure_var]].values.flatten()})
    tmpDF = tmpDF.loc[selectionVec, :]
    if x_feature != feature:
        tmpDF[feature] = self.df.drop(columns = self.target).loc[selectionVec, :].iloc[:,self.mdlFeatures.index(feature)].values
    if x_feature == exposure_var:
        color_by=feature
    else:
        color_by=exposure_var

    if 'shap_exposure_interaction_prob_df' in self.__dir__():
        y_scale = '∆ probability'
        tmpDF['SHAP value'] = self.shap_exposure_interaction_prob_df.loc[selectionVec, :][features].sum(axis=1)
    else: 
        if self.hyperparams['objective'] == 'binary:logistic':
            print('Interaction values in probability scale were not found; using log-odds scale')
            print('Hint: self.generate_shap_exposure_interaction_prob_df(exposure_var)')
            y_scale='log-odds'
            tmpDF['SHAP value'] = self.shap_interaction_values[selectionVec, self.mdlFeatures.index(exposure_var)][:,[self.mdlFeatures.index(feature) for feature in features]].sum(axis=1)    
        else:
            y_scale='∆ prediction'
            tmpDF['SHAP value'] = self.shap_interaction_values[selectionVec, self.mdlFeatures.index(exposure_var)][:,[self.mdlFeatures.index(feature) for feature in features]].sum(axis=1)
    ylabel = ylabel + f' ({y_scale})'
    
    sns.scatterplot(x=x_feature, y='SHAP value', hue = color_by, data = tmpDF[(tmpDF[exposure_var]>=exposure_thr_vals[0]) & (tmpDF[exposure_var]<=exposure_thr_vals[1]) ], 
                    palette=sns.color_palette(palette='RdYlBu_r', as_cmap=True), s=s, ax = ax, 
                    hue_norm=matplotlib.colors.Normalize(vmin=self.df[color_by].quantile(.1), vmax=self.df[color_by].quantile(.9), clip=True),
                    edgecolor='gray')
    if type(title_str) == type(None):
        ax.set_title(f'Impact of {", ".join(features)} on ER relationship\n (ER relationship: Effect of {exposure_var} on {self.target})')
    else:
        ax.set_title(title_str)
    xlims= [tmpDF[x_feature].quantile(.01)-tmpDF[x_feature].std(), tmpDF[x_feature].quantile(.99)+tmpDF[x_feature].std()]
    ax.set_xlim(xlims)
    if type(ylims) == type(None):
        ylims = [tmpDF['SHAP value'].min()-tmpDF['SHAP value'].std(), tmpDF['SHAP value'].max()+tmpDF['SHAP value'].std()]
    ax.set_ylim(ylims)
    if type(ylabel) != type(None):
        ax.set_ylabel(ylabel)
    if x_feature == feature:
        ax.plot([feature_thr, feature_thr], ylims, '--k',alpha=.5, lw=1.5)
    ax.plot(xlims, [0, 0], '--k',alpha=.5)
    if show_legend:
        ax.legend(loc=loc, bbox_to_anchor=bbox_to_anchor, title=color_by)
    else:
        ax.legend().remove()
    ax.grid(alpha=.5, ls='--')
    if return_fig:
        return fig
    
def get_tx_binned_feature_interaction_effect(df, dose_var, x, feature_thr, y):
    """
    dose_var : used to determine treatment/placebo arms 
    x : feature to be binned for interaction analysis
    feature_thr : x≤feature_thr, x>feature_thr
    y : target variable
    """
    tmpDF = df[[dose_var, x, y]].rename(columns={y:'target'})
    selectVec = (tmpDF[x].notna() & tmpDF[dose_var].notna())
    tmpDF=tmpDF[selectVec]
    tmpDF["Tx"] = (tmpDF[dose_var]>0).astype(int)
    if type(feature_thr) == type(None):
        feature_thr = tmpDF[x].median()
    tmpDF[x+'_bin'] = (tmpDF[x]>feature_thr).astype(int)
    tx_interaction_analysis = logistic_shap(df=tmpDF, target='target', remove_outliers=False)
    out = tx_interaction_analysis.summary_stats(f'target~Tx + {x}_bin + {x}_bin:Tx');
    interaction_effect = pd.DataFrame(out.tables[1].data[1:], columns = out.tables[1].data[0]).set_index('').astype(float).loc[f"{x}_bin:Tx"]
    return interaction_effect

def cutoff_analysis(self, x='BHBA1C', y='HGLY≥2', feature_thr = None, exposure_var = 'Cminsd', dose_var = None, trt_arm_name='Tx', 
                    ms=10, mew=.5, annotation_font_size=14, ylims = [0, .7] , y_label_offset_1 = .05, y_label_offset_2 = .05, 
                    x_offset_1 = 0, x_offset_2 = 0, figsize=(24,10), show_interaction=True, save_fig=False, return_fig=False,
                    legend_font_size=12, return_table=False, interaction_type='additive', fig_labels=True, **kwargs):
    """
    kwargs get supplied to shap_interaction_plot_cutoff (i.e. selectionVec)
    """                
    # if type(df)==type(None):
    #     print('No df was supplied therefore, using self.df')
    def sep(count, nobs):
        p=count/nobs
        return np.sqrt(p*(1-p)/nobs)
    def sep_difference(count1, nobs1, count2,nobs2):
        return np.sqrt(sep(count1,nobs1)**2+sep(count2,nobs2)**2)

    try:
        df=pd.concat([self.df, self.meta_df.copy()],axis=1)                                        
    except:
        df=self.df.copy()
    if type(feature_thr)==type(None):
        feature_thr = df[x].median()
    feature_bins=[df[x].min(),feature_thr, df[x].max()]
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 8, hspace=.3,wspace=3)
    try: # This allows for generating cutoff analysis using features not in the 
        assert(x in self.df.columns)
        ax01 = fig.add_subplot(gs[0, 0:2])
        features = [x]
        features.extend(set(self.mdlFeatures).difference([x]))
        shap_interaction_plot_cutoff(self, x_feature=self.exposure_var, features=features, exposure_var=exposure_var, 
                            feature_thr=feature_thr, s=30, ylabel=f'1st-order SHAP value\n{self.exposure_var}', ax=ax01,
                            title_str="ER relationship", loc=4, bbox_to_anchor=(1, 0))
        ax02 = fig.add_subplot(gs[0, 2:4])
        shap_interaction_plot_cutoff(self, x_feature=self.exposure_var, features=[x], exposure_var=exposure_var, 
                            feature_thr=feature_thr, s=30, ylabel=f'2nd-order SHAP value\n{exposure_var}:{x}', 
                                    ax=ax02, show_legend=False, title_str=f"Impact of {x} on ER",**kwargs)

        ax03 = fig.add_subplot(gs[0, 4:])
        shap_interaction_plot_cutoff(self, x_feature=x, features=[x], exposure_var=exposure_var, 
                            feature_thr=feature_thr, s=30, ylabel=f'2nd-order SHAP value\n{exposure_var}:{x}', ax=ax03, bbox_to_anchor=(1.2, .7),**kwargs)
    except:
        
        pass

    ax1=fig.add_subplot(gs[1,0:4])
    # Plot Placebo:feature on ax1:
    color='gray'
    if type(dose_var)==type(None):
        print("No dose_var was supplied assumming placebo pts using: ((df[exposure_var]==0)")
        tmpDF = df[(df[exposure_var]==0)]
    else: 
        tmpDF = df[df[dose_var]==0]
    n=tmpDF.shape[0]
    tmpDF[x+'_binned'] = pd.cut(tmpDF[x], bins=feature_bins, include_lowest=True)
    errDF = tmpDF.groupby(x+'_binned').agg({y:['mean', binom_err, 'count']}).reset_index()
    errDF.columns = [x+'_binned', 'mean', 'binom_err', 'count']

    legend_label=f'Placebo, n={n}'
    placebo_errDF=errDF
    for idx in range(len(tmpDF[x+'_binned'].unique().categories)):
        currDF = errDF.iloc[idx,:]
        x_val = idx + x_offset_1#had this at left
        y_val= currDF['mean']
        yerr = np.array(currDF['binom_err']).reshape(-1,1)
        plt.errorbar(x=x_val, y=y_val, yerr=yerr, marker='o', ms=ms, mew=mew,mec='gray', color=color, 
                    label=legend_label if idx == 0 else None)
    plt.plot(errDF.index+x_offset_1, errDF['mean'], color=color, ls=':')
    plt.text(.5, errDF['mean'].mean()+y_label_offset_1, s="∆={:.2f}".format(errDF['mean'][1]-errDF['mean'][0]) ,color=color, fontsize=annotation_font_size)

    #Plot Tx:feature on ax1:
    color='red'
    if type(dose_var)==type(None):
        print("No dose_var was supplied assumming Tx pts using: (df[exposure_var].isna()) | (df[exposure_var]>0)")
        tmpDF = df[(df[exposure_var].isna()) | (df[exposure_var]>0)]
        dose_var = 'dose'
        df['dose'] = df.apply(lambda x: int(np.isnan(x[exposure_var]) | (x[exposure_var]>0)), axis=1)
    else: 
        tmpDF = df[df[dose_var]>0]
    n=tmpDF.shape[0]
    tmpDF[x+'_binned'] = pd.cut(tmpDF[x], bins=feature_bins, include_lowest=True)
    legend_label=f'{trt_arm_name}, n={n}'

    errDF = tmpDF.groupby(x+'_binned').agg({y:['mean', binom_err, 'count']}).reset_index()
    errDF.columns = [x+'_binned', 'mean', 'binom_err', 'count']
    tx_errDF = errDF.copy()
    for idx in range(len(tmpDF[x+'_binned'].unique().categories)):
        currDF = errDF.iloc[idx,:]
        x_val = idx + x_offset_1#had this at left
        y_val= currDF['mean']
        yerr = np.array(currDF['binom_err']).reshape(-1,1)
        ax1.errorbar(x=x_val, y=y_val, yerr=yerr, marker='o', ms=ms, mew=mew,mec='gray', color=color, 
                    label=legend_label if idx == 0 else None)
    ax1.plot(errDF.index+x_offset_1, errDF['mean'], color=color, ls=':')
    ax1.text(.4, errDF['mean'].mean()+y_label_offset_1, s="∆={:.2f}".format(errDF['mean'][1]-errDF['mean'][0]) ,color=color, fontsize=annotation_font_size)
    
    ax1.set_yticks(np.arange(ylims[0],ylims[1],.05))
    ax1.set_ylim([ylims[0],ylims[1]])
    ax1.set_xticks([0,1])
    ax1.set_xticklabels([f'{x}≤{feature_thr}\nn={(df[x]<=feature_thr).sum()}', f'{x}>{feature_thr}\nn={(df[x]>=feature_thr).sum()}'])
    hl=df[y].mean()
    ax1.hlines(hl, ax1.get_xlim()[0], ax1.get_xlim()[1], color='gray', ls='--', lw=1)
    ax1.legend(loc=1, bbox_to_anchor=(1, 1), fontsize=legend_font_size)
    ax1.grid(alpha=.5, ls=':', axis='y')
    
    title_str = f"{y} rate vs. {x} binned"
    
    lm,t = additive_interaction(self, x=x, feature_thr=feature_thr) # This is used regardless of the type of interaction we show in the plot
    if show_interaction: 
        if interaction_type=='logistic':
            interaction_effect = get_tx_binned_feature_interaction_effect(df, dose_var=dose_var, x=x, feature_thr=feature_thr, y=y)
            title_str = title_str + f"\n{x}_binned:Tx : {np.round(interaction_effect['coef'],2)}, p={interaction_effect['P>|z|']}"
        elif interaction_type=='additive':
            interaction_effect = lm.loc['GROUP:TRT']
            title_str = title_str + "\n{}_binned:Tx : {:.2f}, p={:.3f}".format(x, interaction_effect['coef'], interaction_effect['P>|t|'])
            if t>0:
                title_str += " (super-additive)"
            elif t<0:
                title_str += " (sub-additive)"
    
    ax1.set_title(title_str)
    ax1.set_ylabel(f"Empirical\n{y} rate")    
    
    count1 = int(placebo_errDF['mean'][0]*placebo_errDF['count'][0]) # HGLY in placebo group with low feature values
    nobs1 = int(placebo_errDF['count'][0]) # N Placebo group with low feature values
    count2 = int(tx_errDF['mean'][0]*tx_errDF['count'][0]) # HGLY in Tx group with low feature values
    nobs2 = int(tx_errDF['count'][0]) # N Tx group with low feature values
    CI=np.round(sep_difference(count1=count1, nobs1=nobs1, count2=count2,nobs2=nobs2)*1.96,3)
    ax1.text(0, (tx_errDF['mean'][0]+placebo_errDF['mean'][0])/2, s="∆={:.3f}±{:.3f}".format(tx_errDF['mean'][0]-placebo_errDF['mean'][0],CI), color='black', fontsize=annotation_font_size)
    
    count1 = int(placebo_errDF['mean'][1]*placebo_errDF['count'][1]) # HGLY in placebo group with low feature values
    nobs1 = int(placebo_errDF['count'][1]) # N Placebo group with low feature values
    count2 = int(tx_errDF['mean'][1]*tx_errDF['count'][1]) # HGLY in Tx group with low feature values
    nobs2 = int(tx_errDF['count'][1]) # N Tx group with low feature values
    CI=np.round(sep_difference(count1=count1, nobs1=nobs1, count2=count2,nobs2=nobs2)*1.96,3)
    ax1.text(1, (tx_errDF['mean'][1]+placebo_errDF['mean'][1])/3, s="∆={:.3f}±{:.3f}".format(tx_errDF['mean'][1]-placebo_errDF['mean'][1], CI), color='black', fontsize=annotation_font_size)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.plot(np.array([0,0])-.01, [placebo_errDF['mean'][0], tx_errDF['mean'][0]], ':k')
    ax1.plot(np.array([1,1])-.01, [placebo_errDF['mean'][1], tx_errDF['mean'][1]], ':k')
    ###############################
    ax2 = fig.add_subplot(gs[1,4:])
    # Plot placebo: 
    color='gray'
    tmpDF = df[df[exposure_var]==0]
    n=tmpDF.count()[exposure_var]
    tmpDF[x+'_binned'] = pd.cut(tmpDF[x], bins=feature_bins, include_lowest=True)
    errDF = tmpDF.groupby(x+'_binned').agg({y:['mean', binom_err, 'count']}).reset_index()
    errDF.columns = [x+'_binned', 'mean', 'binom_err', 'count']
    errDF[exposure_var+'_bin'] = f"{0}"
    errDF[exposure_var+'_count'] = n
    legend_label=f'Placebo, n={n}'
    
    for idx in range(len(tmpDF[x+'_binned'].unique().categories)):
        currDF = errDF.iloc[idx,:]
        x_val = idx + x_offset_2#had this at left
        y_val= currDF['mean']
        yerr = np.array(currDF['binom_err']).reshape(-1,1)
        ax2.errorbar(x=x_val, y=y_val, yerr=yerr, marker='o', ms=ms, mew=mew,mec='gray', color=color, label=legend_label if idx == 0 else None)
    ax2.plot(errDF.index+x_offset_2, errDF['mean'], color=color, ls=':')
    ax2.text(.5, errDF['mean'].mean()+y_label_offset_2, s="∆={:.2f}".format(errDF['mean'][1]-errDF['mean'][0]) ,color=color, fontsize=annotation_font_size)
    # ax2.legend(loc=1, bbox_to_anchor=(.8, 1.5))

    # Plot Tertiles of exposure
    exposure_bins = 3
    # colors = sns.color_palette(palette='RdYlBu_r', n_colors=exposure_bins)
    colors = sns.color_palette(palette='RdYlBu_r', n_colors=8)
    colors = [colors[0], colors[6], colors[-1]]

    for i in range(exposure_bins):
        lwr = df[df[exposure_var]>0][exposure_var].quantile((1/exposure_bins)*i); 
        upr = df[df[exposure_var]>0][exposure_var].quantile((1/exposure_bins)*(i+1));
        tmpDF = df[df[exposure_var]>0][(df[exposure_var]>=lwr) & (df[exposure_var]<upr)]
        tmpDF[x+'_binned'] = pd.cut(tmpDF[x], bins=feature_bins, include_lowest=True)
        n=tmpDF.count()[exposure_var]
        legend_label=f'T{i+1}: {exposure_var} {np.round(lwr,2)}-{np.round(upr,2)}, n={n}'
        errDF = tmpDF.groupby(x+'_binned').agg({y:['mean', binom_err, 'count']}).reset_index()
        errDF.columns = [x+'_binned', 'mean', 'binom_err', 'count']
        errDF[exposure_var+'_bin'] = f"{np.round(lwr,2)}-{np.round(upr,2)}"
        errDF[exposure_var+'_count'] = n
        color=colors[i]
        x_offset = (i-(exposure_bins//2))*.02+x_offset_2
        for idx in range(len(tmpDF[x+'_binned'].unique().categories)):
            currDF = errDF.iloc[idx,:]
            x_val = idx + x_offset#had this at left
            y_val= currDF['mean']
            yerr = np.array(currDF['binom_err']).reshape(-1,1)
            ax2.errorbar(x=x_val, y=y_val, yerr=yerr, marker='o', ms=ms, mew=mew,mec='gray', color=color, 
                        label=legend_label if idx == 0 else None)
        ax2.plot(errDF.index+x_offset, errDF['mean'], color=color, ls=':')
        ax2.text(.5+x_offset, errDF['mean'].mean()+y_label_offset_2, s="∆={:.2f}".format(errDF['mean'][1]-errDF['mean'][0]) ,color=color, fontsize=annotation_font_size)
    ax2.set_ylim([ylims[0],ylims[1]])
    ax2.set_yticks(np.arange(ylims[0],ylims[1],.05))
    ax2.hlines(hl, ax2.get_xlim()[0], ax2.get_xlim()[1], color='gray', ls='--', lw=1)
    ax2.legend(loc=2, bbox_to_anchor=(0, 1), fontsize=legend_font_size)
    ax2.grid(alpha=.5, ls=':', axis='y')
    ax2.set_title(f"{y} rate vs. {x}")
    ax2.set_ylabel(f"Empirical\n{y} rate")    
    ax2.set_xticks([0,1])
    ax2.set_xticklabels([f'{x}≤{feature_thr}\nn={(df[x]<=feature_thr).sum()}', f'{x}>{feature_thr}\nn={(df[x]>=feature_thr).sum()}'])
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    plt.tight_layout()
    if fig_labels:
        add_annotations(fig, x_offset=-40, y_offset=25, fontsize=30, weight='bold')
    if save_fig:
        file_name= self.outputs_dir + f'{x}_interaction_plots.png'
        plt.savefig(file_name, bbox_inches='tight', dpi=300)
        print("Saved: ", file_name)
    
    placebo_errDF['TRT']=0
    tx_errDF['TRT']=1
    tab_out = pd.concat([placebo_errDF, tx_errDF])
    tab_out['Covariate'] = x
    tab_out=tab_out.rename(columns={x+"_binned":"bin"}).reset_index().rename(columns={'index': 'Group'})

    tab_out['Rate (%) (95% CI)'] = tab_out.apply(lambda x: "{:.2f} ({:.2f}, {:.2f})".format(x['mean']*100, x['binom_err'][0]*100, x['binom_err'][1]*100),axis=1)
    tab_out['SEP'] = tab_out.apply(lambda x: sep(int(x['mean']*x['count']), int(x['count'])), axis=1)

    # For the combined std: multiplying by 100 after taking the sqrt puts the std into the correct right order of magnitude.
    tab_out['Risk difference ± std (Tx-Placebo)'] = np.nan
    tab_out.loc[2, 'Risk difference ± std (Tx-Placebo)'] = "{:.2f}±{:.2f}".format((tab_out.loc[2, 'mean']-tab_out.loc[0, 'mean'])*100, 100*np.sqrt(tab_out.loc[0, 'SEP']**2+tab_out.loc[2, 'SEP']**2))
    tab_out.loc[3, 'Risk difference ± std (Tx-Placebo)'] = "{:.2f}±{:.2f}".format((tab_out.loc[3, 'mean']-tab_out.loc[1, 'mean'])*100, 100*np.sqrt(tab_out.loc[1, 'SEP']**2+tab_out.loc[3, 'SEP']**2))

    tab_out['Relative risk ± std (Tx/Placebo)'] = np.nan
    tab_out.loc[2, 'Relative risk ± std (Tx/Placebo)'] = "{:.2f}±{:.2f}".format(tab_out.loc[2, 'mean']/tab_out.loc[0, 'mean'], 100*np.sqrt((tab_out.loc[0, 'SEP']**2+tab_out.loc[2, 'SEP']**2)/tab_out.loc[0, 'mean']))
    tab_out.loc[3, 'Relative risk ± std (Tx/Placebo)'] = "{:.2f}±{:.2f}".format(tab_out.loc[3, 'mean']/tab_out.loc[1, 'mean'], 100*np.sqrt((tab_out.loc[1, 'SEP']**2+tab_out.loc[3, 'SEP']**2)/tab_out.loc[1, 'mean']))

    
    s= "{:.3f}, p={:.5f}".format(interaction_effect['coef'], interaction_effect['P>|t|'])
    if t>0: 
        s+= " (super-additive)"
    else:
        s+= " (sub-additive)"
    tab_out['Additive interaction'] = np.nan
    tab_out.loc[3,'Additive interaction'] = s
    tab_out=tab_out[['Covariate', 'bin', 'count', 'Rate (%) (95% CI)', 'Risk difference ± std (Tx-Placebo)', 'Additive interaction', 'Relative risk ± std (Tx/Placebo)', 
            'Group', 'mean', 'binom_err', 'TRT', 'SEP']]

    if return_fig & return_table:
        return fig, tab_out
    elif return_fig:
        return fig
    elif return_table: 
        return tab_out