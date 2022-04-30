from math import exp
import os, glob
import re
from bokeh.models.markers import X
from numpy.lib.type_check import real
import seaborn
import streamlit as st
import numpy as np
import pandas as pd
import base64
# from bokeh.plotting import figure
# import datetime
# from datetime import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
import pickle 
import sys, importlib
sys.path.append("/Users/harunr/proj/shapml/") # replace with path-to-RH
import shapml
from shapml.utils.misc import shap_interaction_plot, shap_interaction_plot_combined
import copy, time, itertools
import seaborn as sns
import shapml.binary_classification as BC
from lib.model_recovery_functions import f_inv_logit, genrate_potential_mdl_terms, generate_df_for_LR_synthetic, generate_df_for_LR_real, synthesize_outcomes, prob_plot, stratified_prob_plot, cutoff_plot_st
from shapml.utils.misc import remove_explainers_models
from shapml.utils.cutoff_analysis_fcns import cutoff_analysis
from streamlit_ace import st_ace

# HERE = os.path.dirname(os.path.abspath(__file__))
def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="dataset.csv">Download csv file</a>'
    return href

st.set_page_config(page_title = 'binary classification', page_icon= '⚕️', layout="wide")
# st.sidebar.subheader('Navigation')
import yaml
meta_data = yaml.safe_load(open('./lib/app.yaml', 'r'))

outputs_dir=meta_data['outputs_dir']
target=meta_data['target']

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_analysis(outputs_dir, target, analysis_name):
    fileName = max(glob.glob(outputs_dir + target+ f"_{analysis_name}*.p"), key=os.path.getctime)
    print("Loaded:", fileName)
    st.write("Loaded:", fileName)
    analysis= pickle.load(open(fileName, 'rb'))
    return analysis

analysis=load_analysis(outputs_dir, target, analysis_name='streamlit')

try:
    exposure_var = analysis.exposure_var
except:
    exposure_var = 'undefined'

# FE=load_analysis(outputs_dir, target, analysis_name='init_FS').feature_selection
# st.write(FE.df)
# st.write(pd.read_csv("./data/trt_grps.csv"))
header = st.container()
with header: 
    st.title(analysis.target + " model")
try:
    st.write('Exposure: ' + analysis.exposure_var)
except:
    pass
section = st.radio('Section:',
         ['Dataset',
         'Correlation structure',
         'Model performance',  
         'SHAP summary',
         'Dependence plots', 
         'Statistical summary (XGBoost)',
         'Statistical summary (LR)',
         'SHAP-based interactions',
         'Cut-off analysis', 
         'Generate (semi)synthetic data', 
         'Model recovery analysis'], index=3)


if section == 'Dataset':
    # st.write()
    msno_fig = analysis.plot_missingness(save_fig=False, colorBy=target, show=False, y_labelpad=25)
    msno_fig.axes[0].set_title(f"{len(analysis.orderedFeatures)} features selected"+ msno_fig.axes[0].get_title(), fontsize=30)
    st.write(msno_fig)
    data_container = st.expander(label="Dataset")
    with data_container: 
        st.markdown(get_table_download_link(analysis.df), unsafe_allow_html=True)
        st.write(analysis.df)

elif section == 'Correlation structure':
    st.write("Correlations matrix")
    st.write(analysis.correlation_martix())
    st.write(analysis.clustermap())
    pca_color_by = st.selectbox('color PCA by', [target, 'prediction'])
    st.write(analysis.plot_pca(return_fig=True, color_by=pca_color_by))

elif section == 'Model performance': 
    stratify_by_list = ['None']
    stratify_by_list.extend(analysis.orderedFeatures)
    stratify_by_list.extend(analysis.meta_columns)
    stratify_by = st.selectbox('Stratify calibration by', stratify_by_list, index=0)
    if stratify_by=='None':
        st.write(analysis.plot_reliability_SHAP(stratify=None, show=False))
    else:
        repair_df = False
        if stratify_by not in analysis.df.columns:
            repair_df = True
            tmpDF = analysis.df.copy()
            analysis.df[stratify_by] = analysis.meta_df[stratify_by]
        st.write(analysis.plot_reliability_SHAP(stratify=stratify_by, show=False))
        if repair_df:
            analysis.df=tmpDF
        # st.write(analysis.plot_reliability(stratify=stratify_by, show=False))
    scores, print_str = analysis.model_performance(n_folds=5, n_reps=5)
    st.write(print_str)
    # Plot ROC
    stratify_by = st.selectbox('Stratify ROC by', stratify_by_list, index=0)
    if stratify_by=='None':
        st.write(analysis.plot_ROC(n_reps=5, stratify=None, plot_width=550, return_fig=True))
    else:
        st.write(analysis.plot_ROC(n_reps=5, stratify=stratify_by, plot_width=550, return_fig=True))

    # col1C, col1D = st.columns(2)
    # with col1C:
    #     thr = st.number_input('threshold', step=.05, value=.5)
    # chart = st.container()
    # with chart:
    #     st.write(analysis.plot_binary_classification_graphs(threshold=thr, return_layout=False), use_container_width=True)
elif section == 'SHAP summary':
    st.write(f"{len(analysis.orderedFeatures)} features")
    st.write(analysis.shap_summary_plots(figsize=(20,12), save_fig=False, show=False))
    st.write('Order of importance:')
    st.write(analysis.orderedFeatures)
elif section == 'Statistical summary (XGBoost)':
    st.write(analysis.plot_bootstrap_summary_table(figsize=(15,20), label='XGB model', show_n=True, return_fig=True))
    fig,out_df = analysis.plot_empirical_rates()
    st.write(fig)
elif section == 'Statistical summary (LR)':
    lr_shap = BC.logistic_shap(df=analysis.df, target=analysis.target)
    st.write(lr_shap.summary_stats()) # This is a standard LR model w/o regularization
elif section == 'Dependence plots':
    col1A, col2A = st.columns(2)
    with col1A: 
        x_feat = st.selectbox('feature (x-axis) (ordered by feature importance)', analysis.orderedFeatures)
    with col2A:
        shap_features = st.multiselect('∑Shap values (y-axis)', analysis.mdlFeatures, default=x_feat)
    col1B, col2B = st.columns(2)
    with col1B: 
        color_by_list = ['None']
        color_by_list.extend(analysis.orderedFeatures)
        color_by_list.extend(analysis.meta_columns)
        color_by = st.selectbox('color by', color_by_list, index=0)
        if color_by == 'None': 
            color_by = None
    with col2B:
        nQuantiles = st.selectbox('quantiles', [2,3,4,5,8,10], index=5)
        ylims = st.slider('ylim', -1.0, 1.0, (-.2, .5))
        repair_df = False
        if type(color_by) != type(None):
            if color_by not in analysis.df.columns:
                repair_df = True
                tmpDF = analysis.df.copy()
                analysis.df[color_by]=analysis.meta_df[color_by]
        
        # TODO: Fix error: For some discrete features, this doesn't work well in streamlit
        try:
            fig, summaryDF = analysis.plot_bootstrapped_feature_dependence(x_feature=x_feat, shap_features =shap_features, color_by=color_by, figsize=(16,5), ylims=ylims, ms0=7,
                                              yaxis_label='∆ probability', level_type='categorical' if type(color_by)==type(None) else 'sequential', return_fig=True, nQuantiles=nQuantiles, return_summary_table=True)
        except:
            fig = analysis.plot_bootstrapped_feature_dependence(x_feature=x_feat, shap_features =shap_features, color_by=color_by, figsize=(16,5), ylims=ylims,ms0=7,
                                              yaxis_label='∆ probability', level_type='categorical' if type(color_by)==type(None) else 'sequential', return_fig=True, nQuantiles=nQuantiles, return_summary_table=False)        
                        
        if repair_df: 
            analysis.df = tmpDF
    st.write(fig)
    col1D, col2D = st.columns(2)
    with col2D:
        st.write(summaryDF)

elif section == 'Cut-off analysis':
    st.write(analysis.shap_exposure_impacts(return_fig=True))

    df_st = pd.concat([analysis.df, analysis.meta_df], axis=1)
    col1C, col2C, col3C = st.columns(3)
    with col1C:

        cutoff_features = copy.deepcopy(analysis.orderedFeatures)
        try:
            cutoff_features.pop(cutoff_features.index(exposure_var))
        except:
            pass
            cutoff_features.extend(analysis.meta_columns)
        x_feature = st.selectbox('feature (x-axis) (ordered by feature importance)', cutoff_features)
    with col2C: 
        thr=st.number_input('threshold', step=1.0, value=np.round(df_st[x_feature].median(), 2))
    with col3C: 
        plot_type = st.selectbox('Interaction plot type', ['Estimated Tx effect', 'Empirical probability', 'Empirical probability (empirical exposure)','Estimated Tx effect (empirical exposure)'], 1)
    st.write(analysis.cutoff_analysis(x=x_feature, feature_thr=thr, legend_font_size=10, return_fig=True))
    # fig = cutoff_plot_st(x_feature=x_feature, thr=thr, plot_type=plot_type)
    # st.write(cutoff_plot_st(analysis, shap_features, df_st, prediction_time, exposure_var, valid_idxs=valid_idxs, outputs_dir=outputs_dir, x_feature=x_feature, thr=thr, plot_type=plot_type))
    primary_feature=exposure_var
    df_st['Treatment_Flag'] = ((df_st[exposure_var] > 0)| (df_st[exposure_var].isna())).astype(int)
    try:
        df_st.rename(columns = {'HGLY≥2':'HGLY2'}, inplace=True)
        tmp_target = 'HGLY2'
        tmpAnalysis = analysis.copy()
        tmpAnalysis.target=tmp_target
        tmpAnalysis.df.rename(columns = {'HGLY≥2':'HGLY2'}, inplace=True)
    except:
        tmpAnalysis = analysis.copy()
        pass

    st.write(cutoff_plot_st(analysis=tmpAnalysis, shap_features=primary_feature, df_st=df_st, prediction_time='Semisynthetic', exposure_var=primary_feature, Treatment_Flag='Treatment_Flag', valid_idxs=list(range(analysis.df.shape[0])), outputs_dir=outputs_dir, x_feature=x_feature, thr=thr, plot_type=plot_type, bootstrap_analysis=False))
    # if plot_type == 'Empirical probability':
    #     st.write("Using 'Cminsd' as dosevar rather than 'IPADOSE' ")
    #     st.write(analysis.cutoff_analysis(x='BHDL', feature_thr=thr, legend_font_size=10, return_fig=True))
        # st.write(cutoff_analysis(analysis, df=df_st, dose_var='Cminsd', x=x_feature, y=target, feature_thr = thr, show_interaction=True, return_fig=True));
    # if x_feature in analysis.mdlFeatures: 
    #     selectionVec=list(range(analysis.df.shape[0]))
    #     exposure_thr_vals = st.slider(min_value=float(analysis.df[exposure_var].min()), max_value=float(analysis.df[exposure_var].max()), value=(float(analysis.df[exposure_var].min()), float(analysis.df[exposure_var].max())), label=f'min/max {exposure_var}')
    #     st.write(shap_interaction_plot(analysis, x_feature, exposure_thr_vals=exposure_thr_vals, feature_thr=thr, exposure_var=exposure_var, selectionVec=selectionVec, return_fig=True))
    
    q=st.slider(label='quantiles', min_value=2, max_value=20, value=4)
    
    fig = stratified_prob_plot(x=x_feature, stratify_thr=0,
        y='HGLY2', q=q, df=df_st, stratify=exposure_var, ylims=None, 
        hl=df_st['HGLY2'].mean())
    plt.grid(alpha=.5)
    plt.yticks(np.linspace(0,.5,11))
    st.write(fig)

elif section == 'SHAP-based interactions': 
    selectionVec=None
    potential_primary_features = [exposure_var]
    potential_primary_features.extend(list(set(analysis.mdlFeatures).difference([exposure_var])))
    primary_feature = st.selectbox('primary feature', potential_primary_features, index=0)
    col1, col2=st.columns(2)
    with col1:
        st.write(analysis.shap_interaction_summary(feature=primary_feature, selectionVec=selectionVec, show=False))
    with col2:
        st.write(analysis.shap_exposure_impacts(exposure_var= primary_feature, selectionVec=selectionVec, return_fig=True))
    fig = analysis.shap_interaction_summary(feature=primary_feature, selectionVec=selectionVec, show=False, max_display=len(analysis.mdlFeatures))
    interaction_features = list(reversed([s._text for s in fig.get_children()[1].get_yticklabels()]))
    import seaborn as sns
    import matplotlib
    col1, col2=st.columns(2)
    with col1:
        features = st.multiselect(f'show feature(s) interactions with {primary_feature}', interaction_features, default=interaction_features[1])
        feature=features[0]
        feature_thr = st.slider(min_value=float(analysis.df[feature].min()), max_value=float(analysis.df[feature].max()), value=float(analysis.df[feature].median()), label=f'potential cutoff ({feature})')
        s=st.slider('marker size', min_value=1, max_value=100, value=10)
    with col2:
        color_by = st.selectbox(f'color by', interaction_features, index=interaction_features.index(primary_feature))
        exposure_range = st.slider(min_value=float(analysis.df[primary_feature].min()), max_value=analysis.df[primary_feature].max(), value=(float(analysis.df[primary_feature].min()), float(analysis.df[primary_feature].max())), label=f'min/max {primary_feature}')
        if color_by != primary_feature:
            feature_range = st.slider(min_value=float(analysis.df[color_by].min()), max_value=analysis.df[color_by].max(), value=(float(analysis.df[color_by].min()), float(analysis.df[color_by].max())), label=f'min/max {color_by}')
            selectionVec=(analysis.df[primary_feature]>=exposure_range[0]) & (analysis.df[primary_feature]<=exposure_range[1]) & (analysis.df[color_by]>=feature_range[0]) & (analysis.df[color_by]<=feature_range[1])
        else:
            selectionVec=(analysis.df[primary_feature]>=exposure_range[0]) & (analysis.df[primary_feature]<=exposure_range[1])
        # color_norm = st.selectbox('color norm', options=['linear', 'log'])
    
    st.write(shap_interaction_plot_combined(analysis, exposure_var=primary_feature, features=features, hue =color_by, selectionVec=selectionVec, return_fig=True, s=s, feature_thr=feature_thr, color_norm='linear'))
    # Generate separate bokeh interaction plot
    from bokeh.plotting import figure
    from bokeh.io import show
    from bokeh.models import ColumnDataSource
    import seaborn as sns
    from bokeh.layouts import Row
    # output_notebook()
    # feature='NEUTR_base'
    # exposure_var = 'CTR1'
    def generate_interaction_chart(analysis, feature, exposure_var):
        tmpDF = pd.DataFrame({exposure_var+'_value': analysis.df[exposure_var],
            feature+'_value':analysis.df[feature],
            exposure_var+'_shap_value':analysis.shapDF_prob[exposure_var],
            'interaction_effect': analysis.shap_interaction_values[:,analysis.mdlFeatures.index(exposure_var), analysis.mdlFeatures.index(feature)]})
        def generate_sequential_color_d(x):
            nLevels = len(x.unique())
            sorted_values = np.unique(list(x.sort_values().reset_index(drop=True).to_dict().values()))
            colors=sns.palettes.color_palette(palette='RdYlBu_r', n_colors=nLevels).as_hex()
            color_d = dict(zip(sorted_values, colors))
            return color_d
        #     import pdb; pdb.set_trace()
        color_d = generate_sequential_color_d(tmpDF[feature+"_value"])
        tmpDF[feature+"_color"] = tmpDF[feature+"_value"].replace(color_d)
        color_d = generate_sequential_color_d(tmpDF[exposure_var+"_value"])
        tmpDF[exposure_var+"_color"] = tmpDF[exposure_var+"_value"].replace(color_d)
        src=ColumnDataSource(tmpDF)
        fig1 = figure(tools=['crosshair', 'lasso_select'], 
                 plot_width=500, plot_height=500, title=f'ER relationship colored by {feature}',
                      y_axis_label='∆ probability',
                      x_axis_label=exposure_var+' (μg/mL)', 
                  x_range=[tmpDF[exposure_var+"_value"].quantile(.01)-1*tmpDF[exposure_var+"_value"].std(),
                           tmpDF[exposure_var+"_value"].quantile(.99)+3*tmpDF[exposure_var+"_value"].std()], 
                  y_range=[tmpDF[exposure_var+'_shap_value'].quantile(.01)-1*tmpDF[exposure_var+'_shap_value'].std(),tmpDF[exposure_var+'_shap_value'].quantile(.99)+2*tmpDF[exposure_var+'_shap_value'].std()])
        fig1.circle(x=exposure_var+'_value', y=exposure_var+'_shap_value', color=feature+"_color", source=src, size=5)
        fig1.toolbar.autohide=True

        fig2 = figure(tools=['crosshair', 'lasso_select'], 
                    plot_width=500, plot_height=500, title=f'{feature}-{exposure_var} interaction effect colored by {exposure_var}',
                         y_axis_label='∆ ER',
                         x_axis_label=feature, x_range=[-1,tmpDF[feature+"_value"].quantile(.99)+tmpDF[feature+"_value"].std()], 
                      y_range=[tmpDF['interaction_effect'].quantile(.01)-2*tmpDF['interaction_effect'].std(),
                      tmpDF['interaction_effect'].quantile(.99)+3*tmpDF['interaction_effect'].std()])
        fig2.circle(x=feature+'_value', y='interaction_effect', color=exposure_var+"_color", source=src, size=5)
        fig2.toolbar.autohide=True
        fig3 = figure(tools=['crosshair', 'lasso_select'], 
            plot_width=500, plot_height=500, title=f'{feature}-{exposure_var} interaction effect colored by {feature}',
                 y_axis_label='∆ ER',
                 x_axis_label=exposure_var, x_range=[-1,tmpDF[exposure_var+"_value"].quantile(.99)+tmpDF[exposure_var+"_value"].std()], 
              y_range=[tmpDF['interaction_effect'].quantile(.01)-3*tmpDF['interaction_effect'].std(),tmpDF['interaction_effect'].quantile(.99)+2*tmpDF['interaction_effect'].std()])
        fig3.circle(x=exposure_var+'_value', y='interaction_effect', color=feature+"_color", source=src, size=5)
        fig3.toolbar.autohide=True
        return fig1,fig2, fig3
    generate_bokeh_interaction_plot = st.checkbox('Generate interaction plot (bokeh)')
    if generate_bokeh_interaction_plot:
        fig1,fig2,fig3 = generate_interaction_chart(analysis, feature, exposure_var)
        from bokeh.io import output_file
        output_file(analysis.outputs_dir+f'{exposure_var}-{feature} interaction.html')
        show(Row(fig1,fig2, fig3))

        # 
        # st.bokeh_chart(fig1)
    # with col1E:
    #     st.write(fig2)

elif section == 'Generate (semi)synthetic data': 
    from shapml.utils.simulations import generate_synthetic_binary_classification_df
    from shapml.utils.superclasses import analysis as analysis_class
    col1, col2, col3 = st.columns(3)
    with col1:
        dataset_type = st.selectbox(label='Data type', options=['Original data', 'Synthetic data'], index=0)
        exposure_var = st.text_input(label = 'exposure var', value=exposure_var, help="Alternatively: synth_exposure")
    with col2:
        random_state=st.number_input('random_state', step=1, value=0, help='Sets random state for both synthesizing data as well as synthesizing outcomes')
        random_state=int(random_state)
    with col3:
        desired_rows=st.number_input('desired # of rows', step=1, value=500 if int(analysis.df.shape[0])> 500 else int(analysis.df.shape[0]), help=f'original dataset had {int(analysis.df.shape[0])} rows')
        if (dataset_type == 'Original data') and (desired_rows > analysis.df.shape[0]):
            st.write(f"More rows selected than actual dataset possess. Output will have {analysis.df.shape[0]} rows")
            desired_rows = int(analysis.df.shape[0])
            st.write(desired_rows)

    # if dataset_type == 'Synthetic data':
    model_recov_analysis_files = glob.glob(outputs_dir + target+ f"_model_recovery*.p")
    if len(model_recov_analysis_files) > 0: 
        model_recov_analysis_files.insert(0, 'None')
        most_recent_file = max(glob.glob(outputs_dir + target+ f"_model_recovery*.p"), key=os.path.getctime)
        fileName = st.selectbox(label='Load previously generated (semi)synthetic data:', options = model_recov_analysis_files, 
            index=model_recov_analysis_files.index(most_recent_file))
        # if st.button("Load the previously saved data"):
        if fileName != 'None':
            prev_out = pickle.load(open(fileName, 'rb'))
            st.write("Loaded: ", most_recent_file)
            st.write("Fit on previous (semi)synthetic data: ", prev_out['LR_analysisDF'])
            @st.cache(allow_output_mutation=True)
            def get_default_terms(outputs_dir, target):
                return prev_out['mdl_terms']
            default_terms = get_default_terms(outputs_dir, target)
        else:
            @st.cache(allow_output_mutation=True)
            def get_default_terms(outputs_dir, target):
                default_terms=copy.deepcopy(analysis.mdlFeatures)
                return default_terms
            default_terms = get_default_terms(outputs_dir, target)
            prev_out={}
    else:
        prev_out={}
        @st.cache(allow_output_mutation=True)
        def get_default_terms(outputs_dir, target):
            return analysis.mdlFeatures
        default_terms = get_default_terms(outputs_dir, target)

    if 'meta_df' in analysis.__dir__():
        original_df = pd.concat([analysis.df, analysis.meta_df], axis =1)
    else:
        original_df = analysis.df.copy()
    additional_code_sec = st.expander('Additional code (e.g. to augment original_df)')
    with additional_code_sec:
        
        # col1, col2 = st.columns(2)
        # with col1:
        default_additional_code = """
# Effect of confounder on response would need to be manually programmed later:
np.random.seed(random_state)
original_df['synth_confounder'] = np.random.randn(original_df.shape[0])
original_df['synth_pk_cov'] = np.random.randn(original_df.shape[0])
original_df['synth_acausal_exp_interaction'] = -.666*original_df['synth_confounder'] + .333*np.random.randn(original_df.shape[0])
synth_exposure_zscore = .4*original_df['synth_confounder'] + .4*original_df['synth_pk_cov'] + .2*np.random.randn(original_df.shape[0])
from sklearn.preprocessing import quantile_transform # exposure is in domain [0,1]
original_df['synth_exposure'] = quantile_transform(synth_exposure_zscore.values.reshape(-1,1)).ravel()
synth_trt_flag = np.random.binomial(n=1, p=.5, size=original_df.shape[0])
# An interaction effect with synthetic_exposure would need to be manually programmed:
original_df['synth_effect_modifier'] = np.random.randn(original_df.shape[0])
original_df['synth_exposure'] = original_df['synth_exposure']*synth_trt_flag
#default_terms.extend(['synth_pk_cov', 'synth_confounder', 'synth_exposure', 'synth_acausal_exp_interaction', 'synth_exposure:synth_effect_modifier', 'synth_exposure:synth_acausal_exp_interaction', 'synth_effect_modifier'])
#default_terms=list(set(default_terms))  
st.write(original_df)
"""
        additional_code = st_ace(value=default_additional_code, language='python', theme='cobalt')
        # with col2:
        # st.write(additional_code)
        try:
            text_file = open(outputs_dir + "additional_code.py", "w")
            n = text_file.write(additional_code)
            text_file.close()
            exec(open(outputs_dir + "additional_code.py").read())
        except:
            st.write('Error in executing additional code')
            pass
        # eval(additional_code)

    if True: # This is applicable for both dataset types
        st.subheader("Select model terms")
        
        potential_mdl_terms = genrate_potential_mdl_terms(original_df, target)
        mdl_terms = st.multiselect(options=potential_mdl_terms, default=default_terms, label='Model terms')
        split_var_list = [col.split(':') for col in mdl_terms]
        primary_mdl_terms = list(set(list(itertools.chain(*list(split_var_list)))))
        terms_automatically_included = set(primary_mdl_terms).difference(mdl_terms)
        if len(terms_automatically_included)>0:
            st.write(terms_automatically_included, ' are not currenly included in the model, but will be automatically included')
            mdl_terms.extend(primary_mdl_terms)

        realLR = BC.logistic_shap(df=original_df, target=target) # This is to show the outputs of real LR
        formula_like = target + " ~ " + " + ".join(mdl_terms)
        out = realLR.summary_stats(f=formula_like)
        real_outDF = pd.DataFrame(out.tables[1].data[1:], columns = out.tables[1].data[0]).set_index('').astype(float)
        st.write("For reference, the LR analysis on real data:", real_outDF)

        mdl_vars = copy.deepcopy(primary_mdl_terms)
        mdl_vars.append(target)
        realDF = original_df[mdl_vars].copy()
        if dataset_type == 'Synthetic data':
            st.subheader("Synthesize data")
            col1, col2, col3 = st.columns(3)
            with col1:
                oversampler_name=st.selectbox(options=['OUPS', 'NT_SMOTE', 'ROSE', 'NDO_sampling', 'Borderline_SMOTE1', 'SMOTE', 'Borderline_SMOTE2', 'SMOTE_OUT', 'SN_SMOTE', 'Selected_SMOTE', 'distance_SMOTE', 'Gaussian_SMOTE', 'Random_SMOTE', 'SL_graph_SMOTE', 'CURE_SMOTE'],
                index=3, label='Synthetic method')
            with col2: 
                examine_synthetic_data = st.selectbox(options=['Yes', 'No'], label='Examine synthetic data?', index=1)
                
            
            @st.cache(hash_funcs={pd.DataFrame: lambda _: None}, suppress_st_warning=True)
            def st_generate_synthetic_binary_classification_df(df, target, mdl_vars, oversampler_name=oversampler_name, desired_rows=2000, verbose=False, random_state=0):
                """ mdl_vars is only used for hashing
                """
                print("Generating synthetic data", time.asctime())
                return generate_synthetic_binary_classification_df(df, target, oversampler_name=oversampler_name, desired_rows=desired_rows, verbose=False, random_state=random_state)
            if 'exposure_var' in locals(): # It should be explicitly defined whether there is an exposure variable or not.
                q_placebo = len(realDF[realDF[exposure_var]==0])/realDF.shape[0]
                syntheticDF = st_generate_synthetic_binary_classification_df(realDF[realDF[exposure_var]>0], target, mdl_vars, desired_rows=int(desired_rows))
                selection_vec=syntheticDF.sample(frac=q_placebo, random_state=random_state).index
                syntheticDF[exposure_var].loc[selection_vec] = 0
                syntheticDF['Treatment_Flag'] = 1
                syntheticDF['Treatment_Flag'].loc[selection_vec] = 0
                st.write("Added in 'Treatment_Flag' and matched placebo proportions from actual data")
            else:
                syntheticDF = st_generate_synthetic_binary_classification_df(realDF, target, mdl_vars, desired_rows=int(desired_rows))
            
            if examine_synthetic_data=='Yes':
                col1, col2 =st.columns(2)
                with col1:
                    st.write('Real outcomes:')        
                    st.write(f"{np.round(original_df[target].mean(),2)*100}% in positive class")
                    fig=plt.figure()
                    sns.heatmap(original_df[mdl_vars].corr(), cmap='coolwarm',vmin=-.5, vmax=1)
                    st.write(fig)
                with col2:
                    st.write('Synthetic outcomes:')
                    st.write(f"{np.round(syntheticDF[target].mean(),2)*100}% in positive class")
                    # synth_a=analysis_class(df=syntheticDF, target=target)
                    fig=plt.figure()
                    sns.heatmap(syntheticDF.corr(), cmap='coolwarm',vmin=-.5, vmax=1)
                    st.write(fig)
                mdl_vars.insert(0, 'Treatment_Flag')
                # mdl_vars.insert(0, 'Treatment_Flag')
                col1, col2 =st.columns(2)
                with col1:
                    comparator = st.selectbox(options=mdl_vars, index=0, label='Compare distributions for:')
                    if comparator=='None':
                        pass
                    else:
                        realDF_copy =realDF.copy()
                        syntheticDF_copy = syntheticDF.copy()
                        realDF_copy['Synthetic_data'] = False
                        syntheticDF_copy['Synthetic_data'] = True
                        comparison_df = pd.concat([realDF_copy, syntheticDF_copy])
                        fig,ax = plt.subplots(1,1)
                        sns.histplot(x=comparator, hue='Synthetic_data', data=comparison_df, ax=ax)
                        st.write(fig)
                with col2: 
                    st.write(syntheticDF)
        

    st.subheader("Synthesize outputs")
    # primary_mdl_terms
    col1, col2= st.columns(2)
    with col1:
        st.write("Term coefficients")
        starting_params=real_outDF['coef'].to_dict()
        # These range are range_multiplier
        range_multiplier = st.slider('Range multiplier', 1.0, 50.0,  value = 10.0, step=1.0)
        min_params = real_outDF['[0.025'].to_dict() 
        max_params = real_outDF['0.975]'].to_dict()
        coeffs_d = copy.deepcopy(starting_params)
        if st.button('Show term coefficients on real data'):
            st.write(coeffs_d)
        for mdl_term in coeffs_d:
            coeffs_d[mdl_term] = st.slider(min_value=float(starting_params[mdl_term]-range_multiplier*(starting_params[mdl_term]-min_params[mdl_term])), 
                                    max_value=float(starting_params[mdl_term]+range_multiplier*(max_params[mdl_term]-starting_params[mdl_term])), 
                                    value = float(coeffs_d[mdl_term]),
                                    label=mdl_term, 
                                    key=mdl_term,
                                    step=float(starting_params[mdl_term]-min_params[mdl_term])/10.0)
        if dataset_type == 'Synthetic data':
            for i,t in enumerate(default_terms):
                if t not in coeffs_d:
                    default_terms.pop(i)
    with col2:
        if st.checkbox('Continuous update'):
            st.write("Synthesized outcomes using ", coeffs_d)
            if dataset_type=='Synthetic data':
                df_for_LR=generate_df_for_LR_synthetic(syntheticDF, target, oversampler_name, outputs_dir, random_state, desired_rows, formula_like=formula_like, mdl_terms=None)
                df_for_LR, logit_df = synthesize_outcomes(df_for_LR, coeffs_d, random_state=random_state)
            elif dataset_type =='Original data':
                df_for_LR=generate_df_for_LR_real(original_df.sample(n=int(desired_rows), random_state=random_state), target, outputs_dir, desired_rows, formula_like=formula_like, mdl_terms=None)
                df_for_LR, logit_df = synthesize_outcomes(df_for_LR, coeffs_d, random_state=random_state)
            # st.write(df_for_LR)
            prob_plot_x_feature = st.selectbox(label='Select feature: ', options=primary_mdl_terms, index = primary_mdl_terms.index(exposure_var) if exposure_var in primary_mdl_terms else 0)
            
            stratify_by_list = copy.deepcopy(primary_mdl_terms)
            stratify_by_list.insert(0, 'None')
            stratify_by = st.selectbox(label='Stratify by: ', options=stratify_by_list, index = 0)

            if stratify_by == 'None':
                stratify_by = None
            q=st.slider(label='quantiles', min_value=2, max_value=20, value=4)
            
            fig = stratified_prob_plot(x=prob_plot_x_feature, 
                y='synthetic_target', q=q, df=df_for_LR, stratify=stratify_by, ylims=[0,1], 
                hl=df_for_LR['synthetic_target'].mean())
            plt.grid(alpha=.5)
            plt.yticks(np.linspace(0,1,11))
            st.write(fig)

            st.write("Real empirical relationship:")
            fig = stratified_prob_plot(x=prob_plot_x_feature, 
                y=target, q=q, df=original_df, stratify=stratify_by, ylims=[0,1], 
                hl=original_df[target].mean())
            plt.grid(alpha=.5)
            plt.yticks(np.linspace(0,1,11))
            st.write(fig)
        

    if st.button(label='Synthesize outputs'):
        if dataset_type=='Synthetic data':
            df_for_LR=generate_df_for_LR_synthetic(syntheticDF, target, oversampler_name, outputs_dir, random_state, desired_rows, formula_like=formula_like, mdl_terms=None)
            df_for_LR, logit_df = synthesize_outcomes(df_for_LR, coeffs_d, random_state=random_state)
            if exposure_var in df_for_LR.columns: 
                q_placebo = len(realDF[realDF[exposure_var]==0])/realDF.shape[0]
                selection_vec=df_for_LR.sample(frac=q_placebo, random_state=random_state).index
                df_for_LR[exposure_var].loc[selection_vec] = 0
                df_for_LR['Treatment_Flag'] = 1
                df_for_LR['Treatment_Flag'].loc[selection_vec] = 0
                st.write("Added in 'Treatment_Flag' and matched placebo proportions from actual data")
            else:
                st.write(exposure_var, ' was not found in list of model terms')
        elif dataset_type =='Original data':
            df_for_LR=generate_df_for_LR_real(original_df.sample(n=desired_rows, random_state=random_state), target, outputs_dir, desired_rows, formula_like=formula_like, mdl_terms=None)
            
            df_for_LR, logit_df = synthesize_outcomes(df_for_LR, coeffs_d, random_state=random_state)            
            if exposure_var in df_for_LR.columns:
                df_for_LR['Treatment_Flag'] = (df_for_LR[exposure_var]!=0).astype(int)
            else:
                st.write(exposure_var, ' was not found in list of model terms')
        
        semiSyntheticDF=df_for_LR.copy()
        semiSyntheticDF[target]=semiSyntheticDF['synthetic_target']
        semiSyntheticDF.drop(columns=['Intercept', 'synthetic_probability','synthetic_target'], inplace=True)
        st.write(semiSyntheticDF)
        st.markdown(get_table_download_link(df_for_LR), unsafe_allow_html=True)
        st.write(df_for_LR)
        
        with col2: 
            real_synthHist_fig,(ax1, ax2)=plt.subplots(1,2, figsize=(12,4))
            ax1.hist(semiSyntheticDF[target], label='Synthetic data', alpha=.5)
            ax1.set_title('(Semi)synthetic data:\n{:.2f}% positive class'.format(semiSyntheticDF[target].mean()*100))
            ax2.hist(realDF[target], label='real data', alpha=.5)
            ax2.set_title('Real data:\n{:.2f}% positive class'.format(realDF[target].mean()*100))
            ax1.set_xlabel(target); ax2.set_xlabel(target)
            ax1.set_ylabel('# of patients'); ax2.set_ylabel('# of patients')
            plt.tight_layout()
            st.write(real_synthHist_fig)
            synthLR = BC.logistic_shap(df=semiSyntheticDF, target=target) # This is to show fit to (semi)synthetic data
            formula_like = target + " ~ " + " + ".join(mdl_terms)
            out = synthLR.summary_stats(f=formula_like)
            synth_outDF = pd.DataFrame(out.tables[1].data[1:], columns = out.tables[1].data[0]).set_index('').astype(float)
            st.write("LR analysis on (semi)synthetic data:", synth_outDF)

    terms2remove = st.multiselect(label='Columns to remove from model recovery analysis: ', options=primary_mdl_terms)
    # st.write(terms2remove) # TODO: implement this        
    if st.button(label='Synthesize/save outputs'):
        st.write('Generating synthetic data...')
        if dataset_type=='Synthetic data':
            df_for_LR=generate_df_for_LR_synthetic(syntheticDF, target, oversampler_name, outputs_dir, random_state, desired_rows, formula_like=formula_like, mdl_terms=None)
            df_for_LR, logit_df = synthesize_outcomes(df_for_LR, coeffs_d, random_state=random_state)
            if exposure_var in df_for_LR.columns: 
                q_placebo = len(realDF[realDF[exposure_var]==0])/realDF.shape[0]
                selection_vec=df_for_LR.sample(frac=q_placebo, random_state=random_state).index
                df_for_LR[exposure_var].loc[selection_vec] = 0
                df_for_LR['Treatment_Flag'] = 1
                df_for_LR['Treatment_Flag'].loc[selection_vec] = 0
                st.write("Added in 'Treatment_Flag' and matched placebo proportions from actual data")
                primary_mdl_terms.append('Treatment_Flag')
            else:
                st.write(exposure_var, ' was not found in list of model terms')
        elif dataset_type =='Original data':
            df_for_LR=generate_df_for_LR_real(original_df.sample(n=desired_rows, random_state=random_state), target, outputs_dir, desired_rows, formula_like=formula_like, mdl_terms=None)
            
            df_for_LR, logit_df = synthesize_outcomes(df_for_LR, coeffs_d, random_state=random_state)
            df_for_LR['Treatment_Flag'] = (df_for_LR[exposure_var]!=0).astype(int)
            primary_mdl_terms.append('Treatment_Flag')
        semiSyntheticDF=df_for_LR.copy()
        semiSyntheticDF[target]=semiSyntheticDF['synthetic_target']
        semiSyntheticDF.drop(columns=['Intercept', 'synthetic_probability','synthetic_target'], inplace=True)
        st.markdown(get_table_download_link(df_for_LR), unsafe_allow_html=True)
        st.write(df_for_LR)
        with col2: 
            real_synthHist_fig,(ax1, ax2)=plt.subplots(1,2, figsize=(12,4))
            ax1.hist(semiSyntheticDF[target], label='Synthetic data', alpha=.5)
            ax1.set_title('(Semi)synthetic data:\n{:.2f}% positive class'.format(semiSyntheticDF[target].mean()*100))
            ax2.hist(realDF[target], label='real data', alpha=.5)
            ax2.set_title('Real data:\n{:.2f}% positive class'.format(realDF[target].mean()*100))
            ax1.set_xlabel(target); ax2.set_xlabel(target)
            ax1.set_ylabel('# of patients'); ax2.set_ylabel('# of patients')
            plt.tight_layout()
            st.write(real_synthHist_fig)
            synthLR = BC.logistic_shap(df=semiSyntheticDF, target=target) # This is to show the outputs of real LR
            formula_like = target + " ~ " + " + ".join(mdl_terms)
            out = synthLR.summary_stats(f=formula_like)
            synth_outDF = pd.DataFrame(out.tables[1].data[1:], columns = out.tables[1].data[0]).set_index('').astype(float)
            st.write("LR analysis on (semi)synthetic data:", synth_outDF)
        st.write("Ground truth coefficients: ", coeffs_d)
        mdl_vars = copy.deepcopy(primary_mdl_terms)
        mdl_vars.append(target)
        for t in terms2remove:
            mdl_vars.pop(mdl_vars.index(t))
        st.write('Performing model recovery analysis...')
        model_recov = BC.xgb_shap(df=semiSyntheticDF[mdl_vars], target=target, generate_interaction_vals=True, max_evals=25)    
        st.write('Tuning model...')
        model_recov.tune_model()
        st.write('Generating shap_values')
        model_recov.shapDF_prob
        st.write(model_recov.shap_summary_plots(show=False))
        lean_copy = remove_explainers_models(model_recov)
        outputs2save_d = dict(
            model_recov = lean_copy,
            df_for_LR=df_for_LR, 
            semiSyntheticDF=semiSyntheticDF, 
            logit_df=logit_df, 
            coeffs_d=coeffs_d, 
            LR_analysisDF = synth_outDF, 
            mdl_terms=mdl_terms)
        name="model_recovery"
        fileName = outputs_dir + "_".join([target, name, dataset_type, str(random_state), str(desired_rows), time.asctime()]) +".p"
        pickle.dump(outputs2save_d, open(fileName, "wb"))
        st.write("Outputs saved: ", fileName)
elif section == 'Model recovery analysis':
    try:
        model_recov_analysis_files = glob.glob(outputs_dir + target+ f"_model_recovery*.p")
        most_recent_file = max(glob.glob(outputs_dir + target+ f"_model_recovery*.p"), key=os.path.getctime)
        fileName = st.selectbox(label='Load semi-synthetic data', options = model_recov_analysis_files, index=model_recov_analysis_files.index(most_recent_file))
        out = pickle.load(open(fileName, 'rb'))
        st.write("Dataset shape: ",  out['semiSyntheticDF'].shape, out['semiSyntheticDF'])
        st.write("Fit on semi-synthetic data: ", out['LR_analysisDF'])
        st.write("Ground truth coefficients: ", out['coeffs_d'])
        model_recov = out['model_recov']
    except:
        st.write("Model recovery analysis must be setup first")
        Model_recovery_analysis_must_be_setup_first 
    

    st.subheader('SHAP feature dependnce plot and interaction analysis')
    col1, col2 = st.columns(2)
    with col2:
        potential_interaction_features = copy.deepcopy(model_recov.mdlFeatures)
        potential_interaction_features.insert(0, 'auto')
        interaction_feature = st.selectbox(label='interaction feature', options=potential_interaction_features, index=0)
    with col1:
        mdlFeatures = copy.deepcopy(model_recov.mdlFeatures)
        mdlFeatures.insert(0, 'None')
        primary_feature = st.selectbox(label='feature', options=mdlFeatures)

        if primary_feature != 'None':
            fig,ax = plt.subplots(1,1)
            model_recov.dependence_plot(feature=primary_feature, interaction_feature=interaction_feature, ax=ax, show=False)
            st.write(fig)

    with col2:
        primary_feature_thr_vals = st.slider(min_value=float(model_recov.df[primary_feature].min()), max_value=float(model_recov.df[primary_feature].max()), value=(float(model_recov.df[primary_feature].min()), float(model_recov.df[primary_feature].max())), label=f'min/max {primary_feature}')
        feature_thr=st.number_input('threshold', step=1.0, value=np.round(float(model_recov.df[interaction_feature].median()), 2))
        st.write(shap_interaction_plot(model_recov, interaction_feature, exposure_thr_vals=primary_feature_thr_vals, feature_thr=feature_thr, exposure_var=primary_feature, return_fig=True))
    if st.button('Show SHAP summary analyses'):
        col1, col2 = st.columns(2)
        with col1:
            st.write(model_recov.shap_summary_plots(show=False))
        with col2:
            st.write(model_recov.shap_interaction_summary(feature=primary_feature, show=False))
        # st.write(model_recov.shap_interaction_summary(feature=exposure_var))
    
    df_st=model_recov.df.copy()
    col1, col2 = st.columns(2)
    with col1:
        plot_type = st.selectbox('Interaction plot type', ['Estimated Tx effect', 'Empirical probability', 'Empirical probability (empirical exposure)','Estimated Tx effect (empirical exposure)'], 2)
        pass
        # x_feature = st.selectbox('feature (x-axis) (ordered by feature importance)', model_recov.orderedFeatures)
    with col2: 
        Treatment_Flag= st.text_input(label='Treatment_Flag', value='Treatment_Flag')
        pass
        # thr=st.number_input('threshold', step=1.0, value=np.round(df_st[x_feature].median(), 2), key=2)
        

    # st.write(exposure_var)
    if plot_type == 'Empirical probability (empirical exposure)':
        interaction_analysis = BC.logistic_shap(df=df_st, target=target)
        out = interaction_analysis.summary_stats(f=f"{target}~{primary_feature} + {interaction_feature} + {primary_feature}:{interaction_feature}")
        lr_output = pd.DataFrame(out.tables[1].data[1:], columns = out.tables[1].data[0]).set_index('').astype(float)
        st.write(lr_output)
    st.write(cutoff_plot_st(analysis=model_recov, shap_features=primary_feature, df_st=df_st, prediction_time='Semisynthetic', exposure_var=primary_feature, Treatment_Flag=Treatment_Flag, valid_idxs=list(range(model_recov.df.shape[0])), outputs_dir=outputs_dir, x_feature=interaction_feature, thr=feature_thr, plot_type=plot_type, bootstrap_analysis=False))



