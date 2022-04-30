from bokeh.plotting import figure
from bokeh.io import show, output_notebook, push_notebook
from bokeh.models import LassoSelectTool, BoxSelectTool, HoverTool
from bokeh.models import LogColorMapper, ColorBar, ColumnDataSource, LabelSet, LogTicker
from bokeh.palettes import RdYlBu10 as palette
from bokeh.palettes import all_palettes
from bokeh.models import ColumnDataSource
from bokeh.models import Slider, CustomJS
from bokeh.layouts import Column, Row
import numpy as np
import pandas as pd
import seaborn as sns

legend_theme=dict(click_policy = 'hide', border_line_alpha = 0,label_text_font_size = '12px', background_fill_alpha = 0)
fig_theme = dict(background_fill_color = "gray", background_fill_alpha = 0.3)
axis_theme = dict(axis_label_text_font_size="20px", major_label_text_font_size = "14px")
title_theme = dict(text_font_size="20px")

def getBinaryClassDF(y_true, y_predict_proba, bin_size = 0.01):
    np.seterr(all='ignore')
    """ getBinaryClassDF(y_true = y_test, y_predict_proba = mdl.predict_proba(X_test)[:,1])
    """
    def predict_pos(thr):
        return y_predict_proba >= thr
    def predict_neg(thr):
        return y_predict_proba < thr

    y_true = y_true.astype(bool)
    thresholds_edges = np.arange(-bin_size,1+bin_size,bin_size)
    thresholds = thresholds_edges[1:]
    nThresholds = thresholds.shape[0]
    FPR, Precision, Recall, Specificity, F1 = [], [], [], [], []
    TP, TN, FP, FN = [], [], [], []
    for thr in thresholds:
        Precision.append( (predict_pos(thr) & y_true).sum() / predict_pos(thr).sum() )
        Recall.append( ((predict_pos(thr) & y_true).sum())/ y_true.sum() )
        F1.append(2*((Precision[-1]*Recall[-1])/(Precision[-1]+Recall[-1])))
        Specificity.append( (predict_neg(thr) & ~y_true).sum() / (~y_true).sum() )
        FPR.append( (predict_pos(thr) & ~y_true).sum() /(~y_true).sum() )
        # Confusion matrix
        TN.append((predict_neg(thr) & ~y_true).sum())
        FP.append((predict_pos(thr) & ~y_true).sum())
        FN.append((predict_neg(thr) & y_true).sum())
        TP.append((predict_pos(thr) & y_true).sum())
    CD_predict_proba = 1-np.cumsum(np.histogram(y_predict_proba, thresholds_edges, density=True)[0]*bin_size)
    actual_pos_CDF = 1-np.cumsum(np.histogram(y_predict_proba[y_true.astype(bool)], bins=thresholds_edges, density=True)[0]*bin_size)
    actual_neg_CDF = 1-np.cumsum(np.histogram(y_predict_proba[~y_true.astype(bool)], bins=thresholds_edges, density=True)[0]*bin_size)
    binary_class_df = pd.concat([pd.Series(l,name=n) for n,l in \
     zip(['Threshold','TP', 'TN', 'FP', 'FN', 'FPR', 'Precision', 'Recall', 'F1', 'Specificity', 'Cumul_dist', 'actual_pos_CDF', 'actual_neg_CDF'] , [thresholds, TP, TN, FP, FN, FPR, Precision, Recall, F1, Specificity, CD_predict_proba, actual_pos_CDF, actual_neg_CDF])],axis=1)
    binary_class_df['Precision'] = binary_class_df.apply(lambda x: 1 if x['TP'] ==0 else x['Precision'], axis=1)
    return binary_class_df

def binary_classificaiton_plots(y_true, y_predict_proba, threshold=.5):
    """ 
    Visualization of binary classification model performance on test dataset
    THR : Threshold
    Example: 
    from ipywidgets import interact
    updateThreshold = binary_classificaiton_plots(y_true=y_test, y_predict_proba=mdl.predict_proba(X_test)[:,1])
    interact(updateThreshold, threshold=(0.0, 1.0, .01));
    """
    binary_class_df = getBinaryClassDF(y_true = y_true, y_predict_proba = y_predict_proba)
    N = binary_class_df[['TP', 'TN', 'FP', 'FN']].iloc[0,:].sum()

    def createConfMatrix_df(thr=threshold):
        result_numstr = []
        colorBy = []
        results = ['FN', 'TP', 'TN', 'FP']
        FPR = []
        Recall = []
        x = [0, 1, 0, 1]
        y = [0, 0, 1, 1]
        y2 = [0, 0, 100, 100]
        threshold = [thr]*4
        x_coords = [[0,1,1,0], [1,2,2,1], [0,1,1,0], [1,2,2,1]]
        y_coords = [[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 2, 2], [1, 1, 2, 2]]
        binaryClassDF_subset = binary_class_df[binary_class_df.Threshold >= thr].head(1)
        total_subjects = binaryClassDF_subset[results].values.sum()

        for resultName in results:
            result_numstr.append(": ".join([resultName, str(binaryClassDF_subset[resultName].values[0])]))
            colorBy.append(binaryClassDF_subset[resultName].values[0]/total_subjects)
            FPR.append(binaryClassDF_subset['FPR'].values[0]) 
            Recall.append(binaryClassDF_subset['Recall'].values[0])
        conf_matDF=pd.DataFrame([results, result_numstr, x_coords, y_coords,x, y, colorBy, threshold, FPR, Recall, y2]).T
        conf_matDF.columns = ['results', 'result_numstr', 'x_coords', 'y_coords', 'x', 'y', 'colorBy', 'threshold', 'FPR', 'Recall', 'y2']
        return conf_matDF
    
    binary_class_source = ColumnDataSource(binary_class_df)
    conf_mat_source = ColumnDataSource(createConfMatrix_df(thr=threshold));
    from sklearn.metrics import auc
    ROC_AUC = auc(binary_class_df.FPR.values[::-1], binary_class_df.Recall.values[::-1])
    Precision_recall_AUC = auc(binary_class_df.Recall.values[::-1], binary_class_df.Precision.fillna(1).values[::-1])

    # Cumulative Distribution Plot (Proportion of subjects vs. Predicted probability) 
    CD_plot = figure(tools='crosshair, xbox_select', 
                      plot_width=500, plot_height=250, title='Cumulative distribution of subjects vs. Threshold',
                  y_axis_label='Cumulative distribution of subjects',
                  x_axis_label='Threshold',
                            x_range=[0,1],
                            y_range=[0,1])
    # Threshold line: 
    CD_plot.line(x="threshold", y="y", source=conf_mat_source, line_color='black', line_dash="dotted")
    Pos_class_line = CD_plot.line(x="Threshold", y='actual_pos_CDF', source = binary_class_source, line_color='red', legend_label='Pos_class')
    Neg_class_line = CD_plot.line(x='Threshold', y='actual_neg_CDF', source = binary_class_source, line_color='green', legend_label='Neg_class')
    F1_line = CD_plot.line(x='Threshold', y='F1', source = binary_class_source, line_color='black', legend_label='F1_score')
    F1_line.visible=False
    CD_plot_circ = CD_plot.circle(x="Threshold", y="Cumul_dist",radius=.0001, source=binary_class_source)
    CD_plot.add_tools(HoverTool(renderers=[Pos_class_line], 
                                tooltips = [("Threshold", "@Threshold{(0.00)}"),
                                            ("Propotion of positive class", "@actual_pos_CDF{(0.00)}"),
                                            ("True Positive", "@TP{(0)}"),
                                            ("False Negative", "@FN{(0)}")], 
                                mode='vline'),
                     HoverTool(renderers=[Neg_class_line], 
                                tooltips = [("Threshold", "@Threshold{(0.00)}"),
                                            ("Propotion of Negative class", "@actual_neg_CDF{(0.00)}"),
                                            ("False Positive", "@FP{(0)}"),
                                            ("True Negative", "@TN{(0)}")], 
                                mode='vline', muted_policy = 'ignore'),
                     HoverTool(renderers=[F1_line], 
                                tooltips = [("F1 socre", "@F1{0.00}")], 
                                mode='vline'))
    CD_plot.legend.__dict__['_property_values'].update(legend_theme)
    CD_plot.legend.click_policy = "hide"
    CD_plot.legend.location = "top_right"
    CD_plot.toolbar.autohide=True
    CD_plot.toolbar.active_inspect = CD_plot.tools[2]


    # ROC 
    ROC_plot = figure(tools='crosshair, xbox_select', 
                      plot_width=250, plot_height=250, title='ROC curve',
                  y_axis_label='Sensitivity (Recall)',
                  x_axis_label='False positive rate (1-specificity)',
                            x_range=[0,1],
                            y_range=[0,1])

    ROC_plot.line(x=[0,1], y=[0,1], line_color='black', line_dash="dotted")
    ROC_line = ROC_plot.line(x="FPR", y="Recall",source=binary_class_source, legend_label=" AUC: {:.3f}".format(ROC_AUC))
    ROC_circ = ROC_plot.circle(x="FPR", y="Recall",radius=.0001, source=binary_class_source)

    # Threshold line: 
    ROC_plot.line(x="FPR", y="y", source=conf_mat_source, line_color='black', line_dash="dotted")
    ROC_plot.add_tools(HoverTool(renderers=[ROC_line], tooltips = [("Threshold", "@Threshold{(0.00)}"),
                                                                   ("Sensitivity", "@Recall{(0.00)}"),
                                                                   ("False positive rate", "@FPR{(0.00)}")], 
                                 mode='vline'))

    ROC_plot.legend.click_policy = "hide"
    ROC_plot.legend.location = "bottom_right"
    ROC_plot.toolbar_location = None
    ROC_plot.legend.__dict__['_property_values'].update(legend_theme)
    # Precision-Recall Curve 
    PrecRecall_plot = figure(tools='crosshair, xbox_select', 
                      plot_width=250, plot_height=250, title='Precision Recall Curve',
                  y_axis_label='Precision',
                  x_axis_label='Recall',
                            x_range=[0,1],
                            y_range=[0,1])
    PrecRecall_line = PrecRecall_plot.line(x="Recall", y="Precision",
                         legend_label=" AUC: {:.3f}".format(Precision_recall_AUC), source=binary_class_source)
    PrecRecall_circ = PrecRecall_plot.circle(x="Recall", y="Precision",
                                             radius=.0001, source=binary_class_source)
    # Threshold line: 
    PrecRecall_plot.line(x="Recall", y="y", source=conf_mat_source, line_color='black', line_dash="dotted")
    # PrecRecall_plot.add_tools(BoxSelectTool(renderers=[PrecRecall_circ]))

    PrecRecall_plot.add_tools(HoverTool(renderers = [PrecRecall_line],
                                tooltips=[("Threshold", "@Threshold{(0.00)}"),
                               ("Recall", "@Recall{(0.00)}"),
                               ("Precision", "@Precision{(0.00)}")], 
                                        mode = 'vline'))
    PrecRecall_plot.legend.click_policy = "hide"
    PrecRecall_plot.toolbar_location = None
    PrecRecall_plot.legend.__dict__['_property_values'].update(legend_theme)


    # Classification matrix
    color_mapper = LogColorMapper(palette=palette, low=.001, high=1)

    conf_mat_plot = figure(plot_width=250, plot_height=300, title="Confusion Matrix", 
                          x_axis_label="Predicted",
                          y_axis_label="Actual",
                          tools='')
    conf_mat_plot.toolbar_location=None
    conf_mat_plot.patches(xs='x_coords', ys='y_coords',
                         fill_color={'field': 'colorBy', 'transform': color_mapper},
                          source=conf_mat_source,
                          fill_alpha=0.7, line_color="white", line_width=0.5)
    labels = LabelSet(x='x', y='y', text='result_numstr',
                  x_offset=15, y_offset=20, source=conf_mat_source)

    confMatrix_color_bar = ColorBar(color_mapper=color_mapper, ticker=LogTicker(),
                            label_standoff=12, border_line_color=None, orientation='horizontal', 
                            location=(0, 0), title='Proportion of observations')#,
    #                                major_label_overrides=tick_labels)
    confMatrix_color_bar.formatter.use_scientific=False

    conf_mat_plot.add_layout(confMatrix_color_bar, 'below')
    conf_mat_plot.add_layout(labels)
    tick_label_overrides_x = {'0': '', '0.5': 'Negative', '1.5': 'Positive',
                   '2': '', '1':''}
    tick_label_overrides_y = {'0': '', '0.5': 'Positive', '1.5': 'Negative',
                   '2': '', '1':''}
    conf_mat_plot.xaxis.major_label_overrides = tick_label_overrides_x
    conf_mat_plot.yaxis.major_label_overrides = tick_label_overrides_y
    conf_mat_plot.yaxis.major_label_orientation = "vertical"
    conf_mat_plot.xaxis.minor_tick_line_color = None
    conf_mat_plot.yaxis.minor_tick_line_color = None

    conf_mat_plot.xaxis.major_tick_line_color = None
    conf_mat_plot.yaxis.major_tick_line_color = None
    conf_mat_plot.xgrid.grid_line_color=None
    conf_mat_plot.ygrid.grid_line_color=None
    
    
    ### Predicted probabilities: 
    def histogram(y_predict, bins=20):
        """ 
        show(make_plot(tmp.predictionsCV_SHAP))
        """
        hist,edges = np.histogram(y_predict, bins=50)
        p = figure(title='Predicted probabilities', tools='', background_fill_color="#fafafa", plot_width=300, plot_height=250)
        p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
               fill_color="navy", line_color="white", alpha=0.5)
        p.y_range.start = 0
        p.xaxis.axis_label = 'Predicted probability'
        p.yaxis.axis_label = '# of patients'
        p.grid.grid_line_color="white"
        p.toolbar_location = None
        return p

    predProb_histogram = histogram(y_predict_proba)

    # Threshold line: 
    predProb_histogram.line(x="threshold", y="y2", source=conf_mat_source, line_color='black', line_dash="dotted")
    predProb_histogram.toolbar.autohide=True

    layout = Row(Column(CD_plot,Row(ROC_plot, PrecRecall_plot)), Column(conf_mat_plot, predProb_histogram))

    return layout#updateConfMatrix_plot

from bokeh.plotting import figure
from bokeh.io import show
import seaborn as sns
import numpy as np 
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

def getCV_ROC_curve_bokehData(self, n_folds=5, n_reps=5, selection_vec=None, verbose=True): 
    """
    selection_vec can be supplied to do stratification, None (default) indicates no stratification
    """
    if type(selection_vec) == type(None):
        selection_vec = np.bool_(np.ones(self.y.shape))
    tprs = []
    aucs = []
    mean_fpr = np.arange(0,1.01,.01)
    
    _, y_pred_mat = self.model_performance(n_folds=n_folds, n_reps=n_reps, verbose=verbose, seed=0, return_predictions_matrix=True)
    
    for r in range(n_reps):
        fpr,tpr,_ = roc_curve(self.y[selection_vec], y_pred_mat[selection_vec, r])    
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc_score(self.y[selection_vec], y_pred_mat[selection_vec, r]))

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = roc_auc_score(self.y[selection_vec], np.mean(y_pred_mat,axis=1)[selection_vec])
    std_auc = np.nanstd(aucs, axis=0) # changed from sem
    std_tpr = np.nanstd(tprs, axis=0)
    # print(std_tpr)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    mean_fpr_patch = [list(np.concatenate([mean_fpr, np.flip(mean_fpr)]))]
    tprs_patch = [list(np.concatenate([tprs_upper, np.flip(tprs_lower)]))]
    mean_fpr = list(mean_fpr)
    mean_tpr = list(mean_tpr)
    return dict(mean_fpr_patch=mean_fpr_patch, tprs_patch=tprs_patch, mean_fpr=mean_fpr, mean_tpr=mean_tpr, mean_auc=mean_auc, std_auc=std_auc)


def plot_ROC(self, n_folds=5, n_reps=1, return_fig=False, verbose=False, stratify=None, thr=None, plot_width=750, plot_height=500):
    """
    Generates Cross-validated ROC curves stratified by stratification factor.
    """
    # Define plot stlyes
#     legend_theme=dict(click_policy = 'hide', border_line_alpha = 0,label_text_font_size = '9px', background_fill_alpha = 0)
    # Initiate plot:
    if 'meta_df' in self.__dir__():
        roc_df=pd.concat([self.df, self.meta_df], axis=1)
    else: 
        roc_df=self.df.copy()

    ROC_plot = figure(plot_width=plot_width, plot_height=plot_height, title='ROC curve', # Ideal slides dimensions: W: 750 H: 500    
                      tools = 'crosshair, save, reset',
                      y_axis_label='True positive rate (Sensitivity/Recall)',
                      x_axis_label='False positive rate')

    ROC_plot.__dict__['_property_values'].update(fig_theme)
    ROC_plot.xaxis.__dict__['_property_values'].update(axis_theme)
    ROC_plot.yaxis.__dict__['_property_values'].update(axis_theme)
    ROC_plot.title.text_font_size="20px"

    m = 0
    renderers = []
    if type(stratify) != type(None):
        if stratify in roc_df.columns: 
            print(f"Stratified ROC by {stratify}")
            nUnique = len(roc_df[stratify][roc_df[stratify].notna()].unique())
            if nUnique <= 4: 
                unique_values = roc_df[stratify].unique()
                colors = sns.color_palette('tab10', n_colors=nUnique).as_hex()
                for u in unique_values:
                    if type(u) != str:
                        if np.isnan(u):
                            continue
                    selection_vec = (roc_df[stratify]==u)
                    analysis_str = f"{stratify}=={u}"
                    if m==0:
                        print('Overall model performance')
                    d = getCV_ROC_curve_bokehData(self, n_folds=n_folds, n_reps=n_reps, selection_vec=selection_vec, verbose=(m==0))
                    label= analysis_str + ' Mean AUROC = {:0.3f} ± {:0.2f} (n={:.0f})'.format(d['mean_auc'], d['std_auc'], np.sum(selection_vec))
                    ROC_plot.patches(xs=d['mean_fpr_patch'], ys=d['tprs_patch'],
                            fill_color=colors[m],
                                fill_alpha=0.1, line_color="white", line_width=0.5, legend_label=label)
                    renderers.append(ROC_plot.line(x=d['mean_fpr'], y=d['mean_tpr'],line_color=colors[m], line_width=2, legend_label=label))
                    m+=1
            else: 
                colors = sns.color_palette('tab10', n_colors=2).as_hex()
                if type(thr) != type(None):
                    med_val = thr
                else:
                    med_val = np.round(roc_df[stratify].median(),2)
                for ii in [0,1]:
                    if ii ==0: 
                        selection_vec = (roc_df[stratify]<med_val)
                        comparator = "<"
                        analysis_str = f"{stratify}{comparator}{med_val}"
                    elif ii==1:
                        selection_vec = (roc_df[stratify]>=med_val)
                        comparator = "≥"
                        analysis_str = f"{stratify}{comparator}{med_val}"
                    d = getCV_ROC_curve_bokehData(self, n_folds=n_folds, n_reps=n_reps, selection_vec=selection_vec)
                    label= analysis_str + ' Mean AUROC = {:0.3f} ± {:0.2f} (n={:.0f})'.format(d['mean_auc'], d['std_auc'], np.sum(selection_vec))
                    ROC_plot.patches(xs=d['mean_fpr_patch'], ys=d['tprs_patch'],
                            fill_color=colors[m],
                                fill_alpha=0.1, line_color="white", line_width=0.5, legend_label=label)
                    renderers.append(ROC_plot.line(x=d['mean_fpr'], y=d['mean_tpr'],line_color=colors[m], line_width=2, legend_label=label))
                    m+=1
        else:
            print(f"{stratify} not in self.meta_df or self.df")
                    
    else: # No stratification
        d = getCV_ROC_curve_bokehData(self, n_folds=n_folds, n_reps=n_reps)
        label= ' Mean AUROC = {:0.3f} ± {:0.2f}'.format(d['mean_auc'], d['std_auc'])
        ROC_plot.patches(xs=d['mean_fpr_patch'], ys=d['tprs_patch'],
                fill_color='black',
                    fill_alpha=0.1, line_color="white", line_width=0.5, legend_label=label)
        renderers.append(ROC_plot.line(x=d['mean_fpr'], y=d['mean_tpr'],line_color='black', line_width=2, legend_label=label))
    for r in renderers:
        ROC_plot.add_tools(HoverTool(renderers=[r], 
                                    tooltips = [("fpr", "@x{0.00}"),("tpr", "@y{0.00}")], 
                                    mode='vline'))
    ROC_plot.line(x=[0,1], y=[0,1], line_color='black', line_dash="dotted", legend_label="Chance")
    # Finalize plot:
    ROC_plot.legend.click_policy="hide"
    ROC_plot.legend.__dict__['_property_values'].update(legend_theme)
    ROC_plot.legend.location='bottom_right'

    if return_fig:
        return ROC_plot
    else:
        show(ROC_plot)
    return