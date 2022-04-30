from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, LabelSet, CustomJS, DataTable, TableColumn, NumberFormatter
from bokeh.layouts import column, row
from bokeh.io import show, output_notebook, output_file, reset_output
import seaborn as sns
output_notebook()

import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from ..utils.misc import CustomScaler
# TODO: Refactor code so that we can do all dimensionality reduction using a single function that calls the scatterplot function

def updateOnSelectCode(source='dt_scat_src', target='dt_source', target_vars = ["Class", "Compound", "PC1", "PC2", "MK-lin", "B-lin", "HSCs"]):
    target_vars = ["'" + var+ "'" for var in target_vars]
    str1 = """
var inds = cb_obj.indices;
var d1 = {}.data;
var d2 = {}.data;\n""".format(source, target)

    str2 = "\n".join(["d2[{}] = []".format(var) for var in target_vars])
    str3 = "\n\nfor (var i = 0; i < inds.length; i++) {\n    "
    str4 = "\n    ".join(["d2[{}].push(d1[{}][inds[i]])".format(var,var) for var in target_vars])
    str5 = """\n}}
{}.change.emit();
table.change.emit();
""".format(target)
    jsCode = str1 + str2 + str3 + str4 + str5
    return jsCode

plot_height = 450
table_height = 200

def plot_pca_3d(df, features = [], biplot_coeff = 1, color_by='Cluster', n_clusters=5, nQuantiles=10, bins=None):
    """ 
    features: to use for clustering algorithm
    biplot_coeff: Magnitude determines the length to rescale eigen vectors TODO: automate this 
    """
    from sklearn.decomposition import PCA
    df=df.copy()
    df2fit = CustomScaler().fit_transform(df[features]) # StandardScaling and ignoring binary columns
    df2fit = df2fit.fillna(df2fit.median())
    pca = PCA(n_components=3).fit(df2fit)
    df['PC1'] = pca.transform(df2fit)[:,0]
    df['PC2'] = pca.transform(df2fit)[:,1]
    df['PC3'] = pca.transform(df2fit)[:,2]
    
    clusters = KMeans(n_clusters=n_clusters, random_state=1).fit_predict(df[['PC1', 'PC2']])
    df['Cluster'] = list(map(str, clusters))
    
    if color_by == 'Cluster':
        level_type='categorical'
        color_by_binned = 'Cluster'
        classes = list(df['Cluster'].unique())
    else:
        level_type='sequential'
        color_by_binned = color_by+"_binned"
        if len(list(df[color_by].unique())) <7:
            classes = list(df[color_by].unique())
            df[color_by+"_binned"] = df[color_by].astype(str)
            classes = [str(v) for v in classes]
        else:
            if type(bins) == list:
                df[color_by+"_binned"] = pd.cut(df[color_by+"_binned"], bins = bins, include_lowest=True, duplicates = 'drop')
            elif ((df[color_by] == df[color_by].min()).sum()/df[color_by].shape[0]) > .025: # If greater than 2.5% of the data is the lowest value: Make a bin for the LoQ/Zero value
                lowest_binEdge = (df[color_by].min() == df[color_by]).sum()/df[color_by].notna().sum()
                bins = np.append(np.array([0]), np.linspace(lowest_binEdge,1, nQuantiles)) 
                df[color_by+"_binned"] = pd.cut(df[color_by], bins = [df[color_by].quantile(q) for q in bins], include_lowest=True, duplicates = 'drop')
            else:
                bins=[df[color_by].quantile(q) for q in np.linspace(0,1, nQuantiles+1)]
                bins[0] = bins[0]-.1
                df[color_by+"_binned"] = pd.cut(df[color_by], bins = bins, include_lowest=True, duplicates = 'drop')
            classes = df[color_by+"_binned"].cat.categories.astype(str).to_list() # sorted list
        classes.append('nan')
        df[color_by_binned] = df[color_by_binned].astype(str)
        
    nLevels = len(classes)
    if level_type == 'sequential':
        colors=sns.palettes.color_palette(palette='RdYlBu_r', n_colors=nLevels-1).as_hex()
    elif level_type == 'categorical':
        colors=sns.palettes.color_palette(palette='tab10', n_colors=nLevels-1).as_hex()
    colors.append('#808080') # add gray for nan
    colorIdx_dict = dict(enumerate(classes))
    colorIdx_dict = defaultdict(lambda:nLevels+1, {v:k for k,v in colorIdx_dict.items()})
    df['color'] = df[color_by_binned].apply(lambda x: colors[colorIdx_dict[x]])

    pc1s = []
    pc2s = []
    pc3s = []
    for feature in features: 
        coeff = biplot_coeff
        pc1 = np.dot((np.array(features)==feature).astype(int), pca.components_[0])*coeff
        pc2 = np.dot((np.array(features)==feature).astype(int), pca.components_[1])*coeff
        pc3 = np.dot((np.array(features)==feature).astype(int), pca.components_[2])*coeff
        pc1s.append(pc1)
        pc2s.append(pc2)
        pc3s.append(pc3)
    biplot_df = pd.DataFrame(zip(features, pc1s, pc2s, pc3s), columns=['feature', 'PC1', 'PC2', 'PC3'])    
    import matplotlib.pyplot as plt
    from ipywidgets import interact
    @interact 
    def plot_3d(elev=(-90,150), azim=(0,600), zoom=(0,10), s=(2,100), coeff=(-50,70)):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        xs, ys, zs, cs=df.PC1,df.PC2, df.PC3, df.color
        ax.scatter(xs, ys, zs, marker='o', color=cs, s=s)
        for r in range(biplot_df.shape[0]):
            text, x,y,z = biplot_df.iloc[r]
            x,y,z=np.array([x,y,z])*coeff
            ax.plot([0, x], [0,y], [0,z], color='gray')
            ax.text(x,y,z, s=text)
        ax.plot([-zoom,zoom], [0,0], [0,0],color='k')
        ax.plot([0,0],[-zoom,zoom], [0,0],color='k')
        ax.plot([0,0], [0,0], [-zoom,zoom],color='k')
        ax.set_xlim([-zoom,zoom]);ax.set_zlim([-zoom,zoom]);ax.set_ylim([-zoom,zoom])
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3')
        ax.view_init(elev=elev, azim=azim)
    return plot_3d


def plot_pca(df, features = [], 
             data_label=None, table_vars = None, 
             plot_width=750, plot_height=plot_height, table_width=725, table_height=table_height,
             biplot_coeff = 20, ms=10, return_outputs=True, color_by='Cluster', n_clusters=5, nQuantiles=10, bins=None, return_fig=False):
    """ 
    features: to use for clustering algorithm
    table_vars: variables that we can look further into inspection
    biplot_coeff: Magnitude determines the length to rescale eigen vectors TODO: automate this
     
    """
    from sklearn.decomposition import PCA
    df=df.copy()
    df2fit = CustomScaler().fit_transform(df[features]) # StandardScaling and ignoring binary columns
    df2fit = df2fit.fillna(df2fit.median())
    pca = PCA(n_components=3).fit(df2fit)
    df['PC1'] = pca.transform(df2fit)[:,0]
    df['PC2'] = pca.transform(df2fit)[:,1]
    df['PC3'] = pca.transform(df2fit)[:,2]
    
    clusters = KMeans(n_clusters=n_clusters, random_state=1).fit_predict(df[['PC1', 'PC2']])
    df['Cluster'] = list(map(str, clusters))
    
    if color_by == 'Cluster':
        level_type='categorical'
        color_by_binned = 'Cluster'
        classes = list(df['Cluster'].unique())
    else:
        level_type='sequential'
        color_by_binned = color_by+"_binned"
        if len(list(df[color_by].unique())) <7:
            classes = list(df[color_by].unique())
            df[color_by+"_binned"] = df[color_by].astype(str)
            classes = [str(v) for v in classes]
        else:
            if type(bins) == list:
                df[color_by+"_binned"] = pd.cut(df[color_by+"_binned"], bins = bins, include_lowest=True, duplicates = 'drop')
            elif ((df[color_by] == df[color_by].min()).sum()/df[color_by].shape[0]) > .025: # If greater than 2.5% of the data is the lowest value: Make a bin for the LoQ/Zero value
                lowest_binEdge = (df[color_by].min() == df[color_by]).sum()/df[color_by].notna().sum()
                bins = np.append(np.array([0]), np.linspace(lowest_binEdge,1, nQuantiles)) 
                df[color_by+"_binned"] = pd.cut(df[color_by], bins = [df[color_by].quantile(q) for q in bins], include_lowest=True, duplicates = 'drop')
            else:
                bins=[df[color_by].quantile(q) for q in np.linspace(0,1, nQuantiles+1)]
                bins[0] = bins[0]-.1
                df[color_by+"_binned"] = pd.cut(df[color_by], bins = bins, include_lowest=True, duplicates = 'drop')
            classes = df[color_by+"_binned"].cat.categories.astype(str).to_list() # sorted list
        classes.append('nan')
        df[color_by_binned] = df[color_by_binned].astype(str)
        
    nLevels = len(classes)
    if level_type == 'sequential':
        colors=sns.palettes.color_palette(palette='RdYlBu_r', n_colors=nLevels-1).as_hex()
    elif level_type == 'categorical':
        colors=sns.palettes.color_palette(palette='tab10', n_colors=nLevels-1).as_hex()
    colors.append('#808080') # add gray for nan
    colorIdx_dict = dict(enumerate(classes))
    colorIdx_dict = defaultdict(lambda:nLevels+1, {v:k for k,v in colorIdx_dict.items()})
    df['color'] = df[color_by_binned].apply(lambda x: colors[colorIdx_dict[x]])
    legend_theme=dict(click_policy = 'hide', border_line_alpha = 0,label_text_font_size = '12px', background_fill_alpha = 0)
    scatter_theme=dict(alpha = .8, size=ms, line_color='black', line_width=.5, line_alpha=.5)
    
    fig = figure(plot_width=plot_width, plot_height=plot_height, tools='hover,lasso_select, pan, wheel_zoom, box_zoom, crosshair, save, reset', title='PCA',
                 x_axis_label='PC1', y_axis_label='PC2', active_drag="lasso_select")
    if data_label:
        # Create a dummy glyph to tie to data labels
        dataLabels = fig.scatter(0, 0, legend_label="data labels", color=None, visible=False)
        data_labels = []

    for cl, CL in enumerate(classes): 
        if data_label:
            pca_source = ColumnDataSource(df[df[color_by_binned]==CL])
            pca_scatter = fig.scatter(x='PC1', y='PC2', color='color', legend_label=CL,
                                  source=pca_source, **scatter_theme, visible = True)
                    # Create a dummy glyph to tie to data labels
            dL_renderer = fig.scatter(0, 0, legend_label="data labels", color=None, visible=False)
            data_labels.append(LabelSet(x='PC1', y='PC2', text=data_label,
                          x_offset=0, y_offset=0, source=pca_source, text_font_size='10px', visible=False))
            fig.add_layout(data_labels[cl])
            pca_scatter.js_on_change('visible', CustomJS(args=dict(ls=data_labels[cl], dL_renderer=dL_renderer),
                                          code="""if(dL_renderer.visible == true){
                                                      ls.visible = cb_obj.visible;
                                                  }"""))
            
            dL_renderer.js_on_change('visible', CustomJS(args=dict(ls=data_labels[cl],pca_scatter=pca_scatter),
                                          code="""if(pca_scatter.visible == true){
                                                      ls.visible = cb_obj.visible;
                                                  }"""))
        else: 
            pca_source = ColumnDataSource(df[df[color_by_binned]==CL])
            pca_scatter = fig.scatter(x='PC1', y='PC2', color='color', legend_label=CL,
                                      source=pca_source, **scatter_theme, visible = True)



    pc1s = []
    pc2s = []
    pc3s = []
    biplotLines = []
    for feature in features: 
        coeff = biplot_coeff
        pc1 = np.dot((np.array(features)==feature).astype(int), pca.components_[0])*coeff
        pc2 = np.dot((np.array(features)==feature).astype(int), pca.components_[1])*coeff
        pc3 = np.dot((np.array(features)==feature).astype(int), pca.components_[1])*coeff
        pc1s.append(pc1)
        pc2s.append(pc2)
        pc3s.append(pc3)
        biplotLines.append(fig.line([0, pc1], [0, pc2], legend_label="biplot", visible=False))
    biplot_df = pd.DataFrame(zip(features, pc1s, pc2s, pc3s), columns=['feature', 'PC1', 'PC2', 'PC3'])    
    feature_label_src = ColumnDataSource(biplot_df)
    labels = LabelSet(x='PC1', y='PC2', text='feature',
                      x_offset=0, y_offset=0, source=feature_label_src, visible=False, text_font_size='12px')
    fig.add_layout(labels) # biplot labels
    
    biplotLines[0].js_on_change('visible', CustomJS(args=dict(ls=labels),
                                                  code="ls.visible = cb_obj.visible;"))
    tooltips = [("PC1", "@PC1{0.00}"),
                             ("PC2", "@PC2{0.00}"),
                             (color_by, "@"+color_by)]
            
    if data_label:
        tooltips.append((data_label, "@"+data_label))
    fig.tools[0].tooltips = tooltips
    fig.toolbar.active_scroll=fig.tools[3]
    fig.legend.__dict__['_property_values'].update(legend_theme)
    fig.add_layout(fig.legend[0], 'right')
    if table_vars: 
        columns = [TableColumn(field=var, title=var) for var in table_vars]
        for c in columns[3:]:
            c.formatter = NumberFormatter(format="0.00")
        dt_scat_src = ColumnDataSource(df)
        dt_source = ColumnDataSource(df)
        dt_scatter = fig.scatter(x="PC1", y="PC2", source=dt_scat_src, size=0)
        data_table = DataTable(source=dt_source, columns = columns,width=table_width, height=table_height, editable=True)
        dt_scat_src.selected.js_on_change('indices', CustomJS(args=dict(dt_scat_src=dt_scat_src, dt_source=dt_source, data_table=data_table), 
                                                              code=updateOnSelectCode(target_vars=table_vars)))
        show(column(fig, data_table))
    if return_fig:
        return fig

        
    else:
        show(fig)
    if return_outputs: 
        return (df, biplot_df)
    return

def plot_tsne(df, features = [], 
             data_label=None, table_vars = None, tsne_params=dict(),
             plot_width=750, plot_height=plot_height, table_width=725, table_height=table_height, 
              n_clusters=5, ms=10, return_outputs=True, color_by='Cluster', nQuantiles=10, bins=None):
    """ 
    features: to use for clustering algorithm
    table_vars: variables that we can look further into inspection
    tsne_params :   dictionary of tSNE params: default=dict() , which yields dict(init = 'pca')
                    perplexity: float, default=30.0
                    learning_rate: float, default=200.0
                    n_iter: int, default=1000
                    init : {‘random’, ‘pca’}
                    random_state : int, default=1
                    More details: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html?highlight=tsne#sklearn.manifold.TSNE
     
    """
    from sklearn.manifold import TSNE
    df=df.copy()
    df2fit = CustomScaler().fit_transform(df[features])
    df2fit = df2fit.fillna(df2fit.median())
    tsne_params_final = dict(init = 'pca') # default params
    if len(tsne_params) == 0:
        pass
    else:
        tsne_params_final.update(tsne_params)

    mdl_outputs = TSNE(n_components=3, **tsne_params_final).fit_transform(df2fit)
    df['tSNE1'] = mdl_outputs[:,0]
    df['tSNE2'] = mdl_outputs[:,1]
    df['tSNE3'] = mdl_outputs[:,2]
    clusters = KMeans(n_clusters=n_clusters, random_state=1).fit_predict(df[['tSNE1', 'tSNE2']])
    df['Cluster'] = list(map(str, clusters))
    if color_by == 'Cluster':
        level_type='categorical'
        color_by_binned = 'Cluster'
        classes = list(df['Cluster'].unique())
    else:
        level_type='sequential'
        color_by_binned = color_by+"_binned"
        if len(list(df[color_by].unique())) <7:
            classes = list(df[color_by].unique())
            df[color_by+"_binned"] = df[color_by].astype(str)
            classes = [str(v) for v in classes]
        else:
            if type(bins) == list:
                df[color_by+"_binned"] = pd.cut(df[color_by+"_binned"], bins = bins, include_lowest=True, duplicates = 'drop')
            elif ((df[color_by] == df[color_by].min()).sum()/df[color_by].shape[0]) > .025: # If greater than 2.5% of the data is the lowest value: Make a bin for the LoQ/Zero value
                lowest_binEdge = (df[color_by].min() == df[color_by]).sum()/df[color_by].notna().sum()
                bins = np.append(np.array([0]), np.linspace(lowest_binEdge,1, nQuantiles)) 
                df[color_by+"_binned"] = pd.cut(df[color_by+"_binned"], bins = [df[color_by].quantile(q) for q in bins], include_lowest=True, duplicates = 'drop')
            else:
                bins=[df[color_by].quantile(q) for q in np.linspace(0,1, nQuantiles+1)]
                bins[0] = bins[0]-.1
                df[color_by+"_binned"] = pd.cut(df[color_by], bins = bins, include_lowest=True, duplicates = 'drop')
            classes = df[color_by+"_binned"].cat.categories.astype(str).to_list() # sorted list
        classes.append('nan')
        df[color_by_binned] = df[color_by_binned].astype(str)
        
    nLevels = len(classes)
    if level_type == 'sequential':
        colors=sns.palettes.color_palette(palette='RdYlBu_r', n_colors=nLevels-1).as_hex()
    elif level_type == 'categorical':
        colors=sns.palettes.color_palette(palette='tab10', n_colors=nLevels-1).as_hex()
    colors.append('#808080') # add gray for nan
    colorIdx_dict = dict(enumerate(classes))
    colorIdx_dict = defaultdict(lambda:nLevels+1, {v:k for k,v in colorIdx_dict.items()})
    df['color'] = df[color_by_binned].apply(lambda x: colors[colorIdx_dict[x]])
    legend_theme=dict(click_policy = 'hide', border_line_alpha = 0,label_text_font_size = '12px', background_fill_alpha = 0)
    scatter_theme=dict(alpha = .8, size=ms, line_color='black', line_width=.5, line_alpha=.5)
    
    fig = figure(plot_width=plot_width, plot_height=plot_height, tools='hover,lasso_select, pan, wheel_zoom, box_zoom, crosshair, save, reset', title='tSNE',
                 x_axis_label='tSNE1', y_axis_label='tSNE2', active_drag="lasso_select")
    if data_label:
        # Create a dummy glyph to tie to data labels
        dataLabels = fig.scatter(0, 0, legend_label="data labels", color=None, visible=False)
        data_labels = []

    for cl, CL in enumerate(classes): 
        if data_label:
            tSNE_source = ColumnDataSource(df[df[color_by_binned]==CL])
            tSNE_scatter = fig.scatter(x='tSNE1', y='tSNE2', color='color', legend_label=CL,
                                  source=tSNE_source, **scatter_theme, visible = True)
                    # Create a dummy glyph to tie to data labels
            dL_renderer = fig.scatter(0, 0, legend_label="data labels", color=None, visible=False)
            data_labels.append(LabelSet(x='tSNE1', y='tSNE2', text=data_label,
                          x_offset=0, y_offset=0, source=tSNE_source, text_font_size='10px', visible=False))
            fig.add_layout(data_labels[cl])
            tSNE_scatter.js_on_change('visible', CustomJS(args=dict(ls=data_labels[cl], dL_renderer=dL_renderer),
                                          code="""if(dL_renderer.visible == true){
                                                      ls.visible = cb_obj.visible;
                                                  }"""))
            
            dL_renderer.js_on_change('visible', CustomJS(args=dict(ls=data_labels[cl],tSNE_scatter=tSNE_scatter),
                                          code="""if(tSNE_scatter.visible == true){
                                                      ls.visible = cb_obj.visible;
                                                  }"""))
        else: 
            tSNE_source = ColumnDataSource(df[df[color_by_binned]==CL])
            tSNE_scatter = fig.scatter(x='tSNE1', y='tSNE2', color='color', legend_label=CL,
                                      source=tSNE_source, **scatter_theme, visible = True)

    tooltips = [("tSNE1", "@tSNE1{0.00}"),
                             ("tSNE2", "@tSNE2{0.00}"),
                             (color_by, "@"+color_by)]
    if data_label:
        tooltips.append((data_label, "@"+data_label))
    fig.tools[0].tooltips = tooltips
    fig.toolbar.active_scroll=fig.tools[3]
    fig.legend.__dict__['_property_values'].update(legend_theme)
    fig.add_layout(fig.legend[0], 'right')
    if table_vars: 
        columns = [TableColumn(field=var, title=var) for var in table_vars]
        for c in columns[3:]:
            c.formatter = NumberFormatter(format="0.00")
        dt_scat_src = ColumnDataSource(df)
        dt_source = ColumnDataSource(df)
        dt_scatter = fig.scatter(x="tSNE1", y="tSNE2", source=dt_scat_src, size=0)
        data_table = DataTable(source=dt_source, columns = columns,width=table_width, height=table_height, editable=True)
        dt_scat_src.selected.js_on_change('indices', CustomJS(args=dict(dt_scat_src=dt_scat_src, dt_source=dt_source, data_table=data_table), 
                                                              code=updateOnSelectCode(target_vars=table_vars)))

        show(column(fig, data_table))
    else:
        show(fig)
    if return_outputs: 
        return df
    return
    
def plot_pacmap(df, features = [], 
             data_label=None, table_vars = None, pacmap_params=dict(),
             plot_width=750, plot_height=plot_height, table_width=725, table_height=table_height, 
              n_clusters=5, ms=10, return_outputs=True, color_by='Cluster', nQuantiles=10, bins=None):
    """ 
    features: to use for clustering algorithm
    table_vars: variables that we can look further into inspection
    pacmap_params :   dictionary of pacmap params: default=dict() 
    n_dims=3,
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
    """
    import pacmap
    df=df.copy()
    df2fit = CustomScaler().fit_transform(df[features])
    df2fit = df2fit.fillna(df2fit.median())
    pacmap_params_final = dict(n_dims=3)
    if len(pacmap_params) == 0:
        pass
    else:
        pacmap_params_final.update(pacmap_params)
    #TODO: update functionality that allows you to see intermediate iterations (This will require a dictionary of scatter plot sources)
    intermediate=False
    if 'intermediate' in pacmap_params:
        if pacmap_params['intermediate']:
            intermediate=True
            return_update_fcn=True
    mdl_outputs = pacmap.PaCMAP(**pacmap_params_final).fit_transform(df2fit.values)
    if intermediate: 
        df['pacmap1'] = mdl_outputs[-1,:,0]
        df['pacmap2'] = mdl_outputs[-1,:,1]
    else:
        df['pacmap1'] = mdl_outputs[:,0]
        df['pacmap2'] = mdl_outputs[:,1]
        df['pacmap3'] = mdl_outputs[:,2]
    clusters = KMeans(n_clusters=n_clusters, random_state=1).fit_predict(df[['pacmap1', 'pacmap2']])
    df['Cluster'] = list(map(str, clusters))
    if color_by == 'Cluster':
        level_type='categorical'
        color_by_binned = 'Cluster'
        classes = list(df['Cluster'].unique())
    else:
        level_type='sequential'
        color_by_binned = color_by+"_binned"
        if len(list(df[color_by].unique())) <7:
            classes = list(df[color_by].unique())
            df[color_by+"_binned"] = df[color_by].astype(str)
            classes = [str(v) for v in classes]
        else:
            if type(bins) == list:
                df[color_by+"_binned"] = pd.cut(df[color_by+"_binned"], bins = bins, include_lowest=True, duplicates = 'drop')
            elif ((df[color_by] == df[color_by].min()).sum()/df[color_by].shape[0]) > .025: # If greater than 2.5% of the data is the lowest value: Make a bin for the LoQ/Zero value
                lowest_binEdge = (df[color_by].min() == df[color_by]).sum()/df[color_by].notna().sum()
                bins = np.append(np.array([0]), np.linspace(lowest_binEdge,1, nQuantiles)) 
                df[color_by+"_binned"] = pd.cut(df[color_by+"_binned"], bins = [df[color_by].quantile(q) for q in bins], include_lowest=True, duplicates = 'drop')
            else:
                bins=[df[color_by].quantile(q) for q in np.linspace(0,1, nQuantiles+1)]
                bins[0] = bins[0]-.1
                df[color_by+"_binned"] = pd.cut(df[color_by], bins = bins, include_lowest=True, duplicates = 'drop')
            classes = df[color_by+"_binned"].cat.categories.astype(str).to_list() # sorted list
        classes.append('nan')
        df[color_by_binned] = df[color_by_binned].astype(str)
        
    nLevels = len(classes)
    if level_type == 'sequential':
        colors=sns.palettes.color_palette(palette='RdYlBu_r', n_colors=nLevels-1).as_hex()
    elif level_type == 'categorical':
        colors=sns.palettes.color_palette(palette='tab10', n_colors=nLevels-1).as_hex()
    colors.append('#808080') # add gray for nan
    colorIdx_dict = dict(enumerate(classes))
    colorIdx_dict = defaultdict(lambda:nLevels+1, {v:k for k,v in colorIdx_dict.items()})
    df['color'] = df[color_by_binned].apply(lambda x: colors[colorIdx_dict[x]])
    legend_theme=dict(click_policy = 'hide', border_line_alpha = 0,label_text_font_size = '12px', background_fill_alpha = 0)
    scatter_theme=dict(alpha = .8, size=ms, line_color='black', line_width=.5, line_alpha=.5)
    
    fig = figure(plot_width=plot_width, plot_height=plot_height, tools='hover,lasso_select, pan, wheel_zoom, box_zoom, crosshair, save, reset', title='PaCMAP',
                 x_axis_label='pacmap1', y_axis_label='pacmap2', active_drag="lasso_select")
    if data_label:
        # Create a dummy glyph to tie to data labels
        dataLabels = fig.scatter(0, 0, legend_label="data labels", color=None, visible=False)
        data_labels = []

    for cl, CL in enumerate(classes): 
        if data_label:
            pacmap_source = ColumnDataSource(df[df[color_by_binned]==CL])
            pacmap_scatter = fig.scatter(x='pacmap1', y='pacmap2', color='color', legend_label=CL,
                                  source=pacmap_source, **scatter_theme, visible = True)
                    # Create a dummy glyph to tie to data labels
            dL_renderer = fig.scatter(0, 0, legend_label="data labels", color=None, visible=False)
            data_labels.append(LabelSet(x='pacmap1', y='pacmap2', text=data_label,
                          x_offset=0, y_offset=0, source=pacmap_source, text_font_size='10px', visible=False))
            fig.add_layout(data_labels[cl])
            pacmap_scatter.js_on_change('visible', CustomJS(args=dict(ls=data_labels[cl], dL_renderer=dL_renderer),
                                          code="""if(dL_renderer.visible == true){
                                                      ls.visible = cb_obj.visible;
                                                  }"""))
            
            dL_renderer.js_on_change('visible', CustomJS(args=dict(ls=data_labels[cl],pacmap_scatter=pacmap_scatter),
                                          code="""if(pacmap_scatter.visible == true){
                                                      ls.visible = cb_obj.visible;
                                                  }"""))
        else: 
            pacmap_source = ColumnDataSource(df[df[color_by_binned]==CL])
            pacmap_scatter = fig.scatter(x='pacmap1', y='pacmap2', color='color', legend_label=CL,
                                      source=pacmap_source, **scatter_theme, visible = True)

    tooltips = [("pacmap1", "@pacmap1{0.00}"),
                             ("pacmap2", "@pacmap2{0.00}"),
                             (color_by, "@"+color_by)]
    if data_label:
        tooltips.append((data_label, "@"+data_label))
    fig.tools[0].tooltips = tooltips
    fig.toolbar.active_scroll=fig.tools[3]
    fig.legend.__dict__['_property_values'].update(legend_theme)
    fig.add_layout(fig.legend[0], 'right')
    if table_vars: 
        columns = [TableColumn(field=var, title=var) for var in table_vars]
        for c in columns[3:]:
            c.formatter = NumberFormatter(format="0.00")
        dt_scat_src = ColumnDataSource(df)
        dt_source = ColumnDataSource(df)
        dt_scatter = fig.scatter(x="pacmap1", y="pacmap2", source=dt_scat_src, size=0)
        data_table = DataTable(source=dt_source, columns = columns,width=table_width, height=table_height, editable=True)
        dt_scat_src.selected.js_on_change('indices', CustomJS(args=dict(dt_scat_src=dt_scat_src, dt_source=dt_source, data_table=data_table), 
                                                              code=updateOnSelectCode(target_vars=table_vars)))

        show(column(fig, data_table))
    else:
        show(fig)
    if return_update_fcn:
        pacmap_source
    if return_outputs: 
        return df
    return    

def scatterplot(x, y, df, color=None, tooltips= 'All', 
             data_label=None, table_vars = 'All', 
             plot_width=750, plot_height=450, 
             table_width=725, table_height=100):
    if color: 
      Class_dict = dict(enumerate(df[color].unique()))
      Class_dict = defaultdict(lambda:12, {v:k for k,v in Class_dict.items()})
      df['color'] = df[color].apply(lambda x: colors[Class_dict[x]] if Class_dict[x]<= 10 else 'gray')
    else: 
      df['color'] = 'gray'
    legend_theme=dict(click_policy = 'hide', border_line_alpha = 0,label_text_font_size = '12px', background_fill_alpha = 0)
    scatter_theme=dict(alpha = .8, size=8, line_color='black')
    
    fig = figure(plot_width=plot_width, plot_height=plot_height, tools='hover,lasso_select, pan, wheel_zoom, box_zoom, crosshair, save, reset', title=x+' vs.' + y,
                 x_axis_label=x, y_axis_label=y, active_drag="lasso_select")

    if color: 
      for cl, CL in enumerate(df[color].unique()): 
          currScat_source = ColumnDataSource(df[df[color]==CL])
          currScatter = fig.scatter(x=x, y=y, color='color', legend_label=str(CL),
                                    source=currScat_source, **scatter_theme, visible = True)
    else: 
        currScat_source = ColumnDataSource(df)
        currScatter = fig.scatter(x=x, y=y, color='color',
                                    source=currScat_source, **scatter_theme, visible = True)


    if type(tooltips) == list:
        # print(tooltips)
        fig.tools[0].tooltips = [(col, "@"+col) for col in tooltips]
    elif tooltips == 'All':
        fig.tools[0].tooltips = [(col, "@"+col) for col in df.columns[:10]]

    fig.toolbar.active_scroll=fig.tools[3]
    fig.legend.__dict__['_property_values'].update(legend_theme)
    fig.legend.title = color
    if color: 
      fig.add_layout(fig.legend[0], 'right')
    if type(table_vars) == list: 
        # print(table_vars)
        columns = [TableColumn(field=var, title=var) for var in table_vars]
        for c in columns:
            c.formatter = NumberFormatter(format="0.000")
        dt_scat_src = ColumnDataSource(df)
        dt_source = ColumnDataSource(df)
        dt_scatter = fig.scatter(x=x, y=y, source=dt_scat_src, size=0)
        data_table = DataTable(source=dt_source, columns = columns,width=table_width, height=table_height, editable=True)
        dt_scat_src.selected.js_on_change('indices', CustomJS(args=dict(dt_scat_src=dt_scat_src, dt_source=dt_source, data_table=data_table), 
                                                              code=updateOnSelectCode(target_vars=table_vars)))
        show(column(fig, data_table))
    elif table_vars == 'All':
        columns = [TableColumn(field=var, title=var) for var in df.columns[:10]]
        for c in columns:
            c.formatter = NumberFormatter(format="0.000")
        dt_scat_src = ColumnDataSource(df)
        dt_source = ColumnDataSource(df)
        dt_scatter = fig.scatter(x=x, y=y, source=dt_scat_src, size=0)
        data_table = DataTable(source=dt_source, columns = columns,width=table_width, height=table_height, editable=True)
        dt_scat_src.selected.js_on_change('indices', CustomJS(args=dict(dt_scat_src=dt_scat_src, dt_source=dt_source, data_table=data_table), 
                                                              code=updateOnSelectCode(target_vars=table_vars)))


        show(column(fig, data_table))
    else:
        show(fig)
    return 



