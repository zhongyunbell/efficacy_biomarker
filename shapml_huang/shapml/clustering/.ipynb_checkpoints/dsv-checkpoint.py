from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, LabelSet, CustomJS, DataTable, TableColumn, NumberFormatter
from bokeh.layouts import column, row
from bokeh.io import show, output_notebook, output_file, reset_output
from bokeh.palettes import all_palettes
output_notebook()
colors = all_palettes['Spectral'][11]

import pandas as pd
import numpy as np
from collections import defaultdict

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
table_height = 50
def plot_PCA(df, features = ['HSCs', 'MPPs', 'GMP', 'Mono-lin', 'Gran-lin', 'Neut-lin', 'Early Eryth', 'MK-lin', 'B-lin'], 
             data_label=None, table_vars = None, plot_width=750, plot_height=plot_height, table_width=725, table_height=table_height, showPoints=True,
             biplot_coeff = -6, showLongContols=True, compound_labels=None):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2).fit(df[features])
    df['PC1'] = PCA(n_components=2).fit_transform(df[features])[:,0]
    df['PC2'] = PCA(n_components=2).fit_transform(df[features])[:,1]
    Class_dict = dict(enumerate(df.Class.unique()))
    Class_dict = defaultdict(lambda:12, {v:k for k,v in Class_dict.items()})
    df['color'] = df.Class.apply(lambda x: colors[Class_dict[x]] if Class_dict[x]<= 10 else 'gray')
    legend_theme=dict(click_policy = 'hide', border_line_alpha = 0,label_text_font_size = '12px', background_fill_alpha = 0)
    scatter_theme=dict(alpha = .8, size=12, line_color='black')
    
    fig = figure(plot_width=plot_width, plot_height=plot_height, tools='hover,lasso_select, pan, wheel_zoom, box_zoom, crosshair, save, reset', title='PCA',
                 x_axis_label='PC1', y_axis_label='PC2', active_drag="lasso_select")
    if showLongContols:
        option = 2
        if option == 1:
            # Place asterisk below all longitudinal controls that will be controlled by one legend label
            longControls = ['Abemaciclib (VERZENIO)', 'Docetaxel (TAXOTERE)', 'A-1331852', 'Pictilisib']
            pca_source = ColumnDataSource(df[df.Compound.apply(lambda x: x in longControls)])
            pca_scatter = fig.asterisk(x='PC1', y='PC2', color='color', legend_label="Long. controls",
			                            source=pca_source, visible = True ,size=10, line_width=1,
                                       line_color='black')
        elif option == 2:
            # Each longitudinal control has a legend label
            # https://docs.bokeh.org/en/latest/docs/reference/models/markers.html
            marker = "square_pin"
            longControls = ['Abemaciclib (VERZENIO)', 'Docetaxel (TAXOTERE)', 'A-1331852', 'Pictilisib']
            for l in longControls:
                pca_source = ColumnDataSource(df[df.Compound == l])
                pca_scatter = fig.scatter(x='PC1', y='PC2', color='color', legend_label=l,
                                            marker=marker, source=pca_source, **scatter_theme,
                                          visible = showPoints)
            # df = df[df.Compound.apply(lambda x: x not in longControls)]


    if data_label:
        # Create a dummy glyph to tie to data labels
        dataLabels = fig.scatter(0, 0, legend_label="data labels", color=None, visible=False)
        data_labels = []

    if compound_labels: 
        eL_renderer = fig.scatter(0, 0, legend_label="explicit labels", color=None, visible=True)
        el_source = ColumnDataSource(df[df.Compound.apply(lambda x: x in compound_labels)])
        explicit_labels = LabelSet(x='PC1', y='PC2', text='Compound',
                          x_offset=0, y_offset=0, source=el_source, text_font_size='10px', text_color = '#FF0000', visible=True)
        fig.add_layout(explicit_labels)
        eL_renderer.js_on_change('visible', CustomJS(args=dict(ls=explicit_labels, eL_renderer=eL_renderer),
                                          code="""ls.visible = cb_obj.visible;"""))

    for cl, CL in enumerate(df.Class.unique()): 
        if data_label:
            if showLongContols and (option==2):
                # Don't replot longitudinal controls
                pca_source = ColumnDataSource(df[(df.Class == CL) & df.Compound.apply(lambda x: x not in longControls)])
            else:
                pca_source = ColumnDataSource(df[df.Class==CL])
            pca_scatter = fig.scatter(x='PC1', y='PC2', color='color', legend_label=CL,
                                  source=pca_source, **scatter_theme, visible = showPoints)
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
            pca_source = ColumnDataSource(df[df.Class==CL])
            pca_scatter = fig.scatter(x='PC1', y='PC2', color='color', legend_label=CL,
                                      source=pca_source, **scatter_theme, visible = True)



    pc1s = []
    pc2s = []
    biplotLines = []
    for feature in features: 
        coeff = biplot_coeff
        pc1 = np.dot((np.array(features)==feature).astype(int), pca.components_[0])*coeff
        pc2 = np.dot((np.array(features)==feature).astype(int), pca.components_[1])*coeff
        pc1s.append(pc1)
        pc2s.append(pc2)
        biplotLines.append(fig.line([0, pc1], [0, pc2], legend_label="biplot", visible=False))
        
    feature_label_src = ColumnDataSource(pd.DataFrame(zip(features, pc1s, pc2s), columns=['feature', 'PC1', 'PC2']))
    labels = LabelSet(x='PC1', y='PC2', text='feature',
                      x_offset=0, y_offset=0, source=feature_label_src, visible=False)
    fig.add_layout(labels) # biplot labels
    
    biplotLines[0].js_on_change('visible', CustomJS(args=dict(ls=labels),
                                                  code="ls.visible = cb_obj.visible;"))

    fig.tools[0].tooltips = [("PC1", "@PC1{0.00}"),
                             ("PC2", "@PC2{0.00}"),
                             ("Class", "@Class"),
                             ("Compound", "@Compound"),
                             ("Run Date", "@Run_date")]
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
    else:
        show(fig)
    return 



def plot_tSNE(df, features=['HSCs', 'MPPs', 'GMP', 'Mono-lin', 'Gran-lin', 'Neut-lin', 'Early Eryth', 'MK-lin', 'B-lin'], 
              init='pca', random_state=1, perplexity=30, data_label=None,table_vars=None, compound_labels = None, 
              plot_width=750, plot_height=plot_height, table_width=725, table_height=table_height, showPoints=True, showLongContols=True):

    from sklearn.manifold import TSNE
    df['tSNE1'] = TSNE(n_components=2, init=init, random_state=random_state, perplexity=perplexity).fit_transform(df[features])[:,0]
    df['tSNE2'] = TSNE(n_components=2, init=init, random_state=random_state, perplexity=perplexity).fit_transform(df[features])[:,1]
    nClasses = len(df.Class.unique())
    colors = all_palettes['Spectral'][11 if nClasses > 11 else nClasses]
    Class_dict = dict(enumerate(df.Class.unique()))
    Class_dict = defaultdict(lambda : 12, {v:k for k,v in Class_dict.items()})
    df['color'] = df.Class.apply(lambda x: colors[Class_dict[x]] if Class_dict[x]<= 10 else 'gray')
    legend_theme=dict(click_policy = 'hide', border_line_alpha = 0,label_text_font_size = '12px', background_fill_alpha = 0)
    scatter_theme=dict(alpha = .8, size=12, line_color='black')
    
    fig = figure(plot_width=plot_width, plot_height=plot_height, tools='hover,lasso_select, pan, wheel_zoom, box_zoom, crosshair, save,reset', title='tSNE', 
                 x_axis_label='tSNE1', y_axis_label='tSNE2')

    if showLongContols:
        option = 2
        if option == 1:
            # Place asterisk below all longitudinal controls that will be controlled by one legend label
            longControls = ['Abemaciclib (VERZENIO)', 'Docetaxel (TAXOTERE)', 'A-1331852', 'Pictilisib']
            tSNE_source = ColumnDataSource(df[df.Compound.apply(lambda x: x in longControls)])
            pca_scatter = fig.asterisk(x='tSNE1', y='tSNE2', color='color', legend_label="Long. controls",
                                       source=tSNE_source, visible=True, size=10, line_width=1,
                                       line_color='black')
        elif option == 2:
            # Each longitudinal control has a legend label
            # https://docs.bokeh.org/en/latest/docs/reference/models/markers.html
            marker = "square_pin"
            longControls = ['Abemaciclib (VERZENIO)', 'Docetaxel (TAXOTERE)', 'A-1331852', 'Pictilisib']
            for l in longControls:
                tSNE_source = ColumnDataSource(df[df.Compound == l])
                pca_scatter = fig.scatter(x='tSNE1', y='tSNE2', color='color', legend_label=l,
                                          marker=marker, source=tSNE_source, **scatter_theme,
                                          visible=showPoints)
            # df = df[df.Compound.apply(lambda x: x not in longControls)]


    if data_label:
        # Create a dummy glyph to tie to data labels
        dL_renderer = fig.scatter(0, 0, legend_label="data labels", color=None, visible=False)
        data_labels = []

    if compound_labels: 
        eL_renderer = fig.scatter(0, 0, legend_label="explicit labels", color=None, visible=True)
        el_source = ColumnDataSource(df[df.Compound.apply(lambda x: x in compound_labels)])
        explicit_labels = LabelSet(x='tSNE1', y='tSNE2', text='Compound',
                          x_offset=0, y_offset=0, source=el_source, text_font_size='10px', text_color = '#FF0000', visible=True)
        fig.add_layout(explicit_labels)
        eL_renderer.js_on_change('visible', CustomJS(args=dict(ls=explicit_labels, eL_renderer=eL_renderer),
                                          code="""ls.visible = cb_obj.visible;"""))
    for cl, CL in enumerate(df.Class.unique()): 
        if data_label:
            if showLongContols and (option==2):
                # Don't replot longitudinal controls
                tSNE_source = ColumnDataSource(df[(df.Class == CL) & df.Compound.apply(lambda x: x not in longControls)])
            else:
                tSNE_source = ColumnDataSource(df[df.Class==CL])
            tSNE_scatter = fig.scatter(x='tSNE1', y='tSNE2', color='color', legend_label=CL,
                                      source=tSNE_source, **scatter_theme, visible = showPoints)
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
            tSNE_source = ColumnDataSource(df[df.Class==CL])
            tSNE_scatter = fig.scatter(x='tSNE1', y='tSNE2', color='color', legend_label=CL,
                                      source=tSNE_source, **scatter_theme, visible = True)

    fig.tools[0].tooltips = [("tSNE1", "@tSNE1{0.00}"),
                             ("tSNE2", "@tSNE2{0.00}"),
                             ("Class", "@Class"),
                             ("Compound", "@Compound"),
                             ("Run Date", "@Run_date")]
    fig.toolbar.active_scroll=fig.tools[3]
    fig.toolbar.active_drag=fig.tools[1]

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
    return



def scatterplot(x, y, df, color=None, tooltips= 'All', 
             data_label=None, table_vars = 'All', 
             plot_width=750, plot_height=450, 
             table_width=725, table_height=100, showPoints=True):
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



