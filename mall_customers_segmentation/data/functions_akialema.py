import pandas as pd
import numpy as np
from IPython.display import HTML
import copy
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# My info.
def my_info(df, catmaxnum=None):
    """Info function to show basic df indicators

    Args:
        df (pd.Dataframe): any df.
        catmaxnum (int): the minimum threshold(rus: порог) after which a column will be considered a category. Defaults to None.

    Returns:
        _type_: HTML object
    """
    
    df_copy = df.copy() # it needs here.
    
    # Loop and condition for moment, when there are unhashable types in our column. (list, dict, set)
    for i in df_copy.columns:
        if any(isinstance(cell, (list, set, dict)) for cell in df_copy[i]):
            df_copy[i] = df_copy[i].astype(str)
                
    func_df = pd.DataFrame({
        
        'column': df_copy.columns,
        'num of unique vals': df_copy.nunique(),
        'type': [str(df_copy.dtypes[i]) for i in df_copy.columns],
        'mode': [round(df_copy[i].mode()[0], 2) if pd.api.types.is_numeric_dtype(df_copy[i].dtype) else df_copy[i].mode()[0] for i in df_copy.columns],
        'number of entries': len(df_copy),
        'NaN vals': df_copy.isna().sum(),
        'number of dublics': df_copy.duplicated().sum(),
        # "rename" for display statistics correctly without conflicts with possible identical column names
        'describe': [f"{df_copy[i].describe().reindex(['min', 'max', 'mean', 'std']).rename({'min':'m1n', 'max':'m4x', 'mean':'m3an', 'std':'s7d'}).round(2).to_string()}" if pd.api.types.is_numeric_dtype(df_copy[i].dtype) else 'see type column' for i in df_copy.columns]
        
    }).sort_values(by=['num of unique vals', 'column'], ascending=[True, True]).reset_index(drop=True)
    
    if catmaxnum:
        func_df.insert(3, 'classification', ['category' if i < catmaxnum else 'numeric' for i in func_df['num of unique vals']])
    
    if len(str(df_copy.memory_usage(deep=True).sum())) > 6:
        print(f"memory usage: {str(round(df_copy.memory_usage(deep=True).sum() / 1e6, 1)) + ' MB'}")
    else:
        print(f"memory usage: {str(round(df_copy.memory_usage(deep=True).sum() / 1e3, 1)) + ' KB'}")
    
    # Add CSS style to left-align the "describe" column
    html_output = func_df.to_html().replace('\\n', '<br>')
    html_output = html_output.replace('<td>m1n', '<td style="text-align: left;">min...:')
    html_output = html_output.replace('<br>m4x', '<br style="text-align: left;">max..:')
    html_output = html_output.replace('<br>m3an', '<br style="text-align: left;">mean:')
    html_output = html_output.replace('<br>s7d', '<br style="text-align: left;">std....:')
    
    return HTML(html_output)

###############################

def inertia_silhouette_plot(X, alg, cluster_range=(1, 10), coef='s', plot_or_df=False, figsize=(12, 5), ax=False):
    """inertia_silhouette_plot 

    Func for plot ineria, silhouette, DBI for k-means alg.
    
    Args:
        X (pd.DataFrame): incoming X. (X should be DataFrame object)
        alg (sklearn.cluster._kmeans.KMeans): incoming alg.
        cluster_range (tuple, optional): range of iterations. Defaults to (1, 10).
        coef (str, optional): choose what we wanna see: i = inertia, s = silhouette, db = davies_bouldin, ch = calinski_harabas. U can use mix: i+s, i+s+d, s+d etc. Defaults to 's'.
        plot_or_df (bool, optional): if False - plot will be shown, else u can see df with best num of cluster + best score. Defaults: False.
        figsize (tuple, optional): figsize of picture. Defaults to (12, 5).
        ax (bool, optional): ax. Defaults to False.

    Returns:
        plt: plt.show()
    """    
    
    # copy of incoming alg. It needs here.
    func_alg = copy.deepcopy(alg)
    
    # for right param name.
    if isinstance(alg, GaussianMixture):
        param_name = 'n_components'
    else:
        param_name = 'n_clusters'
    
    # variable for conditions 
    condition_series = pd.Series(index=['i', 's', 'db', 'ch']).fillna(0)
    condition_series.update(Counter(coef.split('+')))
    
    # final list
    list_of_choosed_coef = []
    
    # here can be 3 types of output:
    # i+s == inertia + silhouette
    # i+s+d == i+s + davies_bouldin_score
    # i+s+d+g = i+s+d + gap statistis score (coming soon)
    if condition_series['i'] == 1:
        # Series for plot.
        inertia_series = pd.Series([func_alg.set_params(**{param_name: i}).fit(X).inertia_ for i in range(*cluster_range)], index=range(*cluster_range), name='Inertia')
        list_of_choosed_coef.append(inertia_series)
        
    if condition_series['s'] == 1:
        # cuz silhouette cant start from 1
        if cluster_range[0] == 1:
            cluster_range = (2, cluster_range[1])
        # Series for plot.
        silhouette_series = pd.Series([silhouette_score(X, func_alg.set_params(**{param_name: i}).fit_predict(X)) for i in range(*cluster_range)], index=range(*cluster_range), name='Silhouette')
        list_of_choosed_coef.append(silhouette_series)
        
    if condition_series['db'] == 1:
        
        davies_bouldin_series = pd.Series([davies_bouldin_score(X, func_alg.set_params(**{param_name: i}).fit_predict(X)) for i in range(*cluster_range)], index=range(*cluster_range), name='Davies-Bouldin Index')
        list_of_choosed_coef.append(davies_bouldin_series)
    
    if condition_series['ch'] == 1:
        
        calinski_harabas_series = pd.Series([calinski_harabasz_score(X, func_alg.set_params(**{param_name: i}).fit_predict(X)) for i in range(*cluster_range)], index=range(*cluster_range), name='Calinski-Harabas Index')
        list_of_choosed_coef.append(calinski_harabas_series)
    
    if not plot_or_df:    
        # if coef was choosen wrong
        if (condition_series == 0).all():
            raise ValueError('"coef" parameter can be: "i", "s", "db", "ch", or mix for ex: "i+s", "i+s+db" etc')
    
        # plotting
        if ax:
            axes = ax
        else:
            fig, axes = plt.subplots(ncols=len(list_of_choosed_coef), figsize=figsize)
            fig.tight_layout(w_pad=5, h_pad=3) # the distance between subgraphs
            #plt.subplots_adjust(top=0.9) # for title
        
        if len(list_of_choosed_coef) == 1:
            axes = [axes]  
        
        for i, j in enumerate(list_of_choosed_coef):
            axes[i] = sns.lineplot(j, ax=axes[i], marker='o')
            
            axes[i].set(xlabel='Cluster', ylabel=f'{j.name}', title=f'{j.name} plot', xticks=range(*cluster_range, 1), yticks=j.round(2))
        
        if not ax:
            return plt
    else: # <-- this is test condition.. back here someday
        output_df = pd.DataFrame()
        
        for i in list_of_choosed_coef:
            if i.name == 'Inertia':
                output_df = pd.concat([output_df, pd.DataFrame([[i.idxmin(), i.min()]], index=[i.name], columns=['Num_of_clusters', 'Best_score'])], axis=0)
            elif i.name == 'Silhouette':
                output_df = pd.concat([output_df, pd.DataFrame([[i.idxmax(), i.max()]], index=[i.name], columns=['Num_of_clusters', 'Best_score'])], axis=0)
            elif i.name == 'Davies-Bouldin Index':
                output_df = pd.concat([output_df, pd.DataFrame([[i.idxmin(), i.min()]], index=[i.name], columns=['Num_of_clusters', 'Best_score'])], axis=0)
            elif i.name == 'Calinski-Harabas Index':
                output_df = pd.concat([output_df, pd.DataFrame([[i.idxmax(), i.max()]], index=[i.name], columns=['Num_of_clusters', 'Best_score'])], axis=0)
        return output_df