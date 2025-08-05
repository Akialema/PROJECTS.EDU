import pandas as pd
import numpy as np
from IPython.display import HTML
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from sklearn.metrics import silhouette_score, davies_bouldin_score

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

##############################

def my_corr(df, targ_col=False, method='pearson', corrneg=0, corrpos=0, figsize=(10, 5), xtickrot=90, annot_kws={'size': 'medium'}, show_high_corr_cols_names=False, ax=None):
    """
    The function of drawing a correlation graph, but not an ordinary one, but one in which the lower part is the entire one,
    and the upper part contains only those correlation values that are greater and less than some number we have specified.

    Args:
        df (pd.Dataframe): any df.
        targ_col (str, optional): if targ_col is set, func just move this col into the end of dataframe, to better visualisation. It will be at the last row of the plot. Defaults: False
        method (str, optional): method u like. Default: pearson.
        corrneg (float, optional): negative threshold for display. (for negative correlation value)
        corrpos (float, optional): positive threshold. (for a positive value) both default to 0. At 0, the full standard correlation matrix will be displayed.
        figsize (tuple, optional): u know what it is.)
        xtickrot(int, optional): rotate x axis labels. Default: 90
        annot_kws(dict, optional): size of annot font. Default: {"size": 'medium'}
        show_high_corr_cols_names(bool, optional): if we wanna watch names of high corr cols. Default: False. Should use without plot and other plot args. Just thresholds and this one.
        ax (ax, optional): use only if u wanna add this plot to your multiple plots. Default: None

    Returns:
        _type_: plt or df
    """
    
    if targ_col:
        # sort indexes by correlation with targ_col from highest to lowest
        most_corr_cols_index = df.corr(numeric_only=True, method=method)[targ_col].drop(targ_col).sort_values(ascending=False).index
        # putting targ_col to fist row (y = 1) of the chart with sorted features (watch ur chart and u will get it)
        corr_df = pd.concat([df.drop(targ_col, axis=1)[most_corr_cols_index], df[targ_col]], axis=1).corr(numeric_only=True, method=method).copy()
    else:
        corr_df = df.corr(numeric_only=True, method=method).copy()
    
    # We find indices where the correlation value is greater and less than a certain number.
    high_corr_pairs = ((corr_df >= corrpos) & (corr_df <= 1)) | (corr_df <= corrneg)
    
    # We receive only those pairs of features where the condition is met. The rest are NaN.
    high_corr_features = corr_df[high_corr_pairs]
    
    # if we wanna watch high corr pairs in the form of a df. Without picture. I decided 2 split it up.
    if show_high_corr_cols_names:
        # stack hight corr feature.
        shccn = high_corr_features.stack().to_frame().reset_index().rename(columns={0: 'corr'})
        # sort'em.
        shccn[['level_0', 'level_1']] = np.sort(shccn[['level_0', 'level_1']], axis=1)
        # keep only 1 pair of high corr cols names.
        shccn = shccn.groupby(['level_0', 'level_1']).first().reset_index()
        # removing the diagonal.
        shccn = shccn.query('level_0 != level_1')
        # return with resetting index.
        return shccn.sort_values(by='corr', ascending=False).reset_index(drop=True)
    
    # if we wanna see the pic.
    else:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
    
        ax.set_title(f'Correlation of incoming dataframe features.', pad=10) # узкое место pad=10, чтоб заголовок был на расстоянии от графика
        
        # 2 masks, one triu (tri upper), the second tril (tri lower).
        mask_for_logging_df = np.triu(np.ones_like(corr_df, dtype=bool))
        mask_for_logging_df_2 = np.tril(np.ones_like(high_corr_features, dtype=bool))
        
        # 2 graphics. I didn’t figure out how to make such a picture using 1 chart. That's why cbar=False helps.
        sns.heatmap(corr_df, annot=True, annot_kws=annot_kws, cmap='coolwarm', mask=mask_for_logging_df, vmax=1, vmin=-1, linewidths=0.1, fmt='.2f', ax=ax) # <--- lower part
        
        # if condition == 1 high_corr_features contains only nan, so we dont need upper part. 
        # or.. high_corr_features has no nan, i.e. corrpos and corrneg == 0, so we draw all corr matrix.
        if (len(high_corr_features.isna().sum().value_counts()) != 1) | ((corrpos == 0) & (corrneg == 0)):
            sns.heatmap(high_corr_features, annot=True, annot_kws=annot_kws, cmap='coolwarm', mask=mask_for_logging_df_2, cbar=False, vmax=1, vmin=-1, linewidths=0.1, fmt='.2f', ax=ax) # <--- upper part
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=xtickrot)
    
        ax.plot([0, len(df.columns)], [0, len(df.columns)], color='black', linewidth=2); # It's just a line. I don't know why. But i like it.
        
        return plt
    
##############################

def fillna_dict_for_custom_df(df, func, targ_col=None, roundnum=10):
    """Function for generating a dictionary for the fillna function argument.
    
    The function takes as input a dataframe with np.nan and the function that we want to aggregate.
    That is, we have a feature in which there is a nan. We want to fill the nan, for example, with the median.
    We run through the names of the columns that have nan and assign each nan a median value for the column.
    Why the condition? When we ask for mode, we get a Series object that needs to be accessed by index to get the value.
    And when we ask for the average or median, it immediately returns the value.

    Args:
        df (pd.Dataframe): any df
        func (str): any func in str format. 
        targ_col (str, optional): name of target column by which we wanna groupby and have needed agregation. Defaults: None.
        roundnum (int, optional): num of np.round(X, ?), how we want to round our outcoming number. Defaults: 10.
        
        Ex: df.fillna(value=fillna_dict_for_custom_df(df, 'mean'))

    Returns:
        _type_: dict
    """
    
    if targ_col:
        return {i: df.groupby(targ_col)[i].transform(lambda x: x.mode().iloc[0]) if np.issubdtype(df[i], 'object') or func == 'mode' else np.round(df.groupby(targ_col)[i].transform(func), roundnum) for i in df.loc[:, df.isna().mean() > 0].columns}
    return {i: df[i].mode().iloc[0] if np.issubdtype(df[i], 'object') or func == 'mode' else np.round(df[i].agg(func), roundnum) for i in df.loc[:, df.isna().mean() > 0].columns}

##############################

def inertia_silhouette_plot(X, alg, cluster_range=(1, 10), coef='silhouette', figsize=(12, 5), ax=False):
    """inertia_silhouette_plot 

    Func for plot ineria, silhouette, DBI for k-means alg.
    
    Args:
        X (pd.DataFrame): incoming X. (X should be DataFrame object)
        alg (sklearn.cluster._kmeans.KMeans): incoming alg.
        cluster_range (tuple, optional): range of iterations. Defaults to (1, 10).
        coef (str): choose what we wanna see: inertia plot, silhouette plot, davies_bouldin plot or maybe mix: i+s, i+s+d. Defaults to 'silhouette'.
        figsize (tuple, optional): figsize of picture. Defaults to (12, 5).

    Returns:
        plt: plt.show()
    """
    
    # copy of incoming alg. It needs here.
    func_alg = copy.deepcopy(alg)
    list_of_choosed_coef = []
    
    # here can be 3 types of output:
    # i+s == inertia + silhouette
    # i+s+d == i+s + davies_bouldin_score
    # i+s+d+g = i+s+d + gap statistis score (coming soon)
    if coef == 'inertia' or coef == 'i+s' or coef == 'i+s+d':
        # Series for plot.
        inertia_series = pd.Series([func_alg.set_params(n_clusters=i).fit(X).inertia_ for i in range(*cluster_range)], index=range(*cluster_range), name='Inertia')
        list_of_choosed_coef.append(inertia_series)
        
    if coef == 'silhouette' or coef == 'i+s' or coef == 'i+s+d':
        # cuz silhouette cant start from 1
        if cluster_range[0] == 1:
            cluster_range = (2, cluster_range[1])
        # Series for plot.
        silhouette_series = pd.Series([silhouette_score(X, func_alg.set_params(n_clusters=i).fit(X).labels_) for i in range(*cluster_range)], index=range(*cluster_range), name='Silhouette')
        list_of_choosed_coef.append(silhouette_series)
        
    if coef == 'davies_bouldin' or coef == 'i+s+d':
        
        davies_bouldin_series = pd.Series([davies_bouldin_score(X, func_alg.set_params(n_clusters=i).fit_predict(X)) for i in range(*cluster_range)], index=range(*cluster_range), name='Davies-Bouldin Index')
        list_of_choosed_coef.append(davies_bouldin_series)
    
    # if coef was choosen wrong
    if coef not in ['inertia', 'silhouette', 'davies_bouldin', 'i+s', 'i+s+d']:
        raise ValueError('"coef" parameter can be: "inertia", "silhouette", "davies_bouldin", "i+s", "i+s+d"')

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

##############################

def good_print(text, answer):
    print(f"{text}\n{'-'*(len(text)-1)}\nОтвет: {answer}\n")