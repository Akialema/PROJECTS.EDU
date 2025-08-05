import pandas as pd
import numpy as np
from IPython.display import HTML
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm # рисовалка qq-plot.
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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
        'mode': [round(df_copy[i].mode()[0], 2) if np.issubdtype(df_copy[i].dtype, np.number) else df_copy[i].mode()[0] for i in df_copy.columns],
        'number of entries': len(df_copy),
        'NaN vals': df_copy.isna().sum(),
        'number of dublics': df_copy.duplicated().sum(),
        'describe': [f"{df_copy[i].describe().reindex(['min', 'max', 'mean', 'std']).round(2).to_string()}" if np.issubdtype(df_copy[i].dtype, np.number) else 'type obj' for i in df_copy.columns]
        
    }).sort_values(by=['num of unique vals', 'column'], ascending=[True, True]).reset_index(drop=True)
    
    if catmaxnum:
        func_df.insert(3, 'classification', ['category' if i < catmaxnum else 'numeric' for i in func_df['num of unique vals']])
    
    if len(str(df_copy.memory_usage(deep=True).sum())) > 6:
        print(f"memory usage: {str(round(df_copy.memory_usage(deep=True).sum() / 1e6, 1)) + ' MB'}")
    else:
        print(f"memory usage: {str(round(df_copy.memory_usage(deep=True).sum() / 1e3, 1)) + ' KB'}")
    
    # Add CSS style to left-align the "describe" column
    html_output = func_df.to_html().replace('\\n', '<br>')
    html_output = html_output.replace('<td>min', '<td style="text-align: left;">min...:')
    html_output = html_output.replace('<br>max', '<br style="text-align: left;">max..:')
    html_output = html_output.replace('<br>mean', '<br style="text-align: left;">mean:')
    html_output = html_output.replace('<br>std', '<br style="text-align: left;">std....:')
    
    return HTML(html_output)

#######################################

def my_corr(df, targ_col=False, method='pearson', corrneg=0, corrpos=0, figsize=(10, 5), xtickrot=90, annot_kws={'size': 'medium'}, show_high_corr_cols_names=False):
    """
    The function of drawing a correlation graph, but not an ordinary one, but one in which the lower part is the entire one,
    and the upper part contains only those correlation values that are greater and less than some number we have specified.

    Args:
        df (pd.Dataframe): any df.
        method (str): method u like. default pearson.
        corrneg (float): negative threshold for display. (for negative correlation value)
        corrpos (float): positive threshold. (for a positive value) both default to 0. At 0, the full standard correlation matrix will be displayed.
        figsize (tuple): u know what it is.)
        xtickrot(int): rotate x axis labels. default 90
        annot_kws(dict): size of annot font. default {"size": 'medium'}
        show_high_corr_cols_names(bool): if we wanna watch names of high corr cols. default False

    Returns:
        _type_: plt or df
    """
    
    if targ_col:
        corr_df = pd.concat([df.drop(targ_col, axis=1), df[targ_col]], axis=1).corr(numeric_only=True, method=method).copy()
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
        fig, ax = plt.subplots(figsize=figsize)
    
        plt.suptitle(f'Correlation of incoming dataframe features.')
        
        # 2 masks, one triu (tri upper), the second tril (tri lower).
        mask_for_logging_df = np.triu(np.ones_like(corr_df, dtype=bool))
        mask_for_logging_df_2 = np.tril(np.ones_like(high_corr_features, dtype=bool))
        
        # 2 graphics. I didn’t figure out how to make such a picture using 1 chart. That's why cbar=False helps.
        sns.heatmap(corr_df, annot=True, annot_kws=annot_kws, cmap='coolwarm', mask=mask_for_logging_df, vmax=1, vmin=-1, linewidths=0.1, fmt='.2f') # <--- lower part
        
        # if condition == 1 high_corr_features contains only nan, so we dont need upper part. 
        # or.. high_corr_features has no nan, i.e. corrpos and corrneg == 0, so we draw all corr matrix.
        if (len(high_corr_features.isna().sum().value_counts()) != 1) | ((corrpos == 0) & (corrneg == 0)):
            sns.heatmap(high_corr_features, annot=True, annot_kws=annot_kws, cmap='coolwarm', mask=mask_for_logging_df_2, cbar=False, vmax=1, vmin=-1, linewidths=0.1, fmt='.2f') # <--- upper part
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=xtickrot)
    
        plt.plot([0, len(df.columns)], [0, len(df.columns)], color='black', linewidth=2); # It's just a line. I don't know why. But i like it.
        
        return plt
    
#######################################
    
# Tukie outliers finder function.
def outliers_iqr_mode_log(data, feature=None, log_scale=False, left=1.5, right=1.5):
    
    func_data = data.copy()
    
    if isinstance(func_data, pd.DataFrame):    
        if log_scale: # if we want to logarithm a feature.
            if any(func_data[feature] == 0) and all(func_data[feature] >= 0): # if we have 0 values and all vals > 0
                x = np.log(func_data[feature] + 1)
            elif any(func_data[feature] == 0) and not all(func_data[feature] >= 0): # if we have 0 and negative vals
                x = np.log(abs(func_data[feature]) + 1)
            else:
                x = np.log(func_data[feature])
        else:
            x = func_data[feature]
    else:
        x = func_data # if data == array or series
        
    quantile_25, quantile_75 = x.quantile(0.25), x.quantile(0.75)
    iqr = quantile_75 - quantile_25
    
    lower_bound = quantile_25 - iqr * left
    upper_bound = quantile_75 + iqr * right
    
    outliers = func_data[(x < lower_bound) | (x > upper_bound)]
    cleaned = func_data[(x >= lower_bound) & (x <= upper_bound)]
    
    return outliers, cleaned.reset_index(drop=True) # .copy() is needed here if we wanna use cleaned daataframe. Or system will swear.

#######################################

def hist_box_qq(arg, full_iqr=False):
    """Function of drawing histogram, box and cuckoo in one bottle.

    full_iqr... It's shorter.. I wrote a function for the lognormal distribution, i.e. there are no outliers on the left.
    But then I thought about it and decided to add it, because then I still log the feature.
    And in general, if the array is usually distributed.
    In a word: full_iqr is Boolean. By default, False. I.e., only the right borders are drawn.
    
    The input is a Series or array.

    Args:
        arg (Series/np.array): any numeric series or array.
        full_iqr (bool, optional): choose, u wanna show only right outliers borders or both. Defaults to False.

    Returns:
        plt
    """
    
    func_arg = arg.copy()
    # replace inf.
    func_arg = func_arg.replace([np.inf, -np.inf], np.nan)
    # If there is a nan, I drop it. Because it does not draw if there is a nan.
    func_arg = pd.Series(func_arg).dropna()
    
    fig, ax = plt.subplots(ncols=3, nrows=1, figsize=[18, 5])
    
    # The histogram. And the settings for the histogram lines. Legend = label in each line.
    # Red is the median, green is the tukey, and blue is 3 sigma.
    sns.histplot(func_arg, kde=True, bins=50, ax=ax[0])
    
    ax[0].get_lines()[0].set_color('black')
    ax[0].axvline(func_arg.median(), color='red', linestyle='--', linewidth='1.8', label='median')
    ax[0].axvline(func_arg.quantile(0.75) + ((func_arg.quantile(0.75) - func_arg.quantile(0.25)) * 1.5), color='g', ls='--', lw=2, label='1.5 IQR Tukie')
    ax[0].axvline(func_arg.mean() + 3 * func_arg.std(), color='b', ls='--', lw=2, label='3 IQR z-score')
    ax[0].legend()
    ax[0].set_xlabel(f'Feauture {func_arg.name}')
    
    # Box. And the settings for it.
    sns.boxplot(func_arg, ax=ax[1], orient='h', medianprops={'color': 'red', 'linestyle': '--'})

    ax[1].axvline(func_arg.quantile(0.75) + ((func_arg.quantile(0.75) - func_arg.quantile(0.25)) * 1.5), color='g', ls='--', lw=2, label='1.5 IQR Tukie')
    ax[1].axvline(func_arg.mean() + 3 * func_arg.std(), color='b', ls='--', lw=2, label='3 IQR z-score')
    ax[1].legend()    
    ax[1].set_xlabel(f'Feauture {func_arg.name}')
    
    if full_iqr: # Left emission catch lines are added.
        ax[0].axvline(func_arg.quantile(0.25) - ((func_arg.quantile(0.75) - func_arg.quantile(0.25)) * 1.5), color='g', ls='--', lw=2)
        ax[0].axvline(func_arg.mean() - 3 * func_arg.std(), color='b', ls='--', lw=2)
        ax[1].axvline(func_arg.quantile(0.25) - ((func_arg.quantile(0.75) - func_arg.quantile(0.25)) * 1.5), color='g', ls='--', lw=2)
        ax[1].axvline(func_arg.mean() - 3 * func_arg.std(), color='b', ls='--', lw=2)
    
    # This line builds a Q-Q graph.
    qq = sm.ProbPlot(func_arg, fit=True).qqplot(marker='*', markerfacecolor='b', markeredgecolor='b', alpha=0.3, ax=ax[2])
    # Line, also with the ability to influence color. fmt parameter.
    sm.qqline(qq.axes[2], line='45', fmt='r', linestyle='--')
    
    # The general title.
    plt.suptitle(f'Distribution of the feature: {func_arg.name}').set_fontsize(20)
    
    return plt

#######################################

def print_metrics(y_test, y_test_pred, list_of_metrics, y_train=None, y_train_pred=None, names=False, metric_params_dict=False):
    """print_metrics 

    function for print any metric u like.
    P.S. some metrics have hard names, so need to adjust names in 'block to adjust names' (see code)

    Args:
        y_test (pd.Series or numpy.ndarray): y_test
        y_test_pred (list): y_test_pred (lgr.predict(X_test))
        list_of_metrics (list): list of any metrics
        y_train (pd.Series or numpy.ndarray, optional): y_train. Defaults to None.
        y_train_pred (list, optional): y_train_pred. Defaults to None.
        names (list, optional): list of algo names (for ex: ['lgr_1', 'lgr_2']). Defaults to False.
        metric_params_dict (dict, optional): dict look like {'metric_name': {'parameter': 'value'}}. Defaults to False.

    Returns:
        pd.DataFrame: with our metrics
        
    ex of using:
    
    list_of_metrics = ['accuracy_score', 'f1_score', 'precision_score', 'recall_score']
    metric_params_dict = {'accuracy_score': {'normalize': False}}
    
    print_metrics(y_test=y_test, 
                  y_test_pred=[y_test_pred], 
                  list_of_metrics=list_of_metrics,
                  names=['lgr_1'], metric_params_dict=metric_params_dict)
    """
    
    # if we didnt set names it will be 0, 1 etc.
    if not names:
        names = [f'{i}' for i in range(len(y_test_pred))]
    
    ################### <----- block to adjust metrics params
    # if we dont want to specify additional metric params
    if not metric_params_dict:
        metric_params_dict_func = {i:{} for i in list_of_metrics}
    # else we make empty dict with each metric and update it with needed external dict
    else:
        metric_params_dict_func = {i:{} for i in list_of_metrics}
        metric_params_dict_func.update(metric_params_dict)
    
    ################### <----- block to adjust names 
    # This for correct metrics names.
    func_list_of_metrics = []
    
    for i in list_of_metrics:
        # For ex: mean_absolute_error = MAE (if len(['mean', 'absolute', 'error']) > 2)
        if len(i.split('_')) > 2:
            func_list_of_metrics.append(''.join([j[0].capitalize() for j in i.split('_')]))
        # else r2_score = r2
        else:
            func_list_of_metrics.append(i.split('_')[0])
    ###################
    
    train_dict = {}
    test_dict = {}
    # if we wanna see just test metrics:
    if y_train is None:
        # for each name + test_pred
        for name, y_test_p in zip(names, y_test_pred):
            # we make dict with {metric_name: metric_score}
            test_dict.update({f"{func_list_of_metrics[i]}_{name}": globals()[j](y_test, y_test_p, **metric_params_dict_func[j]) for i, j in enumerate(list_of_metrics)})
        # and then make series object from dict
        func_df = pd.Series(test_dict).to_frame().rename(columns={0: 'test'})
    
    # if we wanna see train and test scores:    
    else:
        # for each name + train_pred + test_pred
        for name, y_train_p, y_test_p in zip(names, y_train_pred, y_test_pred):
            # we make 2 dicts with train and test scores
            train_dict.update({f"{func_list_of_metrics[i]}_{name}": globals()[j](y_train, y_train_p, **metric_params_dict_func[j]) for i, j in enumerate(list_of_metrics)})
            test_dict.update({f"{func_list_of_metrics[i]}_{name}": globals()[j](y_test, y_test_p, **metric_params_dict_func[j]) for i, j in enumerate(list_of_metrics)})
        # and make dataframe from this dict
        func_df = pd.DataFrame({'train': train_dict,
                                'test': test_dict})
    
    return func_df