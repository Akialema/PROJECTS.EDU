import pandas as pd
import numpy as np
from IPython.display import HTML
import matplotlib.pyplot as plt
import statsmodels.api as sm # рисовалка qq-plot.
import seaborn as sns
from sklearn.metrics import mean_absolute_error, root_mean_squared_log_error, root_mean_squared_error

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
        'describe': [f"{df_copy[i].describe().reindex(['min', 'max', 'mean', 'std']).round(2).to_string()}" if pd.api.types.is_numeric_dtype(df_copy[i].dtype) else 'see type column' for i in df_copy.columns]
        
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

#########################################

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

#########################################

def print_metrics(y_test, y_test_pred, list_of_metrics, y_train=None, y_train_pred=None, names=False, metric_params_dict=False):
    """print_metrics 

    function for print any metric u like.
    P.S. some metrics have hard names, so need to adjust names in 'block to adjust names' (see code)

    Args:
        y_test (pd.Series or numpy.ndarray): y_test
        y_test_pred (list): y_test_pred (lgr.predict(X_test))
        list_of_metrics (list): list of any metrics !!CAUTION!! Metrics should be imported like: from sklearn.metrics import accuracy_score
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
    train_list = [[] for i in range(len(list_of_metrics))]
    test_list = [[] for i in range(len(list_of_metrics))]
    
    ###################
    # if we wanna see just test metrics:
    if y_train is None:
        for y_test_p in y_test_pred:
            for i, j in enumerate(list_of_metrics):
                # values of metrics
                test_value = globals()[j](y_test, y_test_p, **metric_params_dict_func[j])
                # adding to list
                test_list[i].append(test_value)
                
                # refresh dict
                test_dict[f"{func_list_of_metrics[i]}"] = test_list[i]
        
        func_df = pd.DataFrame(test_dict, index=names).T

    
    # if we wanna see train and test scores:    
    else:
        # for train_pred + test_pred
        for y_train_p, y_test_p in zip(y_train_pred, y_test_pred):
            for i, j in enumerate(list_of_metrics):
                # values of metrics
                train_value = globals()[j](y_train, y_train_p, **metric_params_dict_func[j])
                test_value = globals()[j](y_test, y_test_p, **metric_params_dict_func[j])
                # adding to lists
                train_list[i].append(train_value)
                test_list[i].append(test_value)
                
                # refresh dicts
                train_dict[f"{func_list_of_metrics[i]}"] = train_list[i]
                test_dict[f"{func_list_of_metrics[i]}"] = test_list[i]
        
        # making MultiIndex for columns
        columns = pd.MultiIndex.from_product([['train', 'test'], names])
        
        # Combining the data into one array for further creation of a df
        data = [train_list[i] + test_list[i] for i, metric in enumerate(func_list_of_metrics)]
        
        # Creating a df with a multi-index
        func_df = pd.DataFrame(data, index=func_list_of_metrics, columns=columns)
    
    return func_df

#########################################

def add_datetime_features(df, col, dayofweek=False):
    
    func_df = df.copy()
    func_df[col] = pd.to_datetime(func_df[col], errors='coerce')
    
    func_df[col+'_date'] = func_df[col].dt.date.astype('datetime64[ns]')
    func_df[col+'_hour'] = func_df[col].dt.hour
    
    if dayofweek:
        func_df[col+'_day_of_week'] = df[col].dt.day_name()
    
    return func_df

#########################################

def add_holiday_features(df1, df2, on):
    
    func_df = df1.merge(df2, how='left', on=on)
    func_df.iloc[:, -1] = func_df.iloc[:, -1].apply(lambda x: 1 if pd.notna(x) else 0)
    
    return func_df

#########################################

def add_osrm_features(df1, df2, on):
    return df1.merge(df2, how='left', on=on)

#########################################

def lat_long_dist(matrix, rad=6.371*1e+6, azimut=False):
    """function to calculate distance between 2 points lat and long
       https://gis-lab.info/qa/great-circles.html

    Args:
        matrix (list, array, or df): any data sequence u like
        rad (float or int, optional): radius of the earth. Defaults to 6.371*1e+6 in meters!!!.
        azimut (bool, optional): the angle in decimal degrees. Defaults to False.

    Returns:
        array: the array can be shape of (m,) or (m, n) depending on what we have submitted to the input
    """
    
    if isinstance(matrix, list):
        matrix = np.array(matrix)
    elif isinstance(matrix, pd.DataFrame):
        matrix = matrix.to_numpy()
    
    # the order of sequence can be changed here
    # the order of start-end will only affect the azimuth
    # cuz for the distance there is no matter what lat or long, its the distance between 2 points
    if len(matrix.shape) == 2:
        lat_start, long_start, lat_end, long_end = matrix[:, 0], matrix[:, 1], matrix[:, 2], matrix[:, 3]
    else:
        lat_start, long_start, lat_end, long_end = matrix[0], matrix[1], matrix[2], matrix[3]
    
    # radians of lat
    f1 = lat_start * (np.pi / 180)
    f2 = lat_end * (np.pi / 180)
    
    # delta radians
    delta_lat = (lat_end - lat_start) * (np.pi / 180)
    delta_long = (long_end - long_start) * (np.pi / 180)
    
    # Haversine formula 
    a = np.power(np.sin(delta_lat / 2), 2) + np.cos(f1) * np.cos(f2) * np.power(np.sin(delta_long / 2), 2)
    haversine = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    dist = rad * haversine
    
    if not azimut:
        return dist
    else:
        # teta formula for azimut
        teta = np.arctan2(np.sin(delta_long) * np.cos(f2), np.cos(f1) * np.sin(f2) - np.sin(f1) * np.cos(f2) * np.cos(delta_long))
        
        return np.array([dist, (teta * (180 / np.pi) + 360) % 360]).T # % 360 == module 360

#########################################

def fill_null_weather_data(df):
    
    func_df = df.copy()
    
    func_df.loc[:, ['total_distance', 'total_travel_time', 'number_of_steps']] = func_df.loc[:, ['total_distance', 'total_travel_time', 'number_of_steps']].fillna(func_df.loc[:, ['total_distance', 'total_travel_time', 'number_of_steps']].median())
    func_df['events'] = func_df['events'].fillna('None')
    
    func_df.loc[:, ['pickup_datetime_date', 'temperature', 'visibility', 'wind speed', 'precip']] = func_df.loc[:, ['pickup_datetime_date', 'temperature', 'visibility', 'wind speed', 'precip']].fillna(func_df.loc[:, ['pickup_datetime_date', 'temperature', 'visibility', 'wind speed', 'precip']].groupby('pickup_datetime_date').transform('median'))
    
    return func_df

#############################################

def func_for_2_4(df, col):
    # в курсе в качестве ответа принимался диапазон, потому функция чисто для этого
    func_df = df.copy()
    
    lambda_for_index = lambda x: 'с 00.00 по 5.00' if x >= 0 and x <= 5 else\
        'с 6.00 по 12.00' if x >= 6 and x <= 12 else\
            'с 13.00 по 18.00' if x >= 13 and x <= 18 else\
                'с 18.00 по 23.00'
    func_df[col] = func_df[col].apply(lambda_for_index)
    return func_df

#############################################

def swarm_hist(df, targ_col, depend_col, figsize=(15, 5)):
    
    # сырая, есть что добавить в гиперпараметры
    
    fig, ax = plt.subplots(ncols=2, figsize=figsize)
    
    # Построение графиков beeswarm + hist
    for i, j in enumerate(df[depend_col].unique()[::-1]):
        y = df[df[depend_col] == j]
        x = np.random.normal(i + 1, 0.1, size=len(y)) # 0.1 отвечает за расстояние между
        ax[0].plot(x, y[targ_col], '*', label=f'{depend_col} {j}', alpha=0.6)
        
        # +i * 12 - чтобы башенки гистограмм не накладывались друг на друга (12 - отвечает за расстояние между)
        sns.histplot(y[targ_col] + i * 12, ax=ax[1], kde=False, 
                     label=f'{depend_col} {j}', binwidth=0.2, color=f"C{i}", alpha=0.5)
    
    ax[0].set_xticks(df[depend_col].unique()[::-1])
    ax[0].set_xlabel(f'{depend_col.capitalize()}')
    ax[0].set_ylabel(f'{targ_col.capitalize()}')
    ax[0].set_title(f'Beeswarm plot of {targ_col.capitalize()} by {depend_col.capitalize()}')
    ax[0].legend()
    
    ax[1].set_title(f'Histogram of {targ_col.capitalize()} by {depend_col.capitalize()}')
    ax[1].set_xticks([])
    ax[1].set_xlabel(f'{targ_col.capitalize()}')
    ax[1].set_ylabel('Count')
    ax[1].legend()
    
    return plt