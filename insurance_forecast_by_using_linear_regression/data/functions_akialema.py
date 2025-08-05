import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import HTML
from itertools import groupby
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error

def my_info(df, catmaxnum=None):
    """Info function to show basic df indicators

    Args:
        df (pd.Dataframe): any df.
        catmaxnum (int): the minimum threshold(rus: порог) after which a column will be considered a category. Defaults to None.

    Returns:
        _type_: pd.Dataframe
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
        
    }).sort_values(by='num of unique vals').reset_index(drop=True)
    
    if catmaxnum:
        func_df.insert(3, 'classification', ['category' if i < catmaxnum else 'numeric' for i in func_df['num of unique vals']])
    
    print(f"memory usage: {str(round(df_copy.memory_usage(deep=True).sum() / 1000, 1)) + ' KB'}")
    
    # Add CSS style to left-align the "describe" column
    html_output = func_df.to_html().replace('\\n', '<br>')
    html_output = html_output.replace('<td>min', '<td style="text-align: left;">min:...')
    html_output = html_output.replace('<br>max', '<br style="text-align: left;">max:..')
    html_output = html_output.replace('<br>mean', '<br style="text-align: left;">mean:')
    html_output = html_output.replace('<br>std', '<br style="text-align: left;">std:....')
    
    return HTML(html_output)

##############################

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

##############################

def add_line(ax, xpos, ypos):
    line = plt.Line2D([xpos, xpos], [ypos + .1, ypos],
                      transform=ax.transAxes, color='gray')
    line.set_clip_on(False)
    ax.add_line(line)

def label_len(my_index,level):
    labels = my_index.get_level_values(level)
    return [(k, sum(np.fromiter((1 for i in g), dtype=int))) for k, g in groupby(labels)]

def label_group_bar_table(ax, df):
    ypos = -.1
    scale = 1./df.index.size
    for level in range(df.index.nlevels)[::-1]:
        pos = 0
        for label, rpos in label_len(df.index, level):
            lxpos = (pos + .5 * rpos)*scale
            ax.text(lxpos, ypos, label, ha='center', transform=ax.transAxes)
            add_line(ax, pos*scale, ypos)
            pos += rpos
        add_line(ax, pos*scale ,ypos)
        ypos -= .1
    
##############################

def plot_train_test_split_balance(X_train, X_test, y_train, y_test, startangle=0):
    
    func_df = pd.DataFrame([list(X_train.shape), list(X_test.shape), list(y_train.shape), list(y_test.shape )],
                            index=['X_train', 'X_test', 'y_train', 'y_test'],
                            columns=['num_of_rows', 'num_of_cols'])
    
    def absolute_value_generator(df, targcol, sort=False, val_type='float', val_las_symbol=''):
        # тонкая дичь, чисто ради забавы. 
        # sort для проверки баланса при валидации. 
        # val_type для печати числа: либо инт, либо нет. 
        # val_last_symbol для печати последнего символа, если нужен.
        labels = df.index if not sort else df.sort_values(by=targcol).index
        return lambda val, idx=iter(range(len(labels))): f"{labels[next(idx) % len(labels)]}\n---------\n{int(round(val / 100. * sum(df[targcol]), 0)) if val_type=='int' else float(round(val / 100. * sum(df[targcol]), 2))}{val_las_symbol}"
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.pie(func_df['num_of_rows'].sort_values(),
           autopct=absolute_value_generator(df=func_df, targcol='num_of_rows', sort=True, val_type='int'),
           explode=[0.01, 0.01, 0.01, 0.01],
           textprops={'color': 'white', 'weight': 'bold', 'fontsize': 9}, startangle=startangle)
    ax.set_title('Баланс выборок')
    return plt