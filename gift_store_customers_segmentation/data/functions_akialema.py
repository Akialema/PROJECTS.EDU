import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

from IPython.display import HTML

from sklearn import metrics
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from sklearn.mixture import GaussianMixture

from collections import Counter

import copy

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

#####################################

def good_print(text, answer):
    print(f"{text}\n{'-'*(len(text)-1)}\nОтвет: {answer}\n")
    
#####################################

def get_quantity_canceled_fast(data):
    # Инициализируем нулями Series той же длины, что и столбцы таблицы
    quantity_canceled = pd.Series(np.zeros(data.shape[0]), index=data.index)
    
    # Группируем данные заранее для оптимизации
    data_grouped = data.groupby(['CustomerID', 'StockCode'])
    
    # Получаем все возвраты
    negative_quantity = data[data['Quantity'] < 0]
    
    for index, col in negative_quantity.iterrows():
        try:
            # Получаем группу для текущей пары CustomerID-StockCode
            group = data_grouped.get_group((col['CustomerID'], col['StockCode']))
            
            # Создаём DataFrame из всех транзакций, противоположных возвратам
            df_test = group[(group.index != index) &  # Исключаем текущую транзакцию
                            (group['InvoiceDate'] < col['InvoiceDate']) & 
                            (group['Quantity'] > 0)]
            # Транзация-возврат не имеет противоположной — ничего не делаем
            if df_test.empty:
                continue
            # Транзакция-возврат имеет только одну противоположную транзакцию
            # Добавляем количество возвращённого товара в столбец QuantityCanceled     
            elif len(df_test) == 1:
                index_order = df_test.index[0]
                quantity_canceled.loc[index_order] = -col['Quantity']
            # Транзакция-возврат имеет несколько противоположных транзакций
            # Вносим количество возвращённого товара в столбец QuantityCanceled для той транзакции на покупку,
            # в которой количество товара > (-1) * (количество товаров в транзакции-возврате)    
            else:
                df_test = df_test.sort_index(ascending=False)
                for ind, val in df_test.iterrows():
                    if val['Quantity'] < -col['Quantity']:
                        continue
                    quantity_canceled.loc[ind] = -col['Quantity']
                    break
                    
        except KeyError:
            continue
            
    return quantity_canceled

#####################################

def rfm_anlysis(df, rfm_cols=['CustomerID', 'InvoiceNo', 'InvoiceDate', 'TotalPrice']):
    """rfm_anlysis 

    Попытка сделать функцию, реализующую RFM таблицу.
    Порядок столбцов важен!
    Первым идёт столбец с id клиента,                                                                                               (индекс 0)
    вторым - номер заказа,                                                                                                          (индекс 1)
    третьим - дата,                                                                                                                 (индекс 2)
    последним - общая сумма покупки, которая изначально рассчитывается: ("кол-во покупок" - "кол-во возвратов") * "цена за единицу" (индекс 3)

    Args:
        df (pd.dataframe): incoming df
        rfm_cols (list, optional): columns for rfm analysis. Defaults to ['CustomerID', 'InvoiceNo', 'InvoiceDate', 'TotalPrice'].

    Returns:
        pd.dataframe: rfm table 
    """
    
    # Определяем дату анализа (следующий день после последней транзакции)
    analysis_date = (df[rfm_cols[2]].max() + pd.Timedelta(days=1)).normalize()
    # Группируем данные по клиентам и считаем метрики
    rfm = df.groupby(rfm_cols[0]).agg({
        rfm_cols[2]: lambda x: (analysis_date - x.max()).days, # Recency
        rfm_cols[1]: 'nunique',                                # Frequency
        rfm_cols[3]: 'sum'                                     # Monetary
    })
    
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    
    return rfm

#####################################

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
    # import nessessary lib for copy incoming alg.
    
    
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

#####################################

def plot_cluster_profile(grouped_data: pd.DataFrame, n_clusters: int, scaler, width: int = 800, height: int = 800, title: str = 'Cluster Profiles') -> None:
    """plot_cluster_profile
    
        Polar chart

    Args:
        grouped_data (pd.DataFrame): DataFrame containing cluster features
        n_clusters (int): Number of clusters to visualize
        scaler (scaler.type): Scaler object (like MinMaxScaler or StandardScaler)
        width (int, optional): Plot width in pixels. Defaults to 800.
        height (int, optional): Plot height in pixels. Defaults to 800.
        title (str, optional): Plot title. Defaults to 'Cluster Profiles'.

    Raises:
        TypeError: grouped_data must be a pandas DataFrame
        ValueError: n_clusters must be positive
        AttributeError: scaler must have fit_transform method
        
    Returns:
        fig
    """
    
    # Проверки входных данных
    if not isinstance(grouped_data, pd.DataFrame):
        raise TypeError("grouped_data must be a pandas DataFrame")
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive")
    if not hasattr(scaler, 'fit_transform'):
        raise AttributeError("scaler must have fit_transform method")
        
    # Нормализация данных
    normalized_data = pd.DataFrame(scaler.fit_transform(grouped_data),
                                   columns=grouped_data.columns)
    
    # Создание фигуры
    fig = go.Figure()
    
    # В цикле визуализируем полярную диаграмму для каждого кластера
    for i in range(n_clusters):
        fig.add_trace(go.Scatterpolar(r=normalized_data.iloc[i].values, # радиусы
                                      theta=normalized_data.columns, # название засечек
                                      fill='toself', # заливка многоугольника цветом
                                      name=f'Cluster {i}', # название — номер кластера
                                      hovertemplate='%{theta}: %{r:.2f}<extra>Cluster %{customdata}</extra>', # корректировка отображения всплывающих окон
                                      customdata=[i]*len(normalized_data.columns)))
    
    # Настройка layout
    fig.update_layout(title=title,
                      showlegend=True,
                      autosize=False,
                      width=width,
                      height=height,
                      polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
    return fig

#####################################

def plot_train_test_split_balance(X_train, X_test, y_train, y_test, startangle=0):
    
    """ Pie diagramm of validation distribution

    Returns:
        plt: plt
    """
    
    func_df = pd.DataFrame([list(X_train.shape), list(X_test.shape), list(y_train.shape), list(y_test.shape )],
                            index=['X_train', 'X_test', 'y_train', 'y_test'],
                            columns=['num_of_rows', 'num_of_cols'])
    
    def absolute_value_generator(df, targcol, sort=False, val_type='float', val_las_symbol=''):
        # тонкая дичь, чисто ради забавы. 
        # sort для проверки баланса при валидации. 
        # val_type для печати числа: либо инт, либо нет. 
        # val_last_symbol для печати последнего символа, если нужен.
        labels = df.index if not sort else df.sort_values(by=targcol).index
        return lambda val, idx=iter(range(len(labels))): f'{labels[next(idx) % len(labels)]}\n---------\n{int(round(val / 100. * sum(df[targcol]), 0)) if val_type=="int" else float(round(val / 100. * sum(df[targcol]), 2))}{val_las_symbol}'
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.pie(func_df['num_of_rows'].sort_values(),
           autopct=absolute_value_generator(df=func_df, targcol='num_of_rows', sort=True, val_type='int'),
           explode=[0.01, 0.01, 0.01, 0.01],
           textprops={'color': 'white', 'weight': 'bold', 'fontsize': 9}, startangle=startangle)
    ax.set_title('Баланс выборок')
    return plt

#####################################

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
                # get incoming metric name
                metric_func = getattr(metrics, j)
                # values of metrics
                test_value = metric_func(y_test, y_test_p, **metric_params_dict_func[j])
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
                # get incoming metric name
                metric_func = getattr(metrics, j)
                # values of metrics
                train_value = metric_func(y_train, y_train_p, **metric_params_dict_func[j])
                test_value = metric_func(y_test, y_test_p, **metric_params_dict_func[j])
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

#####################################

def to_markdown_table(a, b):
    max_a = max(len(str(i)) for i in a + ['Вопрос'])
    max_b = max(len(str(i)) for i in b + ['Ответ'])

    row_template = f"| {{:<{max_a}}} | {{:<{max_b}}} |"
    separator = f"|{'-' * (max_a + 2)}|{'-' * (max_b + 2)}|"

    lines = [row_template.format("Вопрос", "Ответ"), separator]
    for q, ans in zip(a, b):
        lines.append(row_template.format(str(q), str(ans)))
    
    return "\n".join(lines)