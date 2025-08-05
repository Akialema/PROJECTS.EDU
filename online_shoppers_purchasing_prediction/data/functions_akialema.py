import pandas as pd
import numpy as np
from IPython.display import HTML
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn import metrics

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

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
        
    }).sort_values(by='num of unique vals').reset_index(drop=True)
    
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

####################################################################

def my_corr(df, targ_col=False, method='pearson', corrneg=0, corrpos=0, figsize=(10, 5), xtickrot=90, annot_kws={'size': 'medium'}, show_high_corr_cols_names=False):
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
    
####################################################################

def plot_learning_curve(model, X, y, cv, scoring='f1', ax=None, title=''):

    # Вычисляем координаты для построения кривой обучения
    train_sizes, train_scores, valid_scores = model_selection.learning_curve(estimator = model, # модель
                                                                             X = X, # матрица наблюдений X
                                                                             y = y, # вектор ответов y
                                                                             cv = cv, # кросс-валидатор
                                                                             scoring = scoring) # метрика
    # Вычисляем среднее значение по фолдам для каждого набора данных
    train_scores = np.mean(train_scores, axis=1)
    valid_scores = np.mean(valid_scores, axis=1)
    
    # Если координатной плоскости не было передано, создаём новую
    if not ax:
        fig, ax = plt.subplots(figsize=(10, 4))
    
    ax.plot(train_sizes, train_scores, label='Train') # Строим кривую обучения по метрикам на тренировочных фолдах
    ax.plot(train_sizes, valid_scores, label='Valid') # Строим кривую обучения по метрикам на валидационных фолдах
    
    ax.set_title(f'Learning curve: {title}')
    ax.set_xlabel('Train data size')
    ax.set_ylabel('Score')
    ax.xaxis.set_ticks(train_sizes) # Устанавливаем отметки по оси абсцисс
    ax.set_ylim(0, 1) # Устанавливаем диапазон оси ординат
    ax.legend()
    
    return plt
    
####################################################################

def pr_curve(y, y_proba_pred, ax=None, title='', toggle_pic_or_threshnum='pic'):
    """pr_curve 
    
    Графическая функция для отображения кривой precision-recall с возможностью выводить либо график, либо наилучший порог.

    Args:
        y (pd.Series): y_train or any u like.
        y_proba_pred (numpy.ndarray): proba_pred.
        ax (?, optional): this is just ax. Defaults to None.
        title (str, optional): title. Defaults to ''.
        toggle_pic_or_threshnum (str, optional): _description_. Defaults to 'pic'.

    Returns:
        plt or float: u will recieve either a pic or a best threshold num.
    """
    
    # Мне нравится пандас, потому сделаю через датафрейм
    prc = pd.DataFrame(metrics.precision_recall_curve(y, y_proba_pred), index=['precision', 'recall', 'thresholds']).T
    
    # Вычисляем F1-меру при различных threshold.
    f1_scores = (2 * prc['precision'] * prc['recall']) / (prc['precision'] + prc['recall'])
    # Определяем индекс максимума.
    idx = np.argmax(f1_scores)
    
    # Условие для: либо рисуем график... 
    if toggle_pic_or_threshnum == 'pic':
        # если не указан ax
        if not ax:
            fig, ax = plt.subplots(figsize=(10, 5))
        # Линия и точка на линии
        ax.plot(prc['recall'], prc['precision'], label=f"{title} auc: {metrics.auc(prc['recall'], prc['precision']):.2f}")
        ax.scatter(prc.loc[idx, 'recall'], prc.loc[idx, 'precision'], c='black', marker='o', s=80, label='Best F1 score')
        
        # Добавляем аннотацию с меткой точки на оси X (порог и ф1_скор)
        recall_best = prc.loc[idx, 'recall'] # координаты точки
        precision_best = prc.loc[idx, 'precision']
        ax.annotate(f"Threshold: {prc.loc[idx, 'thresholds']:.2f}\nF1_score: {f1_scores[idx]:.2f}", 
                    xy=(recall_best+0.01, precision_best), 
                    xytext=(recall_best + 0.05, precision_best),  # Смещаем текст немного вправо
                    arrowprops=dict(facecolor='black', shrink=0.05, width=0.7, headwidth=7, headlength=7))
        # Метки и заголовок.
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-recall curve: {title}')
        
        plt.legend()
        
        return plt
    # либо выводим наилучший порог метрики.
    else:
        return prc.loc[idx, 'thresholds']

####################################################################

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

####################################################################

def plot_train_test_split_balance(X_train, X_test, y_train, y_test, startangle=0):
    
    """ Pie diagramm of validation distribution

    Returns:
        plt: plt
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    
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