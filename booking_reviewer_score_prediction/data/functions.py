import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2 # хи-квадрат
from sklearn.feature_selection import f_classif # anovav
from sklearn import metrics

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor

import time

from IPython.display import HTML

from itertools import groupby

from colorama import Fore, Back, Style
from tqdm.notebook import tqdm

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

# --------------------------------------------------------------------------------------------------------------------------------------

# Tukie outliers finder function.
def outliers_iqr_mode_log(data, feature=None, log_scale=False, left=1.5, right=1.5):
    
    func_data = data.copy()
    
    if isinstance(func_data, pd.DataFrame):    
        if log_scale: # if we want to logarithm a feature.
            if any(func_data[feature] == 0):
                x = np.log(func_data[feature] + 1)
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
    
    return outliers, cleaned.reset_index(drop=True)

# --------------------------------------------------------------------------------------------------------------------------------------

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

# --------------------------------------------------------------------------------------------------------------------------------------

def my_corr(df, method='spearman', corrneg=0, corrpos=0, figsize=(10, 5), xtickrot=90, show_high_corr_cols_names=False):
    """
    The function of drawing a correlation graph, but not an ordinary one, but one in which the lower part is the entire one,
    and the upper part contains only those correlation values that are greater and less than some number we have specified.

    Args:
        df (pd.Dataframe): any df.
        method (str): method u like. default spearman.
        corrneg (float): negative threshold for display. (for negative correlation value)
        corrpos (float): positive threshold. (for a positive value) both default to 0. At 0, the full standard correlation matrix will be displayed.
        figsize (tuple): u know what it is.)
        xtickrot(int): rotate x axis labels. default 90
        show_high_corr_cols_names(bool): if we wanna watch names of high corr cols. default False

    Returns:
        _type_: plt or df
    """
    corr_df = df.corr(numeric_only=True, method=method)
    
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
        sns.heatmap(corr_df, annot=True, cmap='coolwarm', mask=mask_for_logging_df, vmax=1, vmin=-1, linewidths=0.1, fmt='.2f') # <--- lower part
        
        # if condition == 1 high_corr_features contains only nan, so we dont need upper part. 
        # or.. high_corr_features has no nan, i.e. corrpos and corrneg == 0, so we draw all corr matrix.
        if (len(high_corr_features.isna().sum().value_counts()) != 1) | ((corrpos == 0) & (corrneg == 0)):
            sns.heatmap(high_corr_features, annot=True, cmap='coolwarm', mask=mask_for_logging_df_2, cbar=False, vmax=1, vmin=-1, linewidths=0.1, fmt='.2f') # <--- upper part
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=xtickrot) # rotation of xticks
    
        plt.plot([0, len(df.columns)], [0, len(df.columns)], color='black', linewidth=2); # It's just a line. I don't know why. But i like it.
        
        return plt

# --------------------------------------------------------------------------------------------------------------------------------------

def label_group_bar_table(ax, df):
    """label_group_bar_table 

    Чумовейшие три функции, которые рисуют мультииндекс разметку по оси Х!!!
    На вход подаётся ax и датафрейм вида: 
     1. главный индекс; 
     2. второстепенный индекс или группа индексов, по иерархии от старшего к младшему; 
     3. последним идёт индекс, который является оттенком (hue).
     Вид группировки: .groupby(['1_st_order_idx', '2_nd_order_idx', ..., 'hue_idx'])['nums_column'].mean().unstack()
     P.S. для работы нужен groupby из itertools
     P.S.2 Сие ваяние не моё, потому хз, что там происходит и я не распишу каждую строчку, но результат меня устраивает.)

    Args:
        ax (_type_): ax with ur plot.
        df (_type_): ur df which u wanna plot.
    """
    
    def add_line(ax, xpos, ypos):
        line = plt.Line2D([xpos, xpos], [ypos + .1, ypos],
                          transform=ax.transAxes, color='gray')
        line.set_clip_on(False)
        ax.add_line(line)
    
    def label_len(my_index,level):
        labels = my_index.get_level_values(level)
        return [(k, sum(np.fromiter((1 for i in g), dtype=int))) for k, g in groupby(labels)]
    
    
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
        
# --------------------------------------------------------------------------------------------------------------------------------------

def f_classif_chi_tester(series, targ_col):
    """f_classif_chi_tester _summary_

    Функция.. проводит тестирование f_classifom и хи квадратом. И показывает число. Чем больше - тем лучше.
    Хи квадрат какой-то левый тест. Многомилионные показатели никак не влияют на модель. Как будто там что-то ломается.
    Но не суть. f_classif более правдивый.

    Args:
        series (pd.Series/pd.DataFrame): incoming series num only. Or if its df
        targ_col (pd.Series): looks like your_df['your_targ_col']

    Returns:
        _type_: series or df with test scores.
    """
    func_series_or_df = series.copy()
    
    # if series
    if isinstance(func_series_or_df, pd.Series):
        chi = pd.Series(chi2(func_series_or_df.to_frame().abs(), targ_col.astype(int))[0], index=['chi']) # abs for negative nums
        f = pd.Series(f_classif(func_series_or_df.to_frame(), targ_col.astype(int))[0], index=['f'])
    
    else: # if df
        if targ_col.name in func_series_or_df.columns: # drop targ_col if it exists.
            func_series_or_df.drop(columns=targ_col.name, inplace=True)
        chi = pd.Series(chi2(func_series_or_df.abs(), targ_col.astype(int))[0], index=func_series_or_df.columns, name='chi')
        f = pd.Series(f_classif(func_series_or_df, targ_col.astype(int))[0], index=func_series_or_df.columns, name='f_classif')
        
        return pd.DataFrame([f, chi]).T
    return pd.concat([f, chi])

# --------------------------------------------------------------------------------------------------------------------------------------

def special_grouper(df, bywhat, what, list_of_agg_func=['sum', 'mean', 'count', 'nunique']):
    """special_grouper 

    Согласитесь, бесит, когда надо писать несколько строчек, чтобы группировка применилась ко всей таблице.
    А метод transform принимает только 1 аргумент и ему плевать список там или нет. Хотя в документации там может быть список.
    Этот момент я до конца не понял.
    
    Эта функция - исключительно для визуальной оценки, а потом мы делаем нужный нам сериес с выбранной агрегацией через transform.
    Т.к. трансформ крутая штука на самом деле, туда можно пихнуть лямбду.
    
    Хотя и отсюда можно взять столбец, но через .loc/iloc, потому что функция возвращает тип датафрейм.

    Args:
        df (pd.DataFrame): any df.
        bywhat (str or list): name or names of columns which we want to group.
        what (str): name of col we wanna group bywhat (my logic works only for single col cuz .iloc[bla bla] (can be expanded))
        list_of_agg_func (list, optional): list of agg func. Defaults to ['sum', 'mean', 'count', 'nunique'].

    Returns:
        pd.DataFrame: a column or columns with grouping applied to the entire df as a df.
    """
    return pd.merge(df[bywhat], df.groupby(bywhat)[what].agg(list_of_agg_func).reset_index(), on=bywhat, how='left').iloc[:, -len(list_of_agg_func):]

# Вишенка-------------------------------------------------------------------------------------------------------------------------------

def multiple_models_training(df, target_feature, list_of_ml_algs, params_dict=False, random_state=42, test_size=0.25):
    print(f'{Fore.RED}КОРОЛЬ АРТУР НА НАС НАПАЛИ!!!{Fore.RESET}') # https://www.youtube.com/watch?v=-XuyTXj_5F8&t=6s хз, чёт вспомнилось...
    """ 
    Так, я прекрасно понимаю, что там, в каждом алгоритме, стопицот тыщ мильнов всяких методов, атрибутов и прочих плюшек.
    В рамках данного проекта я решил накалякать цикл, чтоб не плодить простынку хтестов-хтрейнов,
    потому что хочу картинку красивую в конце.
    Скажу сразу, "по полной" я эту функцию не гонял, параметры фита не ставил. В рамках данного проекта мне лично хватило.)
    
    Функция проведения нескольких (или одной) тренировок моделей.
    
    ---> На вход подаём итоговый датафрейм, в котором только числа и нет нанок, 
         целевой признак,
         список названий алгоритмов для тренировки,
         и словарь с параметрами для алгоритмов.

    Args:
        df (pd.Dataframe): итоговый дф.
        target_feature (str): целевой признак.
        list_of_ml_algs (list): список названий Агрессоров, каждый из них в str формате.)) (агрессоров может быть как 1, так и сколько душе угодно, но важно, чтобы они соответствовали структуре "инит класса -> тренировка -> предсказание").
        params_dict (dict, optional): словарь вида {alg: [{параметры алгоритма}, {параметры фита}]}, по которому создаётся датафрейм с параметрами. Первый внутренний словарь для параметров алгоритма, второй - для фита. Defaults to False.
        random_state (int, optional): я хз, что это, я так до сих пор и не прочувствовал np.random.seed. Defaults to 42.
        test_size (float, optional): размер тестовой выборки. Defaults to 0.25.

    Returns:
        pd.Dataframe: На выход получаем датафрейм с натренированными моделями, числом МАРЕ и чем-нибудь ещё.
        
    Пример работы. Нужно скопировать код ниже и вставить в ячейку после функции.
    ------------------------------------------------------------------------------------------------------
    # Здесь нужно выбрать, какими алгоритмами будем гонять числа.)
    list_of_ml_algs = ['RandomForestRegressor', 'ExtraTreesRegressor', 'BaggingRegressor', 'GradientBoostingRegressor', 'CatBoostRegressor', 'AdaBoostRegressor', 'DecisionTreeRegressor']
    
    # Параметры для каждого алгоритма. Порядок не важен, как и указание параметров для всех алгоритмов. 
    # Можно указать параметры только для тех, для которых нужно, остальные отработают по своим дефолтным параметрам.
    # Важна структура {alg: [{параметры алгоритма}, {параметры фита}]}, иначе можно хлебнуть горя.)) Индекс-столбец-столбец.
    # А можно и не делать этот словарь вовсе. Алгоритмы отработают по своим настройкам по-умолчанию.
    params_dict = {
            'RandomForestRegressor': [{'n_estimators': 10, 'n_jobs': -1, 'random_state': 42}, {}],
            'ExtraTreesRegressor': [{'n_estimators': 10, 'max_depth': 3}, {}],
            'BaggingRegressor': [{'n_estimators': 10}, {}],
            'GradientBoostingRegressor': [{'n_estimators': 10, 'learning_rate': .1}, {}],
            'CatBoostRegressor': [{'iterations': 1000, 'learning_rate': .1, 'depth': 3, 'verbose': False}, {}],
            'AdaBoostRegressor': [{'n_estimators': 10, 'learning_rate': 1.0}, {}],
            'DecisionTreeRegressor': [{}, {}]
    } # везде эстиматоры 10, чтоб оно отработало быстро и показало работу. 
    
    # Здесь нужно вставить свой end_df, свой 'целевой_признак', решить, какими алгоритмами будем баловаться и с какими параметрами.)
    any_var_name = multiple_models_training(end_df, 'reviewer_score', list_of_ml_algs=list_of_ml_algs, params_dict=params_dict)
    
    any_var_name
    ------------------------------------------------------------------------------------------------------
    
    Нужно быть внимательным/ной, чтоб всё сработало, как надо. Я вроде всё расписал.
    """
    # Если мы передаём параметры словарём извне.
    if params_dict:
        params_dict_func = {alg: [{}, {}] for alg in list_of_ml_algs} # Создаётся словарь следующей формы: {alg: [{}, {}]}, для каждого элемента списка с названиями алгоритмов.
        params_dict_func.update(params_dict) # Обновление словаря словарём извне.
        params_df = pd.DataFrame.from_dict(params_dict_func, orient='index', columns=['alg_params', 'fit_params']) # ДФ по словарю.
    # Если мы не передаём параметры в функцию, то все параметры будут по умолчанию для каждого алгоритма.
    else:
        params_df = pd.DataFrame([[{},{}]], index=list_of_ml_algs, columns=['alg_params', 'fit_params'])
    
    X = df.drop(columns=[target_feature], axis=1) # дропаем целевой признак из икса.
    y = df[target_feature] # и помещаем его в игрек.
    
    # Эти штуки.. Я надеюсь они в правильном порядке стоят. Вроде бы у всех так.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    list_of_algs_output = [] # список по которому будем формировать выходящий датафрейм.
    
    total_progress = 100  # Общий прогресс прогресс бара от 0 до 100%
    
    # Инициализация прогресс бара с настройками отображения.
    with tqdm(total=total_progress, desc=f"Cell progress: ", bar_format='{desc}: {percentage:.0f}%|{bar}| {n:.0f}/{total_fmt} [Elapsed: {elapsed} m/s]') as pbar:
        
        cumulative_progress = 0 # Чтобы прогресс бар работал, как надо и завершался всегда на 100.
        
        # Цикл по списку алгоритмов.
        for i, alg in enumerate(list_of_ml_algs): # i пригодился в вычислении шага прогрессбара.
            
            start_alg_time = time.time() # Время начала работы обучения модели.
            
            # Инит алгоритма через globals, т.к. название алгоритма == str. (можно через eval(), но гпт чату не понравилась эта идея) 
            # Параметры == параметры столбца 'class_params' для строки с именем alg.
            model = globals()[alg](**params_df.loc[alg, 'alg_params']) 
            
            model.fit(X_train, y_train, **params_df.loc[alg, 'fit_params']) # Здесь сама тренировка. Параметры из столбца 'fit_params'.
            
            current_time = time.time() # Время окончания работы алгоритма.
            
            y_pred = model.predict(X_test) # Предсказание.
            
            # Делаем МАПЕ.
            mape = metrics.mean_absolute_percentage_error(y_test, y_pred) * 100
            # Я не знаю, что это и для чего это здесь. Просто добавил, чтобы убедиться, что в вывод можно ещё чего-нибудь прикрутить.
            # Соответственно, нужно обновлять аппенд и формирование столбцов итогового дф.
            accuracy = model.score(X_test, y_test) 
            
            list_of_algs_output.append([alg, mape, accuracy, model]) # Добавляем всю эту петрушку в список, по которому будет формироваться таблица.
            
            ##########################################################################################################################
            
            # Вычисление времени для принта ниже. Типа вот, мол, модель обучилась, поздравляю.
            execution_time_alg = current_time - start_alg_time
            # Можно прикрутить, что-то другое для принта.
            print(f"Model training completed! With: {str(round(execution_time_alg, 2)) + ' sec.' if execution_time_alg < 60 else str(round(execution_time_alg/60, 2)) + ' min.'} Alg: {Fore.CYAN}{alg}{Fore.RESET}.") # чтоб секунды были секундами, а минуты - минутами.
            time.sleep(0.2) # Сон я добавил, потому что сначала прогресс бар отрабатывает, а потом принт. А надо наоборот.)
            
            # Рассчитываем размер шага на основе оставшегося прогресса и оставшихся итераций.
            remaining_iterations = len(list_of_ml_algs) - i # вот тут пригодился индекс i
            remaining_progress = total_progress - cumulative_progress
            step_size = remaining_progress / remaining_iterations
            
            # Обновляем общий прогресс. Все округления происходят при инициализации прогрессбара.
            cumulative_progress += step_size
            pbar.update(step_size)
        
    func_df = pd.DataFrame(list_of_algs_output, columns=['alg_name', 'mape_percent', 'accuracy', 'model']).sort_values(by='mape_percent').reset_index(drop=True) # ДФ-чик 
    
    print(f'{Fore.GREEN}Mission accomplished{Fore.RESET}.')
    # Возвращаем дф, в котором мапе и сама модель. 
    return func_df

# --------------------------------------------------------------------------------------------------------------------------------------

def fillna_dict_for_custom_df(df, func):
    """Function for generating a dictionary for the fillna function argument.
    
    The function takes as input a dataframe with np.nan and the function that we want to aggregate.
    That is, we have a feature in which there is a nan. We want to fill the nan, for example, with the median.
    We run through the names of the columns that have nan and assign each nan a median value for the column.
    Why the condition? When we ask for mode, we get a Series object that needs to be accessed by index to get the value.
    And when we ask for the average or median, it immediately returns the value.

    Args:
        df (pd.Dataframe): any df
        func (str): any func in str format. 
        
        Ex: df.fillna(value=fillna_dict_for_custom_df(df, 'mean'))

    Returns:
        _type_: dict
    """
    return {i: df[i].agg(func)[0] if func == 'mode' else df[i].agg(func) for i in df.loc[:, df.isna().mean() > 0].columns}