import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from itertools import groupby
from IPython.display import HTML
import statsmodels.api as sm

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

###################################

# Метод работы с выбросами от Тьюки. Всё по курсу. Только if else - отсебятина.
def outliers_iqr_mode_log(data, feature, log_scale=False, left=1.5, right=1.5):
    if log_scale:
        if any(data[feature] == 0):
            x = np.log(data[feature] + 1)
        else:
            x = np.log(data[feature])
    else:
        x = data[feature]
    
    quantile_25, quantile_75 = x.quantile(0.25), x.quantile(0.75)
    iqr = quantile_75 - quantile_25
    
    lower_bound = quantile_25 - iqr * left
    upper_bound = quantile_75 + iqr * right
    
    outliers = data[(x < lower_bound) | (x > upper_bound)]
    cleaned = data[(x >= lower_bound) & (x <= upper_bound)]
    
    return outliers, cleaned.reset_index(drop=True).copy() #если вот здесь не написать copy(), оно будет делать мозги потом.

###################################

def hist_box_qq(arg, full_iqr=False):
    """Функция рисования гистограммы, коробки и кукушки в одном флаконе.

    full_iqr... это короче.. я писал функцию под логнормальное распределение, т.е. слева выбросов нет.
    Но потом подумал и решил добавить, потому что я потом всё же логарифмирую признак.
    Да и в целом, если массив обычно распределён.
    Одним словом: full_iqr - булевое. По умолчанию False. Т.е. рисуются только правые границы.
    
    На вход подаётся Series или массив.

    Args:
        arg (_type_): Series/np.array
        full_iqr (bool, optional): choose, u wanna show only right outliers borders or both. Defaults to False.

    Returns:
        plt.show()
    """
    # Если есть нанки - дропаю. Т.к. не рисует, если есть нанки.
    arg = pd.Series(arg).dropna()
    
    fig, ax = plt.subplots(ncols=3, nrows=1, figsize=[18, 5])
    
    # Гистограмма. И настройки для линий гистограммы. Легенда = label в каждой строчке.
    # Красная - медиана, зеленая - тьюки, синяя - 3 сигмы.
    sns.histplot(arg, kde=True, bins=50, ax=ax[0])
    
    ax[0].get_lines()[0].set_color('black')
    ax[0].axvline(arg.median(), color='red', linestyle='--', linewidth='1.8', label='median')
    ax[0].axvline(arg.quantile(0.75) + ((arg.quantile(0.75) - arg.quantile(0.25)) * 1.5), color='g', ls='--', lw=2, label='1.5 IQR Tukie')
    ax[0].axvline(arg.mean() + 3 * arg.std(), color='b', ls='--', lw=2, label='3 IQR z-score')
    ax[0].legend()
    ax[0].set_xlabel(f'Признак {arg.name}')
    
    # Коробка. И настройки для неё.
    sns.boxplot(arg, ax=ax[1], orient='h', medianprops={'color': 'red', 'linestyle': '--'})

    ax[1].axvline(arg.quantile(0.75) + ((arg.quantile(0.75) - arg.quantile(0.25)) * 1.5), color='g', ls='--', lw=2, label='1.5 IQR Tukie')
    ax[1].axvline(arg.mean() + 3 * arg.std(), color='b', ls='--', lw=2, label='3 IQR z-score')
    ax[1].legend()    
    ax[1].set_xlabel(f'Признак {arg.name}')
    
    if full_iqr: # Добавляются левые линии отлова выбросов.
        ax[0].axvline(arg.quantile(0.25) - ((arg.quantile(0.75) - arg.quantile(0.25)) * 1.5), color='g', ls='--', lw=2)
        ax[0].axvline(arg.mean() - 3 * arg.std(), color='b', ls='--', lw=2)
        ax[1].axvline(arg.quantile(0.25) - ((arg.quantile(0.75) - arg.quantile(0.25)) * 1.5), color='g', ls='--', lw=2)
        ax[1].axvline(arg.mean() - 3 * arg.std(), color='b', ls='--', lw=2)
    
    # Эта строчка строит Q-Q график.
    qq = sm.ProbPlot(arg, fit=True).qqplot(marker='*', markerfacecolor='b', markeredgecolor='b', alpha=0.3, ax=ax[2])
    # Линия, тоже с возможностью влиять на цвет. Параметр fmt.
    sm.qqline(qq.axes[2], line='45', fmt='r', linestyle='--')
    
    # Общий заголовок.
    plt.suptitle(f'Распределение признака {arg.name}').set_fontsize(20)
    
    return plt

###################################

# Чумовейшие три функции, которые рисуют мультииндекс разметку по оси Х!!!
# Работа происходит по третьей функции. 
# На вход подаётся ax и датафрейм вида: 
# 1. главный индекс; 
# 2. второстепенный индекс или группа индексов, по иерархии от старшего к младшему; 
# 3. последним идёт индекс, который является оттенком (hue).
# Вид группировки: .groupby(['1_st_order_idx', '2_nd_order_idx', ..., 'hue_idx'])['nums_column'].mean().unstack()
# P.S. для работы нужен groupby из itertools

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

###################################

# Функция проверки на нормальность Шапиро-Уилка.
def shapirowilk(df_or_arr, alpha=0.05):
    """Продвинутая функция проверки нормальности Шапиро-Уилка.
    
    Тут всё просто. Если датафрейм, то формируем через цикл списки названий нормальных и не нормальных столбцов.
    Иначе сразу печатаем реультат теста. Т.к. у нас Series или массив.
    Далее смотрим на длины списков и формируем список выводов функции. И выводим его.
    Почему продвинутая? Потому что я раньше считал количество норм и не норм столбцов через: dict(Counter([shapirowilk(df_for_this_cell[i]) for i in df_for_this_cell.columns]))
    а сама функция выглядела, как функция Левена ниже.))

    Args:
        df_or_arr (_type_): Dataframe or Series or array

    Returns:
        _type_: list or str
    """
    func_list = []
    norm_list = []
    abnorm_list = []
    if isinstance(df_or_arr, pd.DataFrame):
        for i in df_or_arr.columns:
            if stats.shapiro(df_or_arr[i])[1] <= alpha:
                abnorm_list.append(df_or_arr[i].name)
            else:
                norm_list.append(df_or_arr[i].name)
    else:
        return 'Распределение не нормальное. При alpha = 0.05' if stats.shapiro(df_or_arr)[1] <= alpha else 'Распределение нормальное. При alpha = 0.05'
    
    if len(norm_list) > 0:
        func_list.append(f'Распределение {norm_list} нормальное. При alpha = 0.05')
    if len(abnorm_list) > 0:
        func_list.append(f'Распределение {abnorm_list} не нормальное. При alpha = 0.05')
    return func_list