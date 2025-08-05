import pandas as pd
import numpy as np
from IPython.display import HTML
import matplotlib.pyplot as plt

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

###################################

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