# Copyright (C) 2022-now yui-mhcp project author. All rights reserved.
# Licenced under a modified Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pandas as pd

_base_aggregation = {
    'count' : len,
    'min'   : np.min,
    'mean'  : np.mean,
    'max'   : np.max,
    'total' : np.sum
}

def set_display_options(columns = 25, rows = 25, width = 125, colwidth = 50):
    pd.set_option('display.max_columns', columns)
    pd.set_option('display.max_row', rows)

    pd.set_option('display.width', width)
    pd.set_option('display.max_colwidth', colwidth)


def filter_df(data, on_unique = [], ** kwargs):
    """
        Filters a `pd.DataFrame`
        
        Arguments : 
            - data      : dataframe to filter
            - on_unique : column or list of columns on which to apply criterion on uniques values (see notes for details)
            - kwargs    : key-value pairs of `{column_id : criterion}`
                where criterion can be : 
                - callable (a function) : take as argument the column and return a boolean based on values
                    --> `mask = data[column].apply(fn)`
                - list / tuple  : list of possible values
                    --> `mask = data[column].isin(value)`
                - else  : expected value
                    --> `mask = data[column] == value`
        Return :
            - filtered_data : filtered dataframe
        
        Note : if `on_unique` is used and value is a callable, it is applied on the result of `data[column].value_counts()` that gives a pd.Series, where index are the unique values and the values are their respective occurences (sorted in decreasing order). 
        The function must return boolean values (useful to get only ids with a minimal / maximal number of occurences)
        You can also pass a string (min / max / mean) or an int which represents the index you want to keep (min = index -1, max = index 0, mean = len(...) // 2)
    """
    if not isinstance(on_unique, (list, tuple)): on_unique = [on_unique]
    
    for column, value in kwargs.items():
        if column not in data.columns: continue
        
        if column in on_unique:
            assert callable(value) or isinstance(value, (str, int))
            uniques = data[column].value_counts()
            if isinstance(value, str):
                if value == 'min':      uniques = [uniques.index[-1]]
                elif value == 'max':    uniques = [uniques.index[0]]
                elif value == 'mean':   uniques = [uniques.index[len(uniques) // 2]]
            elif isinstance(value, int):
                uniques = [uniques.index[value]]
            else:
                uniques = uniques[value(uniques)].index
            
            mask = data[column].isin(uniques)
        elif callable(value):
            mask = data[column].apply(value)
        elif isinstance(value, (list, tuple)):
            mask = data[column].isin(value)
        else:
            mask = data[column] == value
        
        data = data[mask]
    return data

def sample_df(data,
              on    = 'id',
              n     = 10,
              n_sample      = 10,
              min_sample    = None,
              random_state  = None,
              drop = True
             ):
    """
        Sample dataframe by taking `n_sample` for `n` different values of column `on`
        Default values means : 'taking 10 samples for 10 different ids'
        
        Arguments :
            - data  : `pd.DataFrame` to sample
            - on    : the `data`'s column to identify groups
            - n     : the number of groups to sample
            - n_sample  : the number of samples for each group (if <= 0, max samples per group)
            - min_sample    : the minimal number of samples for a group to be selected.
                Note that if less than `n` groups have at least `min_sample`, some groups can have less than `min_sample` in the final result.
            - random_state  : state used in the sampling of group's ids and samples (for reproducibility)
            - drop          : cf `drop` argument in `reset_index`, if `False`, tries to ad an `index` column
        Returns :
            - samples   : a pd.DataFrame with `n` different groups and (hopefully) at least `n_sample` for each group
        
        raise ValueError if `n` is larger than the number of groups
    """
    rnd = np.random.RandomState(random_state)

    uniques = data[on].value_counts()
    
    if n is None or n <= 0: n = len(uniques)
    if n_sample is None or n_sample <= 0: n_sample = len(data)
    
    if min_sample is not None:
        uniques = uniques[uniques >= min_sample]
    
    uniques = uniques.index
    if len(uniques) > n:
        uniques = rnd.choice(uniques, n, replace = False)
    
    indexes = []
    for u in uniques:
        samples_i = data[data[on] == u]
        
        n_sample_i = min(
            len(samples_i), n_sample
        ) if not isinstance(n_sample, float) else int(n_sample * len(samples_i))
        
        indexes.extend(rnd.choice(
            samples_i.index, size = n_sample_i, replace = False
        ))
    
    return data.loc[indexes].reset_index(drop = drop)

def aggregate_df(data, group_by, columns = [], filters = {}, merge = False, ** kwargs):
    """
        Computes some aggregation functions (e.g., `np.sum`, `np.mean`, ...) on `data`
        
        Arguments :
            - data  : the original `pd.DataFrame`
            - group_by  : the columns to group for the aggregation
            - columns   : the columns on which to apply the aggregation functions
            - filters   : mapping `{column : filter}` to apply (see `filter_df`)
            - kwargs    : mapping `{aggregation_name : aggregation_fn}`, the aggregation to perform
        Return :
            - aggregated_data   : `pd.DataFrame` with columns `group_by + list(kwargs.keys())`
        
        Example usage :
        ```python
        dataset = get_dataset('common_voice') # contains columns ['id', 'filename', 'time']
        aggregated = aggregate_df(
            dataset,                # audio dataset
            group_by    = 'id',     # groups by the 'id' column
            columns     = 'time',   # computes the functions on the 'time' column
            total   = np.sum,       # computes the total time for each 'id'
            mean    = np.mean       # computes the average time for each 'id'
        )
        print(aggregated.columns)   # ['id', 'total', 'mean']
        ```
        
        Note : if no `kwargs` is provided, the default computation is `count, min, mean, max, total`
    """
    if not isinstance(group_by, (list, tuple)): group_by = [group_by]
    if not isinstance(columns, (list, tuple)): columns = [columns]
    if len(columns) == 0: columns = [c for c in data.columns if c not in group_by]
    if len(kwargs) == 0: kwargs = _base_aggregation
    
    for k, v in kwargs.items():
        if isinstance(v, int): kwargs[k] = lambda x: x.values[v]
        elif isinstance(v, str): kwargs[k] = _base_aggregation[v]
    
    name_format = '{name}_{c}' if len(columns) > 1 else '{name}'
    
    data = filter_df(data, ** filters)
    
    result = []
    for group_values, grouped_data in data.groupby(group_by):
        if not isinstance (group_values, (list, tuple)): group_values = [group_values]
        
        grouped_values = {n : v for n, v in zip(group_by, group_values)}
        for c in columns:
            grouped_values.update({
                name_format.format(name = name, c = c) : fn(grouped_data[c])
                for name, fn in kwargs.items()
            })
        result.append(grouped_values)
    
    result = pd.DataFrame(result)
    
    if merge:
        result = pd.merge(data, result, on = group_by)
    
    return result


def compare_df(df1, df2):
    """
        Compare 2 pd.DataFrame element-wise and return a pd.DataFrame with each [row, col] is True (equal) or False
    """
    union = [c for c in df1.columns if c in df2]
    
    df1 = df1.reset_index()
    df2 = df2.reset_index()
    
    result = []
    for idx in range(len(df1)):
        same_idx = {}
        for c in union:
            v1, v2 = df1.at[idx, c], df2.at[idx, c]
            if not isinstance(v1, type(v2)): same_idx[c] = False
            elif isinstance(v1, np.ndarray): same_idx[c] = np.allclose(v1, v2)
            else: same_idx[c] = v1 == v2
        result.append(same_idx)
    
    return pd.DataFrame(result)
