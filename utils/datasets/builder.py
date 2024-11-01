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

import os
import keras
import logging
import numpy as np

from keras import tree
from sklearn.utils import shuffle as sklearn_shuffle
from sklearn.model_selection import train_test_split as sklearn_train_test_split

from utils.pandas_utils import is_dataframe
from utils.stream_utils import create_iterable

logger  = logging.getLogger(__name__)

def prepare_dataset(data,
                    *,

                    augment_raw_fn  = None,
                    prepare_fn  = None,
                    filter_fn   = None,
                    process_fn  = None,
                    augment_fn  = None,
                    process_batch_fn    = None,
                    
                    map_fn  = None,

                    cache       = True,
                    shuffle     = False,
                    prefetch    = True,
                    batch_size  = -1,
                    pad_kwargs  = {},
                    
                    shuffle_size    = None,
                    prefetch_size   = None,
                    num_parallel_calls  = None,
                    
                    ** kwargs
                   ):
    """
        Prepare the dataset for training with all configuration given
        
        The operations are executed in the following order :
            1) Convertion of `data` to a `tf.data.Dataset`
            2) Apply `augment_raw_fn`
            3) Apply `prepare_fn`
            4) Apply `filter_fn`
            5) Apply `process_fn`
            6) Apply `cache_fn` (if `cache = True`)
            7) Apply `shuffle` (if `shuffle = True`)
            8) Apply `augment_fn`
            9) Apply `batch` or `padded_batch` (if `pad_kwargs` is provided) (if `batch_size > 0`)
            10) Apply `process_batch_fn`
            11) Apply `prefetch` (if `prefetch = True`)
        
        If `shuffle_size`, `prefetch_size` or `num_parallel_calls` are not specified, they are set to `tf.data.AUTOTUNE` by default
    """
    dataset = build_tf_dataset(data, ** kwargs)
    if dataset is None: return None
    
    import tensorflow as tf
    
    if map_fn is not None: process_fn = map_fn
    
    if shuffle_size is None:        shuffle_size = tf.data.experimental.cardinality(dataset)
    if prefetch_size is None:       prefetch_size = tf.data.AUTOTUNE
    if num_parallel_calls is None:  num_parallel_calls = tf.data.AUTOTUNE
    logger.debug("Original dataset : {}".format(dataset))
    
    if augment_raw_fn is not None:
        if cache:
            logger.warning('Cache is disabled because `augment_raw_fn` is provided')
            cache = False
        dataset = dataset.map(augment_raw_fn, num_parallel_calls = num_parallel_calls)
        logger.debug("- Dataset after original data augmentation : {}".format(dataset))

    if prepare_fn is not None:
        dataset = dataset.map(prepare_fn, num_parallel_calls = num_parallel_calls)
        logger.debug("- Dataset after preparation : {}".format(dataset))
    
    if filter_fn is not None:
        dataset = dataset.filter(filter_fn)
        logger.debug("- Dataset is filtered")
    
    if process_fn is not None:
        dataset = dataset.map(process_fn, num_parallel_calls = num_parallel_calls)
        logger.debug("- Dataset after processing : {}".format(dataset))
    
    if cache:
        dataset = dataset.cache('' if cache is True else cache)
        logger.debug('- Dataset is cached')

    if shuffle:
        dataset = dataset.shuffle(shuffle_size)
        logger.debug('- Dataset is shuffled')
    
    if augment_fn is not None: 
        dataset = dataset.map(augment_fn, num_parallel_calls = num_parallel_calls)
        logger.debug("- Dataset after augmentation : {}".format(dataset))

    if batch_size > 0:
        if pad_kwargs or _has_variable_shape(dataset):
            dataset = dataset.padded_batch(batch_size, ** pad_kwargs)
        else:
            dataset = dataset.batch(batch_size)
        logger.debug("- Dataset after batch : {}".format(dataset))
    
    if process_batch_fn is not None:
        dataset = dataset.map(process_batch_fn, num_parallel_calls = num_parallel_calls)
        logger.debug("- Dataset after batch processing : {}".format(dataset))
    
    if prefetch:
        dataset = dataset.prefetch(prefetch_size)
        logger.debug("- Dataset is prefetched")
    
    return dataset

def build_tf_dataset(data, as_dict = True, is_rectangular = True, siamese = False, ** kwargs):
    """
        Build a `tf.data.Dataset` supporting multiple types of `data`
        
        Arguments : 
            - data  : the dataset to transform to `tf.data.Dataset`
                - `tf.data.Dataset` : returns `data`
                - `list, tuple, dict, np.ndarray`   : `tf.data.Dataset.from_tensor_slices`
                - `pd.DataFrame`    :
                    - if `rectangular = True`   : `from_tensor_slices(data.to_dict('list'))`
                    - if `rectangular = False`  : `from_list(data.to_dict('records'))`
                    Note that the 1st option is more optimized for large dataset (> 1000 items)
                - `str`     :
                    - if `os.path.isdir(data)`  : `from_files(data + '/*')`
                    - `.csv`    : `experimental.make_csv_dataset(data)`
                    - `.txt`    : `TextLineDataset(data)`
                    - others    : raise `ValueError`
                - other formats : fallbacks to `keras.src.trainers.data_adapters.get_data_adapter`
    """
    if data is None: return None
    
    import tensorflow as tf

    if isinstance(data, tf.data.Dataset): 
        dataset = data
    elif isinstance(data, (list, tuple, dict, np.ndarray)):
        dataset = tf.data.Dataset.from_tensor_slices(data)
    elif is_dataframe(data):
        if siamese:
            dataset = build_siamese_dataset(data, ** kwargs)
        elif as_dict:
            if is_rectangular:
                dataset = tf.data.Dataset.from_tensor_slices(data.to_dict('list'))
            else:
                dataset = tf.data.experimental.from_list(data.to_dict('records'))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(data.values)
    elif isinstance(data, keras.utils.PyDataset) and hasattr(data, 'output_signature'):
        dataset = tf.data.Dataset.from_generator(
            create_iterable(data), output_signature = tree.map_structure(
                lambda s: tf.TensorSpec(shape = s.shape, dtype = s.dtype), data.output_signature
            )
        )
    elif isinstance(data, str):
        if os.path.isdir(data):
            dataset = tf.data.Dataset.list_files(data + '/*')
        elif data.endswith('.csv'):
            dataset = tf.data.experimental.make_csv_dataset(data, ** kwargs)
        elif data.endswith('.txt'):
            dataset = tf.data.TextLineDataset(data)
        else:
            raise ValueError('This file format is not convertible to `tf.data.Dataset` : {}'.format(data))
    else:
        if not isinstance(data, keras.utils.PyDataset):
            logger.info('The `data` format is not supported by this custom function. Trying the `keras` `get_data_adapter` function...')
        
        from keras.src.trainers.data_adapters import get_data_adapter
        
        dataset = get_data_adapter(data).get_tf_dataset()
        
    return dataset

def train_test_split(dataset,
                     *,
                     train_size     = None,
                     valid_size     = None,
                     labels     = None,
                     
                     shuffle    = False,
                     random_state   = 10,
                     
                     split_by_unique    = False,
                     split_column   = 'id',
                     min_occurence  = -1,
                     ** _
                    ):
    """
        Split `dataset` into distinct `training` and `validation` sets
        
        Arguments :
            - dataset   : the data to split
                - `list / tuple` in the form (x, y) : calls `sklearn.train_test_split`
                - `pd.DataFrame / list / np.ndarray`    : calls `sklearn.train_test_split`
                    * special case for `pd.DataFrame` when `split_by_unique = True`
                - others    : creates a `tf.data.Dataset` then calls `take`
            
            - {train / valid}_size  : the specific subset size, either absolute (int) or relative (float)
            - labels    : forwarded as the `stratify` argument to `sklearn.train_test_split`
            
            - shuffle / random_state    : forwarded to `sklearn.train_test_split`
            
            - split_by_unique   : whether to creates subsets based on unique ids (only applicable on `pd.DataFrame`)
            - split_column      : the column to use to define unique ids
            - min_occurence     : the minimal number of occurence of a given id
        Return :
            - train : the training set
            - valid : the validation set
        
        The `split_by_unique` arguments divides the dataset into two distinct subsets with non-overlappling items in the `split_column` column
        The `{train / valid}_size` arguments are then relative to the number of unique values in `split_column` rather than the `dataset` length
        
        Suppose a dataset with the following repartition of labels : 
            {A : 100, B : 50, C : 20, D : 10}, and `train_size = 0.75`
        A regular split with stratify used will produce :
            - train : {A : 75, B : 37, C : 15, D : 7}
            - valid : {A : 25, B : 13, C : 5, D : 3}
        By using the `split_by_unique`
            - train : {A : 100, B : 50, D : 10}
            - valid : {C : 20}
        In this example, A, B, C and D are ids into the `id` column of the dataset
        
        Such split procedure is typically used in encoder models to evaluate their generalization capabilities on brand new labels / ids.
        For regular classification models, this can be used to evaluate the generalization to new subjects / contexts, but it should not be used on the label column directly
        
        As an example in the context of `speech-to-text` (audio transcription), it may be relevant to split the dataset into distinct speakers, such that the model is evaluated on new speakers
    """
    if not train_size and not valid_size:
        raise ValueError('You must provide either `train_size` either `valid_size` (or both)')
    
    if isinstance(dataset, (list, tuple)) and len(dataset) == 2:
        x_train, x_valid, y_train, y_valid = sklearn_train_test_split(
            dataset, 
            train_size      = train_size, 
            test_size       = valid_size, 
            random_state    = random_state, 
            shuffle         = shuffle, 
            stratify        = labels
        )
        train, valid = (x_train, y_train), (x_valid, y_valid)
    elif is_dataframe(dataset) and split_by_unique:
        uniques = dataset[split_column].value_counts()
        if min_occurence > 0: uniques = uniques[uniques > min_occurence]

        train_uniques, valid_uniques = sklearn_train_test_split(
            uniques.index,
            train_size      = train_size, 
            test_size       = valid_size, 
            random_state    = random_state, 
            shuffle         = shuffle
        )
        
        train = dataset[dataset[split_column].isin(train_uniques)]
        valid = dataset[dataset[split_column].isin(valid_uniques)]
    elif is_dataframe(dataset) or isinstance(dataset, (np.ndarray, list, tuple)):
        if is_dataframe(dataset) and isinstance(labels, str):
            labels = dataset[labels].values
        
        train, valid = sklearn_train_test_split(
            dataset, 
            train_size      = train_size, 
            test_size       = valid_size, 
            random_state    = random_state, 
            shuffle         = shuffle, 
            stratify        = labels
        )
    else:
        length  = len(dataset) if hasattr(dataset, '__len__') else -1
        
        if length == -1:
            if isinstance(train_size, float) or isinstance(valid_size, float):
                raise ValueError('`float` sizes are unsupported when `dataset` has no fixed length')
            if train_size is None or valid_size is None:
                raise ValueError('`None` sizes are unsupported when `dataset` has no fixed length')
        
        if isinstance(train_size, float): train_size = int(train_size * length)
        if isinstance(valid_size, float): valid_size = int(valid_size * length)
        
        if train_size is None: train_size = length - valid_size
        if valid_size is None: valid_size = length - train_size
        
        assert valid_size > 0 and train_size > 0, f"Negative sizes are not supported :\n  Train size : {train_size}\n  Valid size : {valid_size}\n  Dataset length : {length}"
        assert length == -1 or (valid_size + train_size <= length), f"Train size + valid size must be less or equal than the total length\n  Train size : {train_size}\n  Valid length : {valid_size}\n  Dataset length : {length}"
        
        dataset = build_tf_dataset(dataset)
        
        train = dataset.take(train_size)
        valid = dataset.skip(train_size).take(valid_size)
    
    return train, valid

def build_siamese_dataset(dataset,
                          
                          column    = 'id',
                          suffixes  = ('_x', '_y'), 
                          
                          nb_unique = 1000,
                          max_by_id = 100,
                          strict_equality  = True,
                          
                          as_tf_dataset     = True,
                          
                          shuffle       = True, 
                          random_state  = 10,
                          tqdm          = lambda x: x,
                          
                          ** kwargs
                         ):
    """
        Build siamese dataset : a dataset with only valid pairs and one with invalid pairs (pairs from 2 different speakers). 
        Arguments :
            - data  : the pd.DataFrame audio datasets
            - nb_unique : the number of speakers to use
            - max_by_id : maximum instances for 'same-pairs' dataset (by id)
            - strict_equality   : whether the same and not same must be of same length
            - shuffle   : whether to shuffle or not
            - random_state  : state to use for reproducibility
    """
    import pandas as pd
    
    assert is_dataframe(dataset)
    # Useful for some datasets where pairs are not based on ID's (such as SNLI)
    if 'same' in dataset.columns:
        same_ds = dataset[dataset['same']]
        not_same_ds = dataset[~dataset['same']]

        if shuffle:
            same_ds = sklearn_shuffle(same_ds, random_state = random_state)
            not_same_ds = sklearn_shuffle(not_same_ds, random_state = random_state)
        
        if not as_tf_dataset: return same_ds.reset_index(), not_same_ds.reset_index()
    
        same_ds = build_tf_dataset(same_ds)
        not_same_ds = build_tf_dataset(not_same_ds)
    
        return tf.data.Dataset.zip((same_ds, not_same_ds))
    
    
    # Get unique IDs
    uniques = dataset[column].value_counts()
    if nb_unique is not None and len(uniques) > nb_unique:
        uniques = uniques[:nb_unique]
    uniques = uniques.index
    
    rng = np.random.default_rng(seed = random_state)
    
    dataset = dataset.to_dict('records')
    
    liste_same, liste_not_same = [], []
    for unique in tqdm(uniques):
        # Get dataset rows with only `unique` (sames) or not (not_sames)
        sames = [row for row in dataset if row[column] == unique]
        not_sames = [row for row in dataset if row[column] != unique]
        
        sames_idx = list(range(len(sames)))
        not_sames_idx = list(range(len(not_sames)))
        
        # Build combinations for `sames` part
        same_combinations   = list(itertools.combinations(sames_idx, 2))
        n = len(same_combinations) if not max_by_id else min(max_by_id, len(same_combinations))
        
        indexes = rng.choice(len(same_combinations), size = n, replace = False)
        
        merged_same = []
        for idx1, idx2 in [same_combinations[idx] for idx in indexes]:
            s1, s2 = sames[idx1], sames[idx2]
            
            s = {column : s1[column]}
            s.update({k + suffixes[0] : v for k, v in s1.items() if k != column})
            s.update({k + suffixes[1] : v for k, v in s2.items() if k != column})

            merged_same.append(s)

        merged_same = pd.DataFrame(merged_same)
        if max_by_id and len(merged_same) > max_by_id: 
            merged_same = merged_same.sample(max_by_id, random_state = random_state)
        
        # Build combinations for `not_sames` part
        nb = min(max(len(merged_same) // len(sames_idx), 10), len(not_sames_idx))
        not_same_combinations   = []
        for idx1 in sames_idx:
            not_same_combinations += [
                (idx1, idx2) for idx2 in rng.choice(
                    len(not_sames_idx), size = nb
                )
            ]
        n = len(not_same_combinations) if not max_by_id else min(max_by_id, len(not_same_combinations))
        
        indexes = rng.choice(len(not_same_combinations), size = n, replace = False)
        
        
        merged_not_same = []
        for idx1, idx2 in [not_same_combinations[idx] for idx in indexes]:
            s1, s2 = sames[idx1], not_sames[idx2]
            
            s = {k + suffixes[0] : v for k, v in s1.items()}
            s.update({k + suffixes[1] : v for k, v in s2.items()})

            merged_not_same.append(s)

        merged_not_same = pd.DataFrame(merged_not_same)
        # Sample a subset (if required)
        if strict_equality and len(merged_same) != len(merged_not_same):
            if len(merged_not_same) > len(merged_same): 
                merged_not_same = merged_not_same.sample(
                    len(merged_same), random_state = random_state
                )
            else:
                merged_same = merged_same.sample(
                    len(merged_not_same), random_state = random_state
                )
        elif max_by_id and len(merged_not_same) > max_by_id:
            merged_not_same = merged_not_same.sample(
                max_by_id, random_state = random_state
            )

        # Append final result to global lists
        liste_same.append(merged_same)
        liste_not_same.append(merged_not_same)

    same_ds = pd.concat(liste_same, ignore_index = True)
    not_same_ds = pd.concat(liste_not_same, ignore_index = True)

    if shuffle:
        same_ds = sklearn_shuffle(same_ds, random_state = random_state)
        not_same_ds = sklearn_shuffle(not_same_ds, random_state = random_state)
        
    if not as_tf_dataset: return same_ds.reset_index(), not_same_ds.reset_index()
    
    same_ds = build_tf_dataset(same_ds)
    not_same_ds = build_tf_dataset(not_same_ds)
    
    return tf.data.Dataset.zip((same_ds, not_same_ds))

def _has_variable_shape(dataset):
    return any(
        any(s is None for s in spec.shape)
        for spec in keras.tree.flatten(dataset.element_spec)
    )