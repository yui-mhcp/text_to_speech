
# Copyright (C) 2022 yui-mhcp project's author. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import time
import logging
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from sklearn.utils import shuffle as sklearn_shuffle
from sklearn.model_selection import train_test_split as sklearn_train_test_split

from loggers import DEV
from utils.generic_utils import time_to_string
from utils.pandas_utils import filter_df

logger  = logging.getLogger(__name__)

AUTOTUNE = tf.data.experimental.AUTOTUNE

_accepted_datasets_types = {
    'tuple' : '(x, y)',
    'np.ndarray'    : 'tableau numpy',
    'pd.Dataframe'  : 'tableau pandas',
    'str'   : {
        'direcotry' : 'répertoire de sous-dossiers contenant les fichiers',
        'file'      : 'file with extension .txt or .csv'
    },
    'tf.data.Dataset'   : 'tensorflow dataset'
}

def _maybe_load_embedding(directory, dataset, ** kwargs):
    if 'embedding' in dataset.columns: return dataset
    if any(k.startswith('embedding_') for k in kwargs.keys()):
        return load_embedding(directory, dataset = dataset, ** kwargs)
    return dataset

def _get_infos(tensor, level = 0):
    indent = ' ' * level
    if isinstance(tensor, (list, tuple)):
        return ''.join([
            '\n{}Item {} : {}'.format(indent, i, _get_infos(t, level+1)) 
            for i, t in enumerate(tensor)
        ])
    elif isinstance(tensor, dict):
        return ''.join([
            '\n{}Item {} : {}'.format(indent, k, _get_infos(t, level+1)) 
            for k, t in tensor.items()
        ])

    infos = 'shape : {} - type : {}'.format(tensor.shape, tensor.dtype.name)
    if tensor.dtype != tf.string:
        infos += '- min : {:.3f} - max : {:.3f}'.format(np.min(tensor), np.max(tensor))
    return infos

def _infer_type_spec(item):
    if isinstance(item, dict):
        return {k : _infer_type_spec(v) for k, v in item.items()}
    
    shape, dtype = (), None
    if isinstance(item, (list, tuple)): shape, dtype = (None, ), _infer_type_spec(item[0]).dtype
    elif isinstance(item, str):                                 dtype = tf.string
    elif isinstance(item, (int, np.integer)):         dtype = tf.int32
    elif isinstance(item, (float, np.floating)):    dtype = tf.float32
    elif isinstance(item, (np.ndarray, tf.Tensor)):
        if isinstance(item, tf.Tensor): dtype = item.dtype
        elif item.dtype == np.int32:    dtype = tf.int32
        elif item.dtype == np.float32:  dtype = tf.float32
        elif item.dtype == np.object:   dtype = tf.string
        shape = tuple([None for _ in range(len(item.shape))])
    else:
        raise ValueError("Unknown type spec for item {}".format(item))
        
    return tf.TensorSpec(shape = shape, dtype = dtype)

def _nested_count(col):
    count = {}
    for l in col:
        if not isinstance(l, list):
            count.setdefault(l, 0)
            count[l] += 1
            continue

        for v in l:
            count.setdefault(v, 0)
            count[v] += 1
    
    return {
        k : v for k, v in sorted(count.items(), key = lambda p: p[1], reverse = True)
    }

def _get_col_summary(col, limit = 0.1, ** kwargs):
    val0 = col.iloc[0]
    if not isinstance(val0, (str, int, float, np.integer, np.floating, list)): return {}
    
    infos = {}
    
    if isinstance(val0, list):
        if not isinstance(val0[0], str): return {}
        
        count = _nested_count(col.values)
    else:
        count   = col.value_counts().to_dict()
    
    if len(count) > limit:
        infos['# uniques']  = len(count)
    else:
        infos['uniques']    = count

    if not isinstance(val0, (str, list)):
        infos.update({
            k : v for k, v in col.describe().items() if k != 'count'
        })
    
    return infos

def _infer_generator_spec(generator):
    return _infer_type_spec(generator[0])

def summarize_dataset(dataset, cols = None, limit = 0.1, ** kwargs):
    if not isinstance(dataset, pd.DataFrame): return {}
    
    if isinstance(limit, float): limit = int(limit * len(dataset))
    
    if cols is None: cols = dataset.columns
    return {
        col : _get_col_summary(dataset[col], limit = limit, ** kwargs)
        for col in cols
    }
    
def filter_dataset(dataset, on_unique = [], ** kwargs):
    return filter_df(dataset, on_unique = on_unique, ** kwargs)

def test_dataset_time(dataset, steps = 100, batch_size = 0, ** kwargs):
    """
        Generate `steps` batchs of `dataset` and compute its average time
        It also shows information on the last generated batch
    """
    start = time.time()
    i = 0
    batch = None
    for batch in tqdm(dataset.take(steps)):
        i += 1
        if i >= steps: break

    temps = time.time() - start
    
    samp_per_s = '\n' if batch_size <= 0 else ' ({:.3f} samples / sec)'.format(batch_size * i / temps)
    logger.info("\n{} batchs in {} sec ({:.3f} batch / sec){}".format(
        i, time_to_string(temps), i / temps, samp_per_s
    ))

    size = tf.data.experimental.cardinality(dataset).numpy()
    if size > 0:
        logger.info("Time estimated for all dataset ({} batch) : {}".format(
            size, time_to_string(size * (i / temps))
        ))
        
    logger.info("Batch infos : {}".format(_get_infos(batch)))
            
    return temps

def build_tf_dataset(data, as_dict = True, is_rectangular = True, siamese = False, ** kwargs):
    """
        Build a tf.data.Dataset based on multiple types of `data`
        
        Arguments : 
            - data  : the dataset to transform to tf.data.Dataset
                - tf.data.Dataset   : return itself
                - list / tuple      : tf.data.Dataset.from_tensor_slices
                - tf.keras.utils.Sequence   : tf.data.Dataset.from_generator
                - pd.DataFrame      : tf.data.Dataset.from_tensor_slices
                    if as_dict  : datas are dict where keys are col names
                    else : treat it as a np.ndarray (data.values)
                - str   : either a directory, .csv file, .txt file
    """
    if data is None:
        return None
    elif isinstance(data, tf.data.Dataset): 
        dataset = data
    elif isinstance(data, (list, tuple, np.ndarray)):
        dataset = tf.data.Dataset.from_tensor_slices(data, **kwargs)
    elif isinstance(data, tf.keras.utils.Sequence):
        if hasattr(data, 'output_types'): kwargs['output_types'] = data.output_types
        if hasattr(data, 'output_shapes'): kwargs['output_shapes'] = data.output_shapes
        if hasattr(data, 'output_signature'): kwargs['output_signature'] = data.output_signature
        if not callable(data): data = default_generator_fn(data)
        dataset = tf.data.Dataset.from_generator(data, **kwargs)
    elif callable(data):
        dataset = tf.data.Dataset.from_generator(data, **kwargs)
    elif isinstance(data, np.ndarray):
        dataset = tf.data.Dataset.from_tensor_slices(data, **kwargs)
    elif isinstance(data, pd.DataFrame):
        if siamese:
            dataset = build_siamese_dataset(data, ** kwargs)
        elif as_dict:
            if is_rectangular:
                dataset = tf.data.Dataset.from_tensor_slices(data.to_dict('list'), ** kwargs)
            else:
                data = data.to_dict('records')
                if 'output_signature' not in kwargs:
                    kwargs['output_signature'] = _infer_generator_spec(data)
                    logger.log(DEV, 'Inferred dataset signature : {}'.format(
                        kwargs['output_signature']
                    ))

                dataset = tf.data.Dataset.from_generator(default_generator_fn(data), ** kwargs)
        else:
            dataset = tf.data.Dataset.from_tensor_slices(data.values, ** kwargs)
    elif isinstance(data, str):
        if os.path.isdir(data):
            path = pathlib.Path(main_directory_path)
            dataset = tf.data.Dataset.list_files(str(path/'*/*'))
        elif '.csv' in data:
            dataset = tf.data.experimental.make_csv_dataset(data, **kwargs)
        elif '.txt' in data:
            dataset = tf.data.TextLineDataset(data)
        else:
            raise ValueError("data de type str non géré !\nReçu : {}\nAcceptés : {}".format(data, json.dumps(_accepted_datasets_types, indent=2)))
    else:
        raise ValueError("Dataset de type inconnu !\nReçu : {} (type : {})\nAcceptés :{}".format(data, type(data), json.dumps(_accepted_datasets_types, indent=2)))
        
    return dataset

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
    assert isinstance(dataset, pd.DataFrame)
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

def train_test_split(dataset, 
                     train_size     = None,
                     valid_size     = None,
                     random_state   = 10,
                     shuffle        = False,
                     labels         = None,
                     split_by_unique    = False,
                     split_column   = 'id',
                     min_occurence  = 5
                    ):
    """
        Split dataset in training and validation sets for various dataset type
        
        Procedure for each format :
        - dataset   : the dataset to split
            - tf.data.Dataset / callable / tf.keras.utils.Sequence : 
                1) Convert it to a tf.data.Dataset (if necessary)
                2) Call `.take(train_size)` to create the training set
                3) Call `.skip(train_size).take(valid_size)` to create the valid set
                It means that both train_size and valid_size must be specified and in absolute values
            - list / tuple of length 2  : calls train_test_split() from sklearn
            - not split_by_unique       : calls train_test_split() from sklearn
            - pd.DataFrame and split_by_unique  : 
                1) Get uniques values in `dataset[split_column]`
                2) Get only values which have at least `min_occurence` occurences
                3) Call train_test_split from sklearn on unique values
                4) Build train / valid sets based on the unique-value split
    """
    if isinstance(dataset, (tf.data.Dataset, tf.keras.utils.Sequence)) or callable(dataset):
        if not train_size and not valid_size:
            raise ValueError("Il faut spécifier au moins train_size ou valid_size pour pouvoir diviser le dataset !")
        
        if isinstance(dataset, tf.keras.utils.Sequence): length = len(dataset)
        elif isinstance(dataset, tf.data.Dataset): 
            length = tf.data.experimental.cardinality(dataset)
        else: length = -1
        
        if length < 0 and (train_size is None or valid_size is None or isinstance(train_size, float) or isinstance(valid_size, float)):
            raise ValueError("Le dataset n'a pas une longueur connue ! il faut donc spécifier train_size et valid_size pour le diviser")
        
        if isinstance(train_size, float): train_size = int(train_size * length)
        if isinstance(valid_size, float): valid_size = int(valid_size * length)
        
        if train_size is None: train_size = length - valid_size
        if valid_size is None: valid_size = length - train_size
        
        assert valid_size > 0 and train_size > 0 and (length < 0 or (train_size + valid_size <= length)), "Length : {}\nTrain size : {}\nValid size : {}".format(length, train_size, valid_size)
        
        dataset = build_tf_dataset(dataset)
        
        train = dataset.take(train_size)
        valid = dataset.skip(train_size).take(valid_size)
        
    elif isinstance(dataset, (list, tuple)) and len(dataset) == 2:
        x_train, x_valid, y_train, y_valid = sklearn_train_test_split(
            dataset, 
            train_size      = train_size, 
            test_size       = valid_size, 
            random_state    = random_state, 
            shuffle         = shuffle, 
            stratify        = labels
        )
        train, valid = (x_train, y_train), (x_valid, y_valid)
    elif isinstance(dataset, pd.DataFrame) and split_by_unique:
        uniques = dataset[split_column].value_counts()
        if min_occurence is not None and min_occurence > 0:
            uniques = uniques[uniques > min_occurence]
        uniques = uniques.index

        train_uniques, valid_uniques = sklearn_train_test_split(
            uniques,
            train_size      = train_size, 
            test_size       = valid_size, 
            random_state    = random_state, 
            shuffle         = shuffle
        )
        
        train = dataset[dataset[split_column].isin(train_uniques)]
        valid = dataset[dataset[split_column].isin(valid_uniques)]
    else:
        train, valid = sklearn_train_test_split(
            dataset, 
            train_size      = train_size, 
            test_size       = valid_size, 
            random_state    = random_state, 
            shuffle         = shuffle, 
            stratify        = labels
        )
    
    return train, valid

def prepare_dataset(data, 
                    batch_size             = 1,
                    shuffle_size           = -1,
                     
                    encode_fn              = None,
                    filter_fn              = None,
                    augment_fn             = None,
                    map_fn                 = None,
                    memory_consuming_fn    = None,
                     
                    prefetch               = True,
                    cache                  = True,
                    batch_before_map       = False,
                    padded_batch           = False,
                    pad_kwargs             = {},
                     
                    prefetch_size          = AUTOTUNE,
                    num_parallel_calls     = AUTOTUNE,
                    
                    ** kwargs
                   ):
    """
        Prepare the dataset for training with all configuration given
        
        Preparation procedure : 
            1) Convert it to a tf.data.Dataset with `build_tf_dataset` function
            2) If provided : map dataset with `encode_fn`
            3) If provided : filter dataset with `filter_fn`
            4) If batch_before_map      : cache then batch
            5) If provided : map dataset with `map_fn`
            6) if not batch_before_map  : Cache dataset
            7) If provided : map dataset with `memory_consuming_fn`
            8) If not batch_before_map  : batch dataset
            9) Prefetch dataset
        
        The cache procedure is : 
            1) cache
            2) shuffle
        The batch procedure is :
            1) If provided : map dataset with `augment_fn`
            2) batch dataset (with padded batch if required)
        
        Note that caching is always done before batching because we do not want to cache augmented data (and data augmentation is always done right before batching)
    """
    def cache_dataset(dataset):
        if cache:
            dataset = dataset.cache('' if cache is True else cache)
        
        if shuffle_size > 0:
            dataset = dataset.shuffle(shuffle_size)
        
        return dataset
        
    def batch_dataset(dataset):
        if augment_fn is not None: 
            dataset = dataset.map(augment_fn, num_parallel_calls = num_parallel_calls)
            logger.log(DEV, "- Dataset after augmentation : {}".format(dataset))
        
        if batch_size > 0:
            if not padded_batch:
                dataset = dataset.batch(batch_size)
            else:
                dataset = dataset.padded_batch(batch_size, ** pad_kwargs)
            logger.log(DEV, "- Dataset after batch : {}".format(dataset))
            
        return dataset 
    
    dataset = build_tf_dataset(data, ** kwargs)
    if dataset is None: return None
    
    logger.log(DEV, "Original dataset : {}".format(dataset))
    
    if encode_fn is not None:
        dataset = dataset.map(encode_fn, num_parallel_calls = num_parallel_calls)
        logger.log(DEV, "- Dataset after encoding : {}".format(dataset))
    
    if filter_fn is not None:
        dataset = dataset.filter(filter_fn)
        logger.log(DEV, "- Dataset after filtering : {}".format(dataset))
    
    if batch_before_map:
        dataset = cache_dataset(dataset)
        dataset = batch_dataset(dataset)
        
    if map_fn is not None:
        dataset = dataset.map(map_fn, num_parallel_calls = num_parallel_calls)
        logger.log(DEV, "- Dataset after mapping : {}".format(dataset))
    
    if not batch_before_map:
        dataset = cache_dataset(dataset)
    
    if memory_consuming_fn is not None:
        dataset = dataset.map(
            memory_consuming_fn, num_parallel_calls = num_parallel_calls
        )
        logger.log(DEV, "- Dataset after memory mapping : {}".format(dataset))
    
    if not batch_before_map:        
        dataset = batch_dataset(dataset)
    
    if prefetch:
        dataset = dataset.prefetch(prefetch_size)
    
    return dataset
    
def default_generator_fn(generator):
    def generator_fn():
        i = 0
        while i < len(generator):
            yield generator[i]
            i += 1
    return generator_fn

