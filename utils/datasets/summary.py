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
import json
import time
import keras
import logging
import numpy as np

from tqdm import tqdm

from .builder import prepare_dataset
from utils.keras_utils import ops
from utils.pandas_utils import is_dataframe
from utils.generic_utils import time_to_string

logger  = logging.getLogger(__name__)

benchmark_message = """
{steps} batches generated in {total_time} ({batch_per_sec:.3f} batch / sec){build_time_msg}
- Initial batch time : {warmup_time}
- Average batch time : {average_time}
- Batch statistics   : {batch_stats}
"""

def summarize_dataset(dataset, columns = None, limit = 0.25, ** kwargs):
    if not is_dataframe(dataset): return {}
    
    if isinstance(limit, float): limit = int(limit * len(dataset))
    
    if columns is None: columns = dataset.columns
    return {
        col : _summarize_column(dataset[col], limit = limit, ** kwargs) for col in columns
    }

def benchmark_dataset(dataset, steps = 100, build = False, ** kwargs):
    """ Iterates over `dataset` for `steps` iterations, and reports statistics """
    
    t0 = time.time()
    
    if build: dataset = prepare_dataset(dataset, ** kwargs)
    t1 = time.time()
    
    times = [t1]
    for i, batch in enumerate(tqdm(dataset, total = steps)):
        times.append(time.time())
        if steps > 0 and i >= steps - 1: break
    
    batch_size  = 1
    input_ds    = dataset
    while hasattr(input_ds, '_input_dataset'):
        if hasattr(input_ds, '_batch_size'):
            batch_size = int(input_ds._batch_size)
            break
        input_ds = input_ds._input_dataset
    
    t2 = times[-1]
    times = [times[i] - times[i - 1] for i in range(1, len(times))]
    
    infos   = {
        'steps'     : len(times),
        'batch_size'    : batch_size,
        'total time'    : t2 - t0,
        'initial batch time'    : times[0],
        'average batch time'    : sum(times) / len(times)
    }
    if build: infos['build time'] = t1 - t0
    
    logger.info(benchmark_message.format(
        steps   = len(times),
        total_time  = time_to_string(sum(times)),
        average_time    = time_to_string(sum(times) / len(times)),
        warmup_time     = time_to_string(times[0]),
        batch_per_sec   = len(times) / np.sum(times),
        build_time_msg  = '' if not build else f'\n- Build time : {time_to_string(t1 - t0)}',
        batch_stats = json.dumps(keras.tree.map_structure(_get_stats, batch), indent = 4)
    ))
            
    return infos

test_dataset_time = benchmark_dataset

def _get_stats(x):
    x = ops.convert_to_numpy(x)
    if ops.is_int(x):
        return 'shape : {} - min : {} - max : {}'.format(
            x.shape, np.min(x), np.max(x)
        )
    elif ops.is_float(x):
        return 'shape : {} - min : {:.3f} - max : {:.3f} - mean : {:.3f}'.format(
            x.shape, np.min(x), np.max(x), np.mean(x)
        )
    else:
        return 'shape : {}'.format(x.shape)

def _summarize_column(col, limit, ** kwargs):
    if not isinstance(col.iloc[0], (str, int, float, np.integer, np.floating, list)): return {}
    
    infos = {}
    
    if isinstance(col.iloc[0], list):
        if not isinstance(col.iloc[0][0], (str, int)): return {}
        count = _nested_count(col.values)
    else:
        count   = col.value_counts().to_dict()
    
    if len(count) > limit:
        infos['# uniques']  = len(count)
    else:
        infos['uniques']    = count

    if not isinstance(col.iloc[0], (str, list)):
        infos.update({
            k : v for k, v in col.describe().items() if k != 'count'
        })
    
    return infos

def _nested_count(col):
    count = {}
    for l in col:
        if not isinstance(l, list): l = [l]
        for v in l:
            count.setdefault(v, 0)
            count[v] += 1
    
    return {
        k : v for k, v in sorted(count.items(), key = lambda p: p[1], reverse = True)
    }
