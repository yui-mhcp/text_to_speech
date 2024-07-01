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
import re
import glob
import json
import pickle
import logging
import numpy as np
import pandas as pd
import keras.ops as K

from tqdm import tqdm
from keras import tree

from utils.keras_utils import ops
from utils.wrapper_utils import dispatch_wrapper, partial
from utils.generic_utils import to_json, convert_to_str

logger = logging.getLogger(__name__)

_load_file_fn   = {}
_dump_file_fn   = {}

def is_path(data):
    """ Returns whether `data` is a valid path or not """
    if not isinstance(data, str): return False
    return (
        len(data) < 512 and any(c.isalnum() for c in data) and os.path.exists(data)
    )

def path_to_unix(path):
    """ Simply replaces '\' to '/' :D """
    if not isinstance(path, str): return path
    return path.replace('\\', '/')

def sort_files(filenames):
    if isinstance(filenames, str):
        if os.path.isdir(filenames):
            return sort_files([os.path.join(filenames, f) for f in os.listdir(filenames)])
        return [filenames]
    return sorted(filenames, key = lambda f: (len(f), f))

def get_files(file_format, sort = False):
    if '{' not in file_format: return file_format
    results = glob.glob(re.sub(r'\{.*\}', r'*', file_format))
    if sort: results = sort_files(results)
    return results

def normalize_filename(filename,
                       keys = 'filename',
                       unix = False,
                       recursive = True,
                       invalid_mode = 'error'
                      ):
    """
        Return (list of) str filenames extracted from multiple formats
        
        Arguments :
            - filename  : (list of) filenames
                - str   : filename / directory path (if `recursive`, directories are extended)
                - pd.DataFrame  : must have a `key` column
                - dict      : must have a `key` entry
                - Tensor / np.ndarray / bytes    : string / bytes
            - keys  : the column / key to use if `filename` is a `dict` or `pd.DataFrame`
            - unix  : whether to convert path to unix-style (i.e. with '/' instead of '\')
            - recursive : whether to expand directories or not
            - invalid_mode  : either `error`, `keep` or `skip`, the action to perform when a data is not a string
    """
    if isinstance(filename, pd.DataFrame):
        if isinstance(keys, (list, tuple)):
            valids  = [k for k in keys if k in filename.columns]
            keys    = valids[0] if len(valids) > 0 else keys[0]
        filename = filename[keys].values if keys in filename.columns else None
    elif isinstance(filename, (dict, pd.Series)):
        if isinstance(keys, (list, tuple)):
            valids  = [k for k in keys if k in filename]
            keys    = valids[0] if len(valids) > 0 else keys[0]
        filename = filename[keys] if keys in filename else None
    
    filename = convert_to_str(filename)
    
    if isinstance(filename, str):
        if not os.path.isdir(filename): return filename if not unix else path_to_unix(filename)
        elif not recursive: return None
        outputs = tree.flatten([
            normalize_filename(os.path.join(filename, f)) for f in os.listdir(filename)
        ])
    elif isinstance(filename, (list, tuple)):
        outputs = tree.flatten([normalize_filename(
            f, key = key, unix = unix, recursive = recursive, invalid_mode = invalid_mode
        ) for f in filename])
    else:
        if invalid_mode == 'skip':      return None
        elif invalid_mode == 'keep' :   return filename
        else: raise ValueError("Unsupported `filename` ({}) : {}".format(type(filename), filename))
    
    return [o for o in outputs if o is not None]

get_filename = partial(
    normalize_filename, unix = True, keys = 'filename', invalid_mode = 'skip'
)

def hash_file(filename):
    """ Return the SHA256 signature of a file """
    import hashlib
    
    with open(filename, 'rb') as file:
        return hashlib.sha256(file.read()).hexdigest()

def download_file(url,
                  filename  = None,
                  directory = None,
                  overwrite = False,
                  sha256    = None,
                  buffer_size   = 8192
                 ):
    """
        Downloads file at `url`
        
        Arguments :
            - url   : the file url
            - filename  : the output filename (default to `os.path.basename(url)`)
            - directory : where to save the output file
            - overwrite : whether to overwrite the file if already there
            - sha256    : expected SHA256 signature of the file
            - buffer_size   : the reading buffer size
        Return :
            - filename  : the output file (None if `urllib` is not installed)
    """
    try:
        import urllib
    except ImportError as e:
        logger.error('Please install `urllib` to download files !')
        return None
    
    if not filename: filename = os.path.basename(url)
    if directory:
        os.makedirs(directory, exist_ok = True)
        filename = os.path.join(directory, filename)

    if os.path.exists(filename):
        if not os.path.isfile(filename):
            raise RuntimeError(f"{filename} exists but is not a regular file".format(filename))
        
        if not overwrite:
            if not sha256 or hash_file(filename) == sha256: return filename

            logger.warning('{} exists but has an invalid SHA256 : re-loading it'.format(filename))
        os.remove(filename)

    with urllib.request.urlopen(url) as source, open(filename, 'wb') as output_file:
        with tqdm(total = int(source.info().get("Content-Length")), unit = 'iB', unit_scale = True, unit_divisor = 1024) as loop:
            while True:
                buffer = source.read(buffer_size)
                if not buffer: break

                output_file.write(buffer)
                loop.update(len(buffer))

    if sha256 and hash_file(filename) != sha256:
        raise RuntimeError("The file has been downloaded but the SHA256 does not match")

    return filename

@dispatch_wrapper(_load_file_fn, 'Filename extension')
def load_data(filename, ** kwargs):
    """ Loads data from `filename`. The loading function differs according to the extension. """
    ext = os.path.splitext(filename)[1][1:]
    
    if ext not in _load_file_fn:
        raise ValueError("Unhandled extension !\n  Accepted : {}\n  Got : {}".format(
            tuple(_load_file_fn.keys()), ext
        ))
    
    if 'default' in kwargs:
        if not os.path.exists(filename): return kwargs['default']
        kwargs.pop('default')
    
    return _load_file_fn[ext](filename, ** kwargs)

@load_data.dispatch
def load_json(filename, default = {}, ** kwargs):
    """ Safely load data from a json file """
    if not filename.endswith('.json'): filename += '.json'
    if not os.path.exists(filename): return default
    with open(filename, 'r', encoding = 'utf-8') as file:
        result = file.read()
    return json.loads(result)

@load_data.dispatch
def load_npz(filename, ** kwargs):
    with np.load(filename) as file:
        data = {k : file[k] for k in file.files()}
    return data

@load_data.dispatch(('h5', 'hdf5'))
def load_h5(filename, keys = None, ** kwargs):
    import h5py
    
    def get_data(v):
        return v.asstr()[:] if v.dtype == object else np.array(v)
    
    def load_group(group, root):
        for k, v in group.items():
            path = '{}/{}'.format(group.name, k)
            if isinstance(v, h5py.Group):
                load_group(v, False)
            else:
                data[k if root else path] = get_data(v)

    data = {}
    with h5py.File(filename, 'r') as file:
        if keys:
            data = {k : get_data(file.get(k)) for k in keys if k in file}
        else:
            load_group(file, True)
    return data

@load_data.dispatch('pkl')
def load_pickle(filename, ** kwargs):
    with open(filename, 'rb') as file:
        return pickle.load(file)

@load_data.dispatch(['txt', 'md'])
def load_txt(filename, encoding = 'utf-8', ** kwargs):
    with open(filename, 'r', encoding = encoding) as file:
        return file.read()

load_data.dispatch(np.load, 'npy')
load_data.dispatch(pd.read_csv, 'csv')
load_data.dispatch(partial(pd.read_csv, sep = '\t'), 'tsv')
load_data.dispatch(pd.read_excel, 'xlsx')
load_data.dispatch(pd.read_pickle, 'pdpkl')


@dispatch_wrapper(_dump_file_fn, 'Filename extension')
def dump_data(filename, data, overwrite = True, ** kwargs):
    """ Dumps `data` into `filename`. The saving function differ according to the extension. """
    if K.is_tensor(data): data = K.convert_to_numpy(data)
    ext = os.path.splitext(filename)[1][1:]
    
    if not ext:
        for types, default_ext in _default_ext.items():
            if isinstance(data, types):
                filename, ext = '{}.{}'.format(filename, default_ext), default_ext
                break
        
        if not ext: filename, ext = '{}.pkl'.format(filename), 'pkl'
    
    if overwrite or not os.path.exists(filename):
        if ext not in _dump_file_fn:
            raise ValueError('Unhandled extention !\n  Accepted : {}\n  Got : {}'.format(
                tuple(_dump_file_fn.keys()), ext
            ))
        
        _dump_file_fn[ext](filename, data, ** kwargs)
    
    return filename

@dump_data.dispatch
def dump_json(filename, data, ** kwargs):
    """ Safely save data to a json file """
    if not filename.endswith('.json'): filename += '.json'
    data = json.dumps(to_json(data), ** kwargs)
    with open(filename, 'w', encoding = 'utf-8') as file:
        file.write(data)

@dump_data.dispatch(('h5', 'hdf5'))
def dump_h5(filename, data, mode = 'w', ** kwargs):
    import h5py
    
    if isinstance(data, pd.DataFrame): data = data.to_dict('list')
    
    with h5py.File(filename, mode) as file:
        for k, v in data.items():
            if k in file: continue
            if not isinstance(v, np.ndarray): v = ops.convert_to_numpy(v)
            
            dtype = h5py.string_dtype() if not ops.is_numeric(v) else None
            if dtype is not None: v = v.astype(dtype)
            file.create_dataset(k, data = v, dtype = dtype)
    
@dump_data.dispatch('pkl')
def dump_pickle(filename, data, ** kwargs):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

@dump_data.dispatch
def dump_txt(filename, data, ** kwargs):
    with open(filename, 'w', encoding = 'utf-8') as file:
        file.write(data)

@dump_data.dispatch
def dump_npy(filename, data, ** kwargs):
    np.save(filename, data)
    
@dump_data.dispatch
def dump_npz(filename, data, ** kwargs):
    np.savez(filename, ** data)

@dump_data.dispatch
def dump_csv(filename, data, ** kwargs):
    _to_df(data).to_csv(filename, ** kwargs)

dump_data.dispatch(partial(dump_csv, sep = '\t'), 'tsv')

@dump_data.dispatch('xlsx')
def dump_excel(filename, data, ** kwargs):
    _to_df(data).to_excel(filename, ** kwargs)

@dump_data.dispatch('pdpkl')
def dump_pandas_pickle(filename, data, ** kwargs):
    _to_df(data).to_pickle(filename, ** kwargs)

def _to_df(data):
    return data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)

_default_ext    = {
    str     : 'txt',
    (list, tuple, set, dict, int, float) : 'json',
    np.ndarray      : 'npy',
    pd.DataFrame    : 'h5'
}