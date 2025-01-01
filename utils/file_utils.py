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
import sys
import glob
import json
import pickle
import logging
import numpy as np

from tqdm import tqdm
from functools import wraps

try:
    from .keras_utils.ops import is_tensor, convert_to_numpy
except:
    from keras.ops import is_tensor, convert_to_numpy
from .generic_utils import to_json, convert_to_str
from .wrapper_utils import dispatch_wrapper, partial

logger = logging.getLogger(__name__)

_index_file_format_re = re.compile(r'\{i?(:\d{2}d)?\}')

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

def expand_path(path, recursive = True, unix = True):
    """
        Expand `path` to all matching file(s) if `path` is a directory or a formatted filename
        
        Arguments :
            - path  : str, a file, directory of formatted-file (i.e., unix-style format, `*.jpg`)
            - recursive : whether to expand nested directories
            - unix      : whether to replace '\' to '/'
        Return :
            - file(s)   : `list` of files or single file if `path` is a file
    """
    if '*' not in path:
        if not is_path(path) or not os.path.isdir(path): return path
        path = path + '/*'
    
    files = []
    for f in glob.glob(path):
        if os.path.isfile(f):
            if unix: f = path_to_unix(f)
            files.append(f)
        elif recursive:
            files.extend(expand_path(f, True))
    
    return files

def contains_index_format(path):
    """ Returns whether `path` has a `{}`, `{i}` or `{i:..d}` format """
    return path.count('{}') == 1 or _index_file_format_re.search(path) is not None

def get_path_index(path):
    if '{}' in path: path = path.replace('{}', '*')
    else:            path = _index_file_format_re.sub('*', path)
    return len(glob.glob(path))

def format_path_index(path):
    idx = get_path_index(path)
    return path.format(idx, i = idx)

def sort_files(filenames):
    if isinstance(filenames, str):               filenames = expand_path(filename)
    if not isinstance(filenames, (list, tuple)): filename = [filename]
    return sorted(filenames, key = lambda f: (len(f), f))

def hash_file(filename):
    """ Return the SHA256 signature of a file """
    import hashlib
    
    with open(filename, 'rb') as file:
        return hashlib.sha256(file.read()).hexdigest()

def remove_path(data, path):
    if path is True:
        try:
            from .datasets import get_dataset_dir
            path = get_dataset_dir()
        except (ImportError, ModuleNotFoundError):
            raise ImportError('Unable to import `get_dataset_dir`. Explicitely provide `prefix`')

    if isinstance(data, str):
        data, path = path_to_unix(data), path_to_unix(path)
        if not path.endswith('/'): path += '/'
        return data[len(path):] if data.startswith(path) else data
    elif isinstance(data, (list, tuple)):
        return [remove_path(d, path) for d in data]
    elif isinstance(data, dict):
        return {k : remove_path(v, path) if 'filename' in k else v for k, v in data.items()}
    elif hasattr(data, 'columns'):
        for col in data.columns:
            if 'filename' in col:
                data[col] = data[col].apply(lambda f: remove_path(f, path))
    
    return data

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
def load_jsonl(filename, ** kwargs):
    with open(filename, 'r', encoding = 'utf-8') as file:
        lines = [l for l in file]
    return [json.loads(l) for l in lines]

@load_data.dispatch
def load_npz(filename, ** kwargs):
    with np.load(filename) as file:
        data = {k : file[k] for k in file.files()}
    return data

@load_data.dispatch(('h5', 'hdf5'))
def load_h5(filename, keys = None, ** kwargs):
    import h5py
    
    def get_data(v):
        if v.dtype == object:
            if len(v.shape) == 0:   return np.array(v).item().decode()
            return v.asstr()[:].tolist()
        v = np.array(v)
        return v if v.ndim > 0 else v.item()
    
    def load_group(group):
        res = {}
        for k, v in group.items():
            if '\\' in k: k = k.replace('\\', '/')
            if logger.isEnabledFor(logging.DEBUG) and not isinstance(v, h5py.Group):
                logger.debug('Loading data for key {} : {}'.format(k, v))
            res[k] = get_data(v) if not isinstance(v, h5py.Group) else load_group(v)
        return res

    with h5py.File(filename, 'r') as file:
        if keys:
            return {k : get_data(file.get(k)) for k in keys if k in file}
        else:
            return load_group(file)

@load_data.dispatch('pkl')
def load_pickle(filename, ** kwargs):
    with open(filename, 'rb') as file:
        return pickle.load(file)

@load_data.dispatch(['txt', 'md'])
def load_txt(filename, encoding = 'utf-8', ** kwargs):
    with open(filename, 'r', encoding = encoding) as file:
        return file.read()

def _pd_read_method(name, ** defaults):
    def wrapped(* args, ** kwargs):
        import pandas as pd
        return getattr(pd, name)(* args, ** {** defaults, ** kwargs})
    
    if 'pandas' in sys.modules:
        import pandas as pd
        wrapped = wraps(getattr(pd, name))(wrapped)
    
    return wrapped

load_data.dispatch(np.load, 'npy')
load_data.dispatch(_pd_read_method('read_csv'), 'csv')
load_data.dispatch(_pd_read_method('read_csv', sep = '\t'), 'tsv')
load_data.dispatch(_pd_read_method('read_excel'), 'xlsx')
load_data.dispatch(_pd_read_method('read_pickle'), 'pdpkl')


@dispatch_wrapper(_dump_file_fn, 'Filename extension')
def dump_data(filename, data, overwrite = True, ** kwargs):
    """ Dumps `data` into `filename`. The saving function differ according to the extension. """
    if is_tensor(data): data = convert_to_numpy(data)
    ext = os.path.splitext(filename)[1][1:]
    
    if not ext:
        for types, default_ext in _default_ext.items():
            if types.__class__.__name__ in ('function', 'method') and types(data):
                filename, ext = '{}.{}'.format(filename, default_ext), default_ext
                break
            elif isinstance(data, types):
                filename, ext = '{}.{}'.format(filename, default_ext), default_ext
                break
        
        if not ext: filename, ext = '{}.pkl'.format(filename), 'pkl'
    
    if overwrite or not os.path.exists(filename):
        if ext not in _dump_file_fn:
            raise ValueError('Unhandled extention !\n  Accepted : {}\n  Got : {}'.format(
                tuple(_dump_file_fn.keys()), ext
            ))
        
        directory = os.path.dirname(filename)
        if directory: os.makedirs(directory, exist_ok = True)
        
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
def dump_h5(filename, data, mode = 'w', overwrite = False, ** kwargs):
    def _create_datasets(group, data):
        for k, v in data.items():
            if v is None: continue
            if '/' in k: k.replace('/', '\\')
            
            if isinstance(v, dict):
                _create_datasets(group.create_group(k), v)
                continue
            elif k in group and not overwrite:
                continue
            elif isinstance(v, list) and v and isinstance(v[0], list):
                from .sequence_utils import pad_batch
                v = pad_batch(v, pad_value = -1)
            elif not isinstance(v, np.ndarray):
                v = convert_to_numpy(v)
            
            try:
                dtype = None
                if 'str' in v.dtype.name or v.dtype == object:
                    dtype = h5py.string_dtype()
                    v = v.astype(dtype)
                
                logger.debug('Saving entry {} with dtype {}'.format(k, dtype))
                
                group.create_dataset(k, data = v, dtype = dtype)
            except Exception as e:
                logger.error('An error occured while saving entry {} : {}'.format(k, e))
                continue

    import h5py
    
    if hasattr(data, 'to_dict'): data = data.to_dict('list')
    
    with h5py.File(filename, mode) as file:
        _create_datasets(file, data)
    
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
    import pandas as pd
    return data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)

_default_ext    = {
    str     : 'txt',
    (list, tuple, set, dict, int, float) : 'json',
    np.ndarray      : 'npy',
    lambda x: hasattr(x, 'columns') : 'h5'
}