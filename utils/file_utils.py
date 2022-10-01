
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
import glob
import json
import pickle
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm

from utils.generic_utils import to_json, flatten, convert_to_str

try:
    from utils.image import _image_formats, _video_formats
    _image_ext = _image_formats + _video_formats
except ImportError:
    _image_ext = ()

try:
    from utils.audio import _audio_formats as _audio_ext
except ImportError:
    _audio_ext = ()

logger = logging.getLogger(__name__)

def normalize_filename(filename, invalid_mode = 'error'):
    """
        Return (list of) str filenames extracted from multiple formats
        
        Arguments :
            - filename  : (list of) filenames of different types
                - str   : return filename or list of filenames (if directory)
                - pd.DataFrame  : must have a 'filename' column
                - dict      : must have a 'filename' entry
                - tf.Tensor / np.ndarray    : string / bytes
    """
    if isinstance(filename, (dict, pd.Series)): filename = filename['filename']
    filename = convert_to_str(filename)
    
    if isinstance(filename, (list, tuple)):
        outputs = flatten([normalize_filename(f) for f in filename])
    elif isinstance(filename, pd.DataFrame):
        outputs = flatten([normalize_filename(row) for _, row in filename.iterrows()])
    elif isinstance(filename, str):
        if not os.path.isdir(filename): return filename
        outputs = flatten([normalize_filename(
            os.path.join(filename, f)
        ) for f in os.listdir(filename)])
    else:
        if invalid_mode == 'skip': return None
        if invalid_mode == 'keep' : return filename
        else:
            raise ValueError("Unknown type for `filename` : {}\n  {}".format(type(filename), filename))
        
    return [o for o in outputs if o is not None]

def download_file(url, filename = None, directory = None, buffer_size = 8192, sha256 = None):
    try:
        import urllib
    except ImportError as e:
        logger.error('Please install `urllib` to download files !')
        return None
    
    if not filename: filename = os.path.basename(url)
    if directory:
        os.makedirs(directory, exist_ok = True)
        filename = os.path.join(directory, filename)

    if os.path.exists(filename) and not os.path.isfile(filename):
        raise RuntimeError(f"{filename} exists but is not a regular file")

    if os.path.isfile(filename):
        if not sha256: return filename
        import hashlib
        
        with open(filename, 'rb') as file:
            if hashlib.sha256(file.read()).hexdigest() == sha256:
                return filename
        
        logger.warning('{} exists but has an invalid SHA256 : re-loading it'.format(filename))

    
    with urllib.request.urlopen(url) as source, open(filename, 'wb') as output_file:
        with tqdm(total = int(source.info().get("Content-Length")), unit = 'iB', unit_scale = True, unit_divisor = 1024) as loop:
            while True:
                buffer = source.read(buffer_size)
                if not buffer:
                    break

                output_file.write(buffer)
                loop.update(len(buffer))

    if sha256:
        import hashlib
        with open(filename, 'rb') as file:
            if hashlib.sha256(file.read()).hexdigest() != sha256:
                raise RuntimeError("The file has been downloaded but the SHA256 does not match")

    return filename

def load_data(filename, ** kwargs):
    ext = os.path.splitext(filename)[1][1:]
    
    if ext not in _load_file_fn:
        raise ValueError("Unhandled extension !\n  Accepted : {}\n  Got : {}".format(
            tuple(_load_file_fn.keys()), ext
        ))
    
    return _load_file_fn[ext](filename, ** kwargs)
    
def load_json(filename, default = {}, ** kwargs):
    """ Safely load data from a json file """
    if not filename.endswith('.json'): filename += '.json'
    if not os.path.exists(filename): return default
    with open(filename, 'r', encoding = 'utf-8') as file:
        result = file.read()
    return json.loads(result)

def load_npz(filename, ** kwargs):
    with np.load(filename) as file:
        data = {k : file[k] for k in file.files}
    return data

def load_pickle(filename, ** kwargs):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def load_txt(filename, encoding = 'utf-8', ** kwargs):
    with open(filename, 'r', encoding = encoding) as file:
        return file.read()

def _load_image(filename, ** kwargs):
    from utils.image import load_image
    
    return load_image(filename, ** kwargs)

def _load_audio(filename, rate = None, ** kwargs):
    from utils.audio import load_audio
    
    return load_audio(filename, rate = rate, ** kwargs)

def dump_data(filename, data, overwrite = True, ** kwargs):
    if isinstance(data, tf.Tensor): data = data.numpy()
    ext = os.path.splitext(filename)[1][1:]
    if not ext:
        for types, default_ext in _default_ext.items():
            if isinstance(data, types):
                filename += '.' + default_ext
                ext = default_ext
                break
        
        if not ext: filename, ext = '{}.pkl'.format(filename), 'pkl'
    
    if overwrite or not os.path.exists(filename):
        if ext not in _dump_file_fn:
            raise ValueError('Unhandled extention !\n  Accepted : {}\n  Got : {}'.format(
                tuple(_dump_file_fn.keys()), ext
            ))
        
        _dump_file_fn[ext](filename, data, ** kwargs)
    
    return filename

def dump_json(filename, data, ** kwargs):
    """ Safely save data to a json file """
    if not filename.endswith('.json'): filename += '.json'
    data = to_json(data)
    data = json.dumps(data, ** kwargs)
    with open(filename, 'w', encoding = 'utf-8') as file:
        file.write(data)

def dump_pickle(filename, data, ** kwargs):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def dump_txt(filename, data, ** kwargs):
    with open(filename, 'w', encoding = 'utf-8') as file:
        file.write(data)

_load_file_fn   = {
    ** {ext : _load_image for ext in _image_ext},
    ** {ext : _load_audio for ext in _audio_ext},
    'json'  : load_json,
    'txt'   : load_txt,
    'pkl'   : load_pickle,
    'npz'   : load_npz,
    'npy'   : np.load,
    'csv'   : pd.read_csv,
    'tsv'   : lambda filename, ** kwargs: pd.read_csv(filename, sep = '\t', ** kwargs)
}
_dump_file_fn   = {
    'json'  : dump_json,
    'txt'   : dump_txt,
    'pkl'   : dump_pickle,
    'csv'   : lambda filename, data, ** kwargs: data.to_csv(filename, ** kwargs),
    'tsv'   : lambda filename, data, ** kwargs: data.to_csv(filename, sep = '\t', ** kwargs),
    'xlsx'  : lambda filename, data, ** kwargs: data.to_excel(filename, ** kwargs),
    'npy'   : lambda filename, data, ** kwargs: np.save(filename, data, ** kwargs),
    'npz'   : lambda filename, data, ** kwargs: np.savez(filename, ** data)
}
_default_ext    = {
    str     : 'txt',
    (list, tuple, dict, int, float) : 'json',
    np.ndarray      : 'npy',
    pd.DataFrame    : 'csv'
}