
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
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from utils.generic_utils import to_json, flatten

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
    if isinstance(filename, tf.Tensor): filename = filename.numpy()
    if isinstance(filename, bytes): filename = filename.decode('utf-8')
    
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

def dump_data(filename, data, overwrite = False, ** kwargs):
    if isinstance(data, tf.Tensor): data = data.numpy()
    if isinstance(data, np.ndarray): filename += '.npy'
    elif isinstance(data, pd.DataFrame): filename += '.csv'
    elif isinstance(data, str): filename += '.txt'
    else: filename += '.pkl'
    
    if overwrite or not os.path.exists(filename):
        if isinstance(data, pd.DataFrame):
            data.to_csv(filename)
        elif isinstance(data, np.ndarray):
            np.save(filename, data)
        elif isinstance(data, str):
            dump_txt(filename, data)
        else:
            dump_pickle(filename, data)
    
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

_image_ext      = ('jpg', 'png', 'gif')
_audio_ext      = ('m4a', 'mp3', 'wav', 'flac', 'opus')

_load_file_fn   = {
    ** {ext : _load_image for ext in _image_ext},
    ** {ext : _load_audio for ext in _audio_ext},
    'json'  : load_json,
    'txt'   : load_txt,
    'pkl'   : load_pickle,
    'npy'   : np.load,
    'csv'   : pd.read_csv
}
