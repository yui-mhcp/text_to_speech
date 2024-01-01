# Copyright (C) 2022-now yui-mhcp project's author. All rights reserved.
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
import numpy as np
import pandas as pd

from utils import load_json
from custom_train_objects.history import History

_pretrained_models_folder = 'pretrained_models'

def get_models(pattern = None, model_class = None):
    """
        Return all models with `pattern` (unix-style) in their name and / or `model_class` (str) as `class_name`
    """
    names = os.listdir(_pretrained_models_folder) if pattern is None else [
        os.path.basename(f) for f in glob.glob(os.path.join(_pretrained_models_folder, pattern))
    ]
    
    names = [n for n in names if is_model_name(n)]
    if model_class is not None:
        if not isinstance(model_class, (list, tuple)): model_class = [model_class]
        names = [n for n in names if get_model_class(n) in model_class]
    return names

def get_model_dir(name, * args):
    return os.path.join(_pretrained_models_folder, name, * args)

def is_model_name(name):
    """ Check if the model `name` has a directory with `config.json` file """
    return os.path.exists(get_model_dir(name, 'config.json'))

def get_model_infos(name):
    if name is None: return {}
    if not isinstance(name, str):
        return {
            'class_name' : name.__class__.__name__,
            'config'     : name.get_config(with_trackable_variables = False)
        }
    return load_json(get_model_dir(name, 'config.json'), default = {})

def get_model_class(name):
    """ Return the (str) class of model named `name` """
    return get_model_infos(name).get('class_name', None)

def get_model_history(name):
    """ Return the `History` class for model `name` """
    return History.load(get_model_dir(name, 'saving', 'historique.json'))

def get_model_config(name):
    return get_model_infos(name).get('config', {})

def infer_model_class(name, possible_class):
    """
        Return the `class` object of model `name` given a dict of possible classes {class_name : class_object}
    """
    if name is None or not isinstance(name, str): return None
    
    config = get_model_infos(name)
        
    return possible_class.get(config.get('class_name', ''), None)

def remove_training_checkpoint(name):
    """ Remove checkpoints in `{model}/training-logs/checkpoints/*` """
    training_ckpt_dir = get_model_dir(name, 'training-logs', 'checkpoints')
    for file in os.listdir(training_ckpt_dir):
        os.remove(os.path.join(training_ckpt_dir, file))

def compare_models(names,
                   skip_identical   = False,
                   order_by_uniques = False,
                   add_training_config  = False,
                   
                   epoch        = 'last',
                   metric       = 'val_loss', # Only relevant if `epoch == 'best'`
                   criteria_fn  = np.argmin
                  ):
    """
        Given a list of names, put all their configuration with their metrics for the selected epoch, in a pd.DataFrame to compare all models
        If `add_training_config`, it will also add the training configuration of the kept epoch
        The selected epoch depends on `epoch` (last or best) and `metric` (if best epoch)
        The `order_by_unique` sorts the DataFrame columns in the increasing order of unique values such that the left-most column will have the less unique values (then the most common between all models)
    """
    def n_unique(c):
        try:
            return len(infos[c].dropna().unique())
        except TypeError:
            return -1
    
    _metrics = None

    infos = {}
    for name in names:
        if not os.path.exists(os.path.join(_pretrained_models_folder, name)): continue
        
        infos_i = get_model_infos(name)
        hist_i  = get_model_history(name)

        metrics_i, train_config_i, epoch_i = {}, {}, -1
        if len(hist_i) > 0:
            if _metrics is None: _metrics = ['epochs'] + list(hist_i.history[-1].keys())

            if epoch == 'last':
                epoch_i = len(hist_i) - 1
            elif epoch == 'first':
                epoch_i = 0
            elif epoch == 'best':
                epoch_i = criteria_fn([e[metric] for e in hist_i])
            elif isinstance(epoch, int):
                epoch_i = epoch

            metrics_i       = hist_i.history[epoch_i]
            train_config_i  = {} if not add_training_config else hist_i.get_epoch_config(epoch_i)
        train_config_i.pop('epoch', None)
        
        infos[name] = {
            'class' : infos_i['class_name'], 'epochs' : epoch_i,
            ** infos_i['config'], ** metrics_i, ** train_config_i
        }
        infos[name] = {k : v for k, v in infos[name].items() if not isinstance(v, str) or name not in v}
    
    infos = pd.DataFrame(infos).T
    
    if skip_identical:
        lengths     = {c : n_unique(c) for c in infos.columns}
        non_uniques = [k for k, v in lengths.items() if v not in (0, 1) or k in _metrics]
        
        infos = infos[non_uniques]
    
    if order_by_uniques:
        lengths     = {c : n_unique(c) for c in infos.columns if c not in _metrics}
        unhashable  = [k for k, v in lengths.items() if v == -1]
        lengths     = {k : v for k, v in lengths.items() if v != -1}
        order       = sorted(lengths.keys(), key = lambda c: lengths[c])
        
        infos = infos[order + _metrics + unhashable]
        infos = infos.sort_values(order)
    
    return infos
