import os
import numpy as np
import pandas as pd

from utils.generic_utils import load_json
from custom_train_objects.history import History

_pretrained_models_folder = 'pretrained_models'

def get_model_infos(name):
    return load_json(os.path.join(_pretrained_models_folder, name, 'config.json'), default = {})

def get_model_history(name):
    return History.load(os.path.join(_pretrained_models_folder, name, 'saving', 'historique.json'))

def get_model_config(name):
    return get_model_infos(name).get('config', {})

def infer_model_class(name, possible_class):
    if name is None or not isinstance(name, str): return None
    
    config = get_model_infos(name)
        
    return possible_class.get(config.get('class_name', ''), None)

def compare_models(names, skip_identical = False, order_by_uniques = False, epoch = 'last',
                   metric = 'val_loss', criteria_fn = np.argmin):
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
        
        if _metrics is None: _metrics = ['epochs'] + list(hist_i.history[-1].keys())
        
        hist_i = hist_i.history
        if epoch == 'last':
            epoch_i = len(hist_i) - 1
        elif epoch == 'first':
            epoch_i = 0
        elif epoch == 'best':
            epoch_i = criteria_fn([e[metric] for e in hist_i])
        elif isinstance(epoch, int):
            epoch_i = epoch
        metrics_i = hist_i[epoch_i]
        
        infos[name] = {
            'class' : infos_i['class_name'], ** infos_i['config'], ** metrics_i, 'epochs' : epoch_i
        }
        infos[name] = {k : v for k, v in infos[name].items() if not isinstance(v, str) or name not in v}
    
    infos = pd.DataFrame(infos).T
    
    if skip_identical:
        lengths     = {c : n_unique(c) for c in infos.columns}
        non_uniques = [k for k, v in lengths.items() if v != 1]
        
        infos = infos[non_uniques]
    
    if order_by_uniques:
        lengths     = {c : n_unique(c) for c in infos.columns if c not in _metrics}
        unhashable  = [k for k, v in lengths.items() if v == -1]
        lengths     = {k : v for k, v in lengths.items() if v != -1}
        order       = sorted(lengths.keys(), key = lambda c: lengths[c])
        
        infos = infos[order + _metrics + unhashable]
        infos = infos.sort_values(order)
    
    return infos
