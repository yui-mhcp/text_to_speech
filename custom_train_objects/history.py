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
import time
import enum
import keras
import logging
import collections
import numpy as np
import pandas as pd

from utils import dump_json, load_json, to_json, plot_multiple

logger = logging.getLogger(__name__)

_max_vlines = 25
_default_plot_config    = {
    'title' : 'History',
    'x_size'    : 5,
    'y_size'    : 3
}

_config_keys_to_exclude = ('x', 'y', 'validation_data', 'callbacks')

class Phase(enum.IntEnum):
    SLEEP   = -1
    TRAIN   = 0
    VALID   = 1
    TEST    = 2

class History(keras.callbacks.Callback):
    """
        History callback storing data in a list of dict :
        __history = {
            epoch : {
                metrics : {
                    metric_name : list of values,
                    ...
                },
                infos   : {...} # time-related information
            }
        }
        __trainings = [
            config  : {...} # training config given by self.set_config(...)
            infos   : {...} # time-related information
        ]
    """
    def __init__(self, filename = None, ** kwargs):
        super().__init__(** kwargs)
        self.filename   = filename
        self.__history  = collections.OrderedDict()
        self.__trainings    = []
        
        self.__phase    = Phase.SLEEP
        self.__test_prefix  = None
        
        self.__current_epoch_infos      = {}
        self.__current_epoch_history    = {}

        self.__current_training_logs    = {}
        self.__current_training_infos   = {}
        self.__current_training_config  = {}
    
    phase   = property(lambda self: self.__phase)
    epochs  = property(lambda self: len(self))
    steps   = property(lambda self: sum(epoch['infos']['train_size'] for epoch in self))
    training_time   = property(lambda self: sum(epoch['infos']['time'] for epoch in self))
    
    is_training = property(lambda self: self.phase in (Phase.TRAIN, Phase.VALID))
    is_validating   = property(lambda self: self.phase == Phase.VALID)
    is_testing  = property(lambda self: self.phase == Phase.TEST)
    is_evaluating   = property(lambda self: self.is_testing or self.is_validating)
    
    training_logs   = property(lambda self: [t['infos'] for t in self.__trainings])
    training_config = property(lambda self: [t['config'] for t in self.__trainings])
    
    @property
    def history(self):
        """ Returns a list containing the metrics for each epoch """
        return [
            {met : vals[-1] for met, vals in epoch['metrics'].items()} for epoch in self
        ]
    
    @property
    def training_logs(self):
        """ Return logs for each training round (e.g., time information) """
        logs = []
        for t in self.__trainings:
            logs.append(t.get('logs', t['infos']))
        return logs
    
    @property
    def training_config(self):
        """ Return training configuration (i.e., training hyperparameters) for each round """
        return [t['config'] for t in self.__trainings]

    @property
    def training_infos(self):
        """ Return training logs + additional information (e.g., dataset summary) """
        infos = []
        for t in self.__trainings:
            infos.append({** t.get('logs', {}), ** t['infos']})
        return infos
    
    @property
    def metrics(self):
        """ Return a `dict` of metrics `{metric : values_for_every_epochs}` """
        metrics = {}
        for i, epoch in enumerate(self.history):
            for metric, value in epoch.items():
                metrics.setdefault(metric, {}).update({i : value})
        return metrics
    
    def __len__(self):
        if not self.__history: return 0
        return max(self.__history.keys()) + 1
    
    def __iter__(self):
        return iter(self.__history.values())
    
    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.history[idx]
        elif isinstance(idx, str):
            if len(self) == 0: raise RuntimeError('The history does not contain any metric')
            
            metrics = self.metrics
            return metrics[idx] if idx in metrics else self[-1].get(idx, None)
        raise ValueError('Unsupported index {}'.format(idx))
    
    def __str__(self):
        return "===== History =====\n{}".format(pd.DataFrame(self.history))
    
    def __repr__(self):
        return '<History epochs={} steps={}>'.format(self.epochs, self.steps)
    
    def set_history(self, history, trainings):
        history = {int(k) : v for k, v in history.items()}
        for ep, infos in sorted(history.items()):
            self.__history[ep] = infos
        self.__trainings = trainings
        
    def set_config(self, hparams, config, dataset_infos = {}, ** kwargs):
        self.__current_training_infos   = {
            'dataset' : dataset_infos, 'additional_infos' : kwargs.pop('additional_infos', {})
        }
        self.__current_training_config  = {
            ** kwargs,
            ** hparams,
            'epochs'    : config['epochs'],
            'initial_epoch' : config['initial_epoch']
        }
    
    def get_best(self, metric):
        if len(self) == 0: return None
        
        metrics = self.metrics
        if metric not in metrics: return None
        
        vals = list(metrics[metric].values())
        if len(vals) == 1:      return vals[0]
        if 'loss' in metric:    return min(vals)
        
        train_vals = list(metrics.get(metric.replace('val_', ''), metrics[metric]).values())
        if train_vals[-1] < train_vals[0]:
            return min(vals)
        return max(vals)
        
    def init_epoch_infos(self, prefix):
        self.__current_epoch_infos.update({
            f'{prefix}_start'   : time.time(),
            f'{prefix}_end'     : -1,
            f'{prefix}_time'    : -1,
            f'{prefix}_size'    : -1,
            f'{prefix}_metrics' : []
        })

    def update_epoch_infos(self, prefix, batch, metrics):
        t = time.time()
        self.__current_epoch_infos.update({
            f'{prefix}_end' : t,
            f'{prefix}_time'    : t - self.__current_epoch_infos[f'{prefix}_start'],
            f'{prefix}_size'    : batch + 1,
            f'{prefix}_metrics' : metrics
        })

    def on_train_begin(self, logs = None):
        self.__phase = Phase.TRAIN
        self.__current_training_logs   = {
            'start' : time.time(),
            'end'   : -1,
            'time'  : -1,
            'interrupted'   : True,
            'start_epoch'   : self.epochs,
            'final_epoch'   : -1
        }
        
        self.__trainings.append({
            'config'    : self.__current_training_config,
            'logs'      : self.__current_training_logs,
            'infos'     : self.__current_training_infos
        })

    
    def on_train_end(self, logs = None):
        if not self.is_training: return
        
        interrupted = False
        if self.__current_epoch_history:
            logger.info("Training interrupted at epoch {} !".format(self.epoch))
            self.on_epoch_end(self.epochs)
            interrupdated   = True
        
        t_end   = time.time()
        self.__current_training_logs.update({
            'end'   : t_end,
            'time'  : t_end - self.__current_training_logs['start'],
            'interrupted'   : interrupted,
            'final_epoch'   : self.epochs
        })

        self.__phase = Phase.SLEEP
        self.__current_training_logs    = None
        self.__current_training_infos   = {}
        self.__current_training_config  = {}
    
    def on_test_begin(self, logs = None):
        if self.is_training:
            default_prefix = 'val'
            self.__phase    = Phase.VALID
        else:
            default_prefix = 'test'
            self.__phase    = Phase.TEST
            self.__history.setdefault(self.epochs - 1, {'metrics' : {}, 'infos' : {}})
            self.__current_epoch_infos      = self.__history[self.epochs - 1]['infos']
            self.__current_epoch_history    = self.__history[self.epochs - 1]['metrics']
        
        if not self.__test_prefix: self.__test_prefix = default_prefix
        self.init_epoch_infos(self.__test_prefix)

    def on_test_end(self, logs = None):
        if not self.is_evaluating: raise RuntimeError('`on_test_begin` has not been called')
        
        prefix  = self.__test_prefix
        t_end   = time.time()
        self.__current_epoch_infos.update({
            f'{prefix}_end'   : t_end,
            f'{prefix}_time'  : t_end - self.__current_epoch_infos[f'{prefix}_start']
        })
        
        if self.is_testing:
            self.__phase = Phase.SLEEP
            self.__current_epoch_infos      = {}
            self.__current_epoch_history    = {}
        else:
            self.__phase = Phase.TRAIN
    
    def on_epoch_begin(self, epoch, logs = None):
        if epoch != self.epochs:
            raise RuntimeError('Unexpected epoch {} - expected is {}'.format(
                epoch, self.epochs
            ))
        
        self.__current_epoch_history    = {}
        self.__current_epoch_infos  = {
            'start' : time.time(), 'end' : -1, 'time' : -1
        }
        self.init_epoch_infos('train')
    
    def on_epoch_end(self, epoch, logs = None):
        if epoch != self.epochs:
            raise RuntimeError('Unexpected epoch {} - expected is {}'.format(
                epoch, self.epochs
            ))
        
        if not self.__current_epoch_history:
            self.on_train_batch_end(0, logs)
        
        t = time.time()
        self.__current_epoch_infos.update({
            'end' : t, 'time' : t - self.__current_epoch_infos['start']
        })

        self.__history[self.epochs] = {
            'metrics' : self.__current_epoch_history, 'infos' : self.__current_epoch_infos
        }
        
        self.__current_training_logs.update({
            'end'   : t,
            'time'  : t - self.__current_training_logs['start'],
            'final_epoch'   : self.epochs
        })
        
        self.__current_epoch_infos      = {}
        self.__current_epoch_history    = {}

    def on_train_batch_end(self, batch, logs = None):
        if not self.is_training: raise RuntimeError('`on_train_begin` has not been called')
        if not logs: logs = {}
        
        self.update_epoch_infos('train', batch, list(logs.keys()))
        for metric, value in logs.items():
            self.__current_epoch_history.setdefault(metric, []).extend(
                value if isinstance(value, list) else [value]
            )       

    
    def on_test_batch_end(self, batch, logs = None):
        if not self.is_evaluating: raise RuntimeError('`on_test_begin` has not been called yet')
        if not logs: logs = {}
        
        prefix = self.__test_prefix
        metrics = []
        for metric, value in logs.items():
            if metric.startswith('val_') and self.is_testing:
                metric = metric.replace('val', prefix)
            elif not metric.startswith(prefix):
                metric = f'{prefix}_{metric}'
            
            metrics.append(metric)
            self.__current_epoch_history.setdefault(metric, []).append(value)
        
        self.update_epoch_infos(prefix, batch, metrics)
    
    def plot(self, ** kwargs):
        if not self.__history:
            logger.warning("No data to plot !")
            return

        epochs = [idx + 1 for idx in self.__history.keys()]
        history_with_none = {}
        for metric, values in self.metrics.items():
            history_with_none[metric] = [None] * len(self.__history)
            for epoch, val in values.items():
                history_with_none[metric][epoch] = val
        
        step    = 1
        index   = 0
        steps   = []
        vlines  = []
        step_history_with_none  = {
            k : [None] * (self.steps + self.epochs - 1) for k in history_with_none.keys()
        }
        for idx, epoch in self.__history.items():
            n = epoch['infos']['train_size']
            for metric in epoch['infos']['train_metrics']:
                step_history_with_none[metric][index : index + n] = epoch['metrics'][metric]

            steps.extend(list(range(step, step + n)) + [step + n])
            step += n
            vlines.append(step)
            
            index += n + 1
        step_history_with_none = {
            k : v for k, v in step_history_with_none.items() if any(vi is not None for vi in v)
        }
        
        plot_data = {
            'epoch_history_loss' : {
                'x' : epochs,
                'y' : {k : v for k, v in history_with_none.items() if 'loss' in k},
                'title'     : 'Loss over epochs',
                'xlabel'    : 'epoch',
                'ylabel'    : 'values'
            },
            'epoch_history_metric' : {
                'x' : epochs,
                'y' : {k : v for k, v in history_with_none.items() if 'loss' not in k},
                'title'     : 'Metrics over epochs',
                'xlabel'    : 'epoch',
                'ylabel'    : 'values'
            },
            'step_history_loss' : {
                'x' : steps[:-1],
                'y' : {k : v for k, v in step_history_with_none.items() if 'loss' in k},
                'title'     : 'Loss over steps',
                'xlabel'    : 'step',
                'ylabel'    : 'values',
                'vlines'    : vlines[:-1] if len(vlines) < _max_vlines else None
            },
            'step_history_metric' : {
                'x' : steps[:-1],
                'y' : {k : v for k, v in step_history_with_none.items() if 'loss' not in k},
                'title'     : 'Metrics over steps',
                'xlabel'    : 'step',
                'ylabel'    : 'values',
                'vlines'    : vlines[:-1] if len(vlines) < _max_vlines else None
            }
        }
        
        plot_data = {k : v for k, v in plot_data.items() if len(v['y']) > 0}
        
        kwargs['use_subplots']  = True
        for k, v in _default_plot_config.items():
            kwargs.setdefault(k, v)
                
        plot_multiple(** plot_data, ** kwargs)
        
    def save(self, filename = None, ** _):
        if not filename: filename = self.filename if self.filename else 'history.json'
        dump_json(filename, {'history' : self.__history, 'trainings' : self.__trainings})

    @classmethod
    def load(cls, filename = 'history.json'):
        instance = cls(filename = filename)
        
        if os.path.exists(filename):
            instance.set_history(** load_json(filename))
        
        return instance
