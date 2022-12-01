
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
import enum
import logging
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

from utils import dump_json, load_json, to_json, plot_multiple

logger = logging.getLogger(__name__)

class Phase(enum.IntEnum):
    SLEEPING    = -1
    TRAINING    = 0
    VALIDATING  = 1
    TESTING     = 2

class History(tf.keras.callbacks.Callback):
    """
        History callback storing data in a list of dict :
        __history = {
            epoch : {
                metrics : {
                    metric_name : list of values,
                    ...
                },
                infos   : {...} # information on timing
            }
        }
        __trainings = [
            config  : {...} # training config given by this.set_params(...)
            infos   : {...} # timing informations
        ]
    """
    def __init__(self, filename = None, ** kwargs):
        super().__init__(** kwargs)
        self.filename   = filename
        self.__history  = {}
        self.__trainings    = []
        
        self.__phase    = Phase.SLEEPING
        self.__test_prefix  = None
        self.__current_training_config  = {}
        self.__current_training_infos   = {}
        
        self.__current_epoch    = -1
        self.__current_batch    = 0
        self.__current_epoch_history    = {}
        self.__current_epoch_infos      = {}
    
    @property
    def sleeping(self):
        return self.__phase == Phase.SLEEPING
    
    @property
    def training(self):
        return self.__phase in (Phase.TRAINING, Phase.VALIDATING)
    
    @property
    def validating(self):
        return self.__phase == Phase.VALIDATING
    
    @property
    def testing(self):
        return self.__phase == Phase.TESTING
    
    @property
    def current_epoch(self):
        if self.__current_epoch != -1:
            return self.__current_epoch
        return -1 if len(self.__history) == 0 else max(self.__history.keys())
    
    @property
    def epoch(self):
        return len(self.__history)
    
    @property
    def steps(self):
        return sum([
            infos['infos']['train_size'] for infos in self.__history.values()
        ])    
    
    @property
    def training_time(self):
        return sum([
            infos['infos']['time'] for infos in self.__history.values()
        ])    
    
    @property
    def trainings(self):
        return self.__trainings

    @property
    def trainings_config(self):
        return [t['config'] for t in self.__trainings]

    @property
    def trainings_infos(self):
        return [t['infos'] for t in self.__trainings]
    
    @property
    def logs(self):
        return [infos['infos'] for _, infos in self.__history.items()]
    
    @property
    def history(self):
        return [
            {m : v[-1] for m, v in infos['metrics'].items()}
            for _, infos in sorted(self.__history.items(), key = lambda i: i[0])
        ]
    
    @property
    def step_history(self):
        return [
            infos['metrics']
            for _, infos in sorted(self.__history.items(), key = lambda i: i[0])
        ]
    
    @property
    def training_logs(self):
        return {k : v[-1] for k, v in self.__current_epoch_history.items()}
    
    @property
    def metrics(self):
        """ Return dict of metrics by epochs {metric_name : {epoch : values}} """
        metrics = {}
        for epoch, infos in self.__history.items():
            for met, values in infos['metrics'].items():
                metrics.setdefault(met, {})
                metrics[met][epoch] = values
        return metrics
        
    @property
    def metric_names(self):
        return self.train_metrics + self.valid_metrics + self.test_metrics
    
    @property
    def train_metrics(self):
        metrics = []
        for _, infos in self.__history.items():
            metrics += infos['infos'].get('train_metrics', [])
        return list(set(metrics))
    
    @property
    def valid_metrics(self):
        metrics = []
        for _, infos in self.__history.items():
            metrics += [m for m in infos['metrics'].keys() if m.startswith('val')]
        return list(set(metrics))
    
    @property
    def test_metrics(self):
        metrics = []
        for _, infos in self.__history.items():
            metrics += [m for m in infos['metrics'].keys() if m.startswith('test')]
        return list(set(metrics))
    
    def __len__(self):
        return len(self.__history)
    
    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.history[idx]
        else:
            return self.metrics[idx]
    
    def __contains__(self, v):
        return v in self.metric_names
    
    def __str__(self):
        return "===== History =====\n{}".format(pd.DataFrame(self.history))
    
    def contains(self, metric_name, epoch = -1):
        if len(self) == 0: return False
        return any([metric_name == k for k, _ in self[epoch].items()])
    
    def pop(self, test_prefix, epoch = -1):
        if len(self) == 0: return None
        if epoch < 0: epoch = epoch + 1 + max(self.__history.keys())
        metrics = self.__history[epoch]['infos'].get('{}_metrics'.format(test_prefix), [])
        
        for k in list(self.__history[epoch]['infos'].keys()):
            if '_'.join(k.split('_')[:-1]) == test_prefix:
                print("Pop info {}".format(k))
                self.__history[epoch]['infos'].pop(k)
        
        for metric in metrics:
            print("Pop metric {}".format(metric))
            self.__history[epoch]['metrics'].pop(metric)
    
    def set_history(self, history, trainings):
        self.__history      = {int(k) : v for k, v in history.items()}
        self.__trainings    = trainings
    
    def set_params(self, params):
        self.__current_training_config.update(to_json(params))
        self.__test_prefix = params.get('test_prefix', None)
        if self.__test_prefix is not None and self.__test_prefix.endswith('_'):
            self.__test_prefix = self.__test_prefix[:-1]
    
    def get_epoch_config(self, epoch):
        for t in self.__trainings:
            if t['infos']['start_epoch'] < epoch and epoch <= t['infos']['final_epoch']:
                return t['config']
        raise ValueError("Unknown epoch : {}".format(epoch))
    
    def str_training(self, win_size = 0, **kwargs):
        if self.sleeping: return self.__str__()
        
        metrics = {k : v[-1] for k, v in self.__current_epoch_history.items()}
        return " - ".join(["{} : {:.4f}".format(k, v) for k, v in metrics.items()])
        
    def plot(self, show_valid_step = False, ** kwargs):
        if self.epoch == -1:
            logger.warning("No data to plot !")
            return
        
        history_with_none = {}
        for epoch_metrics in self.history:
            for metric in self.metrics:
                history_with_none.setdefault(metric, []).append(
                    epoch_metrics.get(metric, None)
                )
        
        step_train_history_with_none    = {}
        step_valid_history_with_none    = {}
        step_test_history_with_none     = {}
        
        train_vlines, valid_vlines, test_vlines = [], [], []
        for epoch, infos in self.__history.items():
            train_vlines.append(infos['infos'].get('train_size', 1))
            valid_vlines.append(infos['infos'].get('valid_size', 1))
            test_vlines.append(infos['infos'].get('test_size', 1))

            for metric in self.train_metrics:
                step_train_history_with_none.setdefault(metric, []).extend(
                    infos['metrics'].get(metric, [None] * infos['infos']['train_size'])
                )
            for metric in self.valid_metrics:
                step_valid_history_with_none.setdefault(metric, []).extend(
                    infos['metrics'].get(metric, [None] * infos['infos'].get('valid_size', 1))
                )
            for metric in self.test_metrics:
                step_test_history_with_none.setdefault(metric, []).extend(
                    infos['metrics'].get(metric, [None] * infos['infos'].get('test_size', 1))
                )
        train_vlines = np.cumsum(train_vlines)
        valid_vlines = np.cumsum(valid_vlines)
        test_vlines  = np.cumsum(test_vlines)

        plot_data = {
            'epoch_history_loss' : {
                'x' : {k : v for k, v in history_with_none.items() if 'loss' in k},
                'title'     : 'Loss over epoch',
                'xlabel'    : 'epoch',
                'ylabel'    : 'values'
            },
            'epoch_history_metric' : {
                'x' : {k : v for k, v in history_with_none.items() if 'loss' not in k},
                'title'     : 'Metrics over epoch',
                'xlabel'    : 'epoch',
                'ylabel'    : 'values'
            },
            'train_history_loss' : {
                'x' : {k : v for k, v in step_train_history_with_none.items() if 'loss' in k},
                'title'     : 'Train loss history',
                'xlabel'    : 'step',
                'ylabel'    : 'values',
                'vlines'    : train_vlines
            },
            'train_history_metric' : {
                'x' : {k : v for k, v in step_train_history_with_none.items() if 'loss' not in k},
                'title'     : 'Train metrics history',
                'xlabel'    : 'step',
                'ylabel'    : 'values',
                'vlines'    : train_vlines
            }
        }
        if show_valid_step:
            plot_data.update({
                'valid_history_loss' : {
                    'x' : {k : v for k, v in step_valid_history_with_none.items() if 'loss' in k},
                    'title'     : 'Valid loss history',
                    'xlabel'    : 'step',
                    'ylabel'    : 'values',
                    'vlines'    : valid_vlines
                },
                'valid_history_metric' : {
                    'x' : {k : v for k, v in step_valid_history_with_none.items() if 'loss' not in k},
                    'title'     : 'Valid metrics history',
                    'xlabel'    : 'step',
                    'ylabel'    : 'values',
                    'vlines'    : valid_vlines
                },
                'test_history_loss' : {
                    'x' : {k : v for k, v in step_test_history_with_none.items() if 'loss' in k},
                    'title'     : 'Test loss history',
                    'xlabel'    : 'step',
                    'ylabel'    : 'values',
                    'vlines'    : test_vlines
                },
                'test_history_metric' : {
                    'x' : {k : v for k, v in step_test_history_with_none.items() if 'loss' not in k},
                    'title'     : 'Test metrics history',
                    'xlabel'    : 'step',
                    'ylabel'    : 'values',
                    'vlines'    : test_vlines
                }
            })
        
        plot_data = {
            k : v for k, v in plot_data.items() if len(v['x']) > 0
        }
        
        kwargs['use_subplots']  = True
        kwargs.setdefault('x_size', 5)
        kwargs.setdefault('y_size', 3)
                
        plot_multiple(** plot_data, ** kwargs)
    
    def on_train_begin(self, logs = None):
        self.__phase    = Phase.TRAINING
        self.__current_training_infos   = {
            'start' : datetime.datetime.now(),
            'end'   : -1,
            'time'  : -1,
            'interrupted'   : False,
            'start_epoch'   : self.current_epoch,
            'final_epoch'   : -1
        }
        
        self.__trainings.append({
            'config'    : self.__current_training_config,
            'infos'     : self.__current_training_infos
        })
        
    def on_train_end(self, logs = None):
        assert self.training
        t_end = datetime.datetime.now()
        
        interrupted = False
        if len(self.__current_epoch_history) != 0:
            logger.info("Training interrupted at epoch {} !".format(self.current_epoch))
            self.on_epoch_end(self.current_epoch)
        
        self.__current_training_config  = {}
        self.__phase = Phase.SLEEPING
        
    def on_epoch_begin(self, epoch, logs = None):
        if epoch in self.__history:
            raise ValueError("Epoch {} already in history !".format(epoch))
        elif self.current_epoch != -1 and epoch != self.current_epoch and epoch != self.current_epoch + 1:
            raise ValueError("Expected to start epoch {} but got epoch {} !".format(self.current_epoch + 1, epoch))
                
        self.__current_epoch    = epoch
        self.__current_batch    = -1
        self.__current_epoch_history    = {}
        self.__current_epoch_infos  = {
            'start' : datetime.datetime.now(),
            'end'   : -1,
            'time'  : -1,
            'train_start'   : datetime.datetime.now(),
            'train_end'     : -1,
            'train_time'    : -1,
            'train_size'    : -1,
            'train_metrics' : []
        }
        
    def on_epoch_end(self, epoch, logs = None):
        if epoch != self.current_epoch:
            raise ValueError("Epoch {} does not match the current epoch {}".format(epoch, self.current_epoch))
        
        t_end = datetime.datetime.now()
        self.__current_epoch_infos.update({
            'end'   : t_end,
            'time'  : (t_end - self.__current_epoch_infos['start']).total_seconds()
        })
        
        if not self.__current_epoch_history:
            self.__current_epoch_infos.update({
                'train_end'     : t_end,
                'train_time'    : (t_end - self.__current_epoch_infos['train_start']).total_seconds(),
                'train_size'    : 1,
                'train_metrics' : list(logs.keys())
            })
            for metric, v in logs.items():
                self.__current_epoch_history.setdefault(metric, [v] if not isinstance(v, list) else v)
        
        self.__history[self.current_epoch] = {
            'metrics'   : self.__current_epoch_history if self.__current_epoch_history else logs,
            'infos'     : self.__current_epoch_infos
        }
        
        self.__current_training_infos.update({
            'end'   : t_end,
            'time'  : (t_end - self.__current_training_infos['start']).total_seconds(),
            'final_epoch'   : self.current_epoch
        })
        
        self.__current_epoch    = -1
        self.__current_epoch_infos      = {}
        self.__current_epoch_history    = {}
    
    def on_test_begin(self, logs = None):
        if self.training:
            default_prefix = 'valid'
            self.__phase    = Phase.VALIDATING
        else:
            default_prefix = 'test'
            self.__phase    = Phase.TESTING
            self.__history.setdefault(self.current_epoch, {'metrics' : {}, 'infos' : {}})
            self.__current_epoch_infos      = self.__history[self.current_epoch]['infos']
            self.__current_epoch_history    = self.__history[self.current_epoch]['metrics']
        
        if self.__test_prefix is None: self.__test_prefix = default_prefix
        
        self.__current_batch    = -1
        self.__current_epoch_infos.update({
            '{}_start'.format(self.__test_prefix)  : datetime.datetime.now(),
            '{}_end'.format(self.__test_prefix)    : -1,
            '{}_time'.format(self.__test_prefix)   : -1,
            '{}_size'.format(self.__test_prefix)   : -1,
            '{}_metrics'.format(self.__test_prefix)    : []
        })
                
    def on_test_end(self, logs = None):
        assert self.training or self.testing
        
        t_end = datetime.datetime.now()
        self.__current_epoch_infos.update({
            '{}_end'.format(self.__test_prefix)   : t_end,
            '{}_time'.format(self.__test_prefix)  : (t_end - self.__current_epoch_infos['{}_start'.format(self.__test_prefix)]).total_seconds()
        })
        
        if self.testing:
            self.__phase = Phase.SLEEPING
            self.__current_epoch_infos      = {}
            self.__current_epoch_history    = {}
    
    def on_train_batch_end(self, batch, logs = None):
        assert self.training
        if logs is None: return
        if isinstance(logs, dict): logs = logs.items()
        
        t = datetime.datetime.now()
        for metric, value in logs:
            self.__current_epoch_history.setdefault(metric, []).append(value)
            
            if metric not in self.__current_epoch_infos['train_metrics']:
                self.__current_epoch_infos['train_metrics'].append(metric)
                
        self.__current_epoch_infos.update({
            'train_end'     : t,
            'train_time'    : (t - self.__current_epoch_infos['train_start']).total_seconds(),
            'train_size'    : batch + 1
        })
    
    def on_test_batch_end(self, batch, logs = None):
        assert self.validating or self.testing
        if logs is None: return
        if isinstance(logs, dict): logs = logs.items()
        
        t = datetime.datetime.now()
        prefix = 'val' if self.training else self.__test_prefix
        for metric, value in logs:
            if metric.startswith('val_') and prefix == 'test':
                metric = metric.replace('val', prefix)
            elif not metric.startswith(prefix): metric = '{}_{}'.format(prefix, metric)
            
            self.__current_epoch_history.setdefault(metric, []).append(value)
            
            if metric not in self.__current_epoch_infos['{}_metrics'.format(self.__test_prefix)]:
                self.__current_epoch_infos['{}_metrics'.format(self.__test_prefix)].append(metric)
               
        self.__current_epoch_infos.update({
            '{}_end'.format(self.__test_prefix)     : t,
            '{}_time'.format(self.__test_prefix)    : (t - self.__current_epoch_infos['{}_start'.format(self.__test_prefix)]).total_seconds(),
            '{}_size'.format(self.__test_prefix)    : batch + 1
        })
            
    
    def json(self):
        return to_json({
            'history'   : self.__history,
            'trainings' : self.__trainings
        })
    
    def save(self, filename = None):
        if filename is None:
            filename = self.filename if self.filename is not None else 'historique.json'
        if not filename.endswith('.json'): filename += '.json'
        dump_json(filename, self.json())

    @classmethod
    def load(cls, filename = 'historique.json'):
        instance = cls(filename = filename)
        
        if os.path.exists(filename):
            hist = load_json(filename)
            
            instance.set_history(** hist)
        
        return instance
