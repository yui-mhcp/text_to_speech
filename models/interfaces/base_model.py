
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
import copy
import time
import shutil
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm as tqdm_progress_bar
from tensorflow.keras.models import load_model, model_from_json

from tensorflow.python.util import nest
from tensorflow.python.eager import def_function
from tensorflow.python.keras.callbacks import CallbackList

from loggers import DEV
from hparams import HParams, HParamsTraining, HParamsTesting
from datasets import train_test_split, prepare_dataset
from utils import time_to_string, load_json, dump_json, get_metric_names, map_output_names
from custom_architectures import get_architecture, custom_objects
from custom_train_objects import History, MetricList
from custom_train_objects import get_optimizer, get_loss, get_metrics, get_callbacks
from models.weights_converter import partial_transfer_learning
from models.model_utils import _pretrained_models_folder, is_model_name, get_model_dir

_trackable_objects = (
    tf.keras.Model, 
    tf.keras.losses.Loss, 
    tf.keras.optimizers.Optimizer, 
    tf.keras.metrics.Metric
)

class ModelInstances(type):
    _instances = {}
    _is_restoring   = False
    def __call__(cls, * args, ** kwargs):
        nom = kwargs.get('nom', None)
        if nom is None: nom = cls.__name__
        if nom in cls._instances:
            pass
        elif not cls._is_restoring and kwargs.get('restore', True) and os.path.exists(os.path.join(_pretrained_models_folder, nom, 'config.json')):
            cls._is_restoring = True
            cls.restore(nom)
            cls._is_restoring = False
        else:
            #cls._is_restoring = False
            instance = super(ModelInstances, cls).__call__(* args, ** kwargs)
            cls._instances[nom] = instance
        return cls._instances[nom]

class BaseModel(metaclass = ModelInstances):
    def __init__(self,
                 nom    = None,
                 restore    = True,
                 max_to_keep    = 3,
                 pretrained_name    = None,
                 
                 ** kwargs
                ):
        """
        Constructor that initialize the model's configuration, architecture, folders, ... 
        """
        # Allows to pass all kwargs to super-classes 
        kwargs  = {k : v for k, v in kwargs.items() if not hasattr(self, k)}
        
        self.nom    = nom if nom is not None else self.__class__.__name__
        self.pretrained_name    = pretrained_name
        
        self.__history  = History.load(self.history_file)
        
        self.__models        = {}
        self.__losses        = {}
        self.__optimizers    = {}
        self.__metrics       = {}
        
        self.__ckpt     = tf.train.Checkpoint()
        self.__ckpt_manager = tf.train.CheckpointManager(
            self.__ckpt, 
            directory   = self.save_dir,
            max_to_keep = max_to_keep
        )
        
        self.backend_kwargs = kwargs        
        if restore and _can_restore(restore, self.config_file):
            restore_kwargs  = {} if not isinstance(restore, dict) else restore
            if isinstance(restore, str): restore_kwargs = {'directory' : restore}
            
            if 'directory' in restore_kwargs:
                if is_model_name(restore_kwargs['directory']):
                    restore_kwargs['directory'] = os.path.join(
                        _pretrained_models_folder, restore_kwargs['directory'], 'saving'
                    )
                self.__history  = History.load(
                    os.path.join(restore_kwargs['directory'], 'historique.json')
                )
            self.restore_models(** restore_kwargs)
            if restore not in (True, False) and os.path.exists(self.folder):
                self.save()
        else:
            self._build_model(** kwargs)
                
        if not os.path.exists(self.folder):
            self._init_folders()
            self.save()
        
        self.__build_call_fn()
        
        self.init_train_config()
        
        logging.info("Model {} initialized successfully !".format(self.nom))
    
    def __build_call_fn(self):
        if not hasattr(self, 'call_fn'):
            if hasattr(self, 'call'): call_fn = self.call
            else:
                model = self.get_model()

                def call_fn(inputs, training = False):
                    return model(inputs, training = training)
            
            self.call_fn = _compile_fn(
                call_fn,
                run_eagerly = self.run_eagerly,
                signature   = None, #self.call_signature if hasattr(self, 'call_signature') else self.input_signature,
                include_signature   = self.include_signature
            )
            
    def _init_folders(self):
        """ Initialize default folders """
        os.makedirs(self.train_dir,     exist_ok=True)
        os.makedirs(self.ckpt_dir,      exist_ok=True)
        os.makedirs(self.train_test_dir, exist_ok=True)
        os.makedirs(self.eval_dir,      exist_ok=True)
        os.makedirs(self.pred_dir,      exist_ok=True)
        os.makedirs(self.save_dir,      exist_ok=True)
    
    def _build_model(self, **kwargs):
        """
            Initialize models' variables
            Arguments :
                - kwargs where keys are models' variable name and values are their configuration passed to the `get_architecture` method
                    Can be either a `tf.keras.Model` or a dict of config. 
                    If it is a dict it must contain the key `architecture_name`
        """
        def _build_single_model(model):
            if isinstance(model, tf.keras.Model):
                return model
            elif isinstance(model, dict):
                return get_architecture(** model)
            else:
                raise ValueError("Impossible d'initialiser le modèle !\nReçu : {}\nAcceptés : tf.keras.Model ou dict")
        
        def _set_single_model(model, name = None):
            if isinstance(model, dict):
                [_set_single_model(m, n) for n, m in model.items()]
            elif isinstance(model, (tuple, list)):
                [_set_single_model(m, 'model_{}'.format(i)) for i, m in enumerate(model)]
            elif isinstance(model, tf.keras.Model):
                if name is None: name = 'model'
                setattr(self, name, model)
            else:
                raise ValueError("Modele de type inconnu : {} (type : {})".format(model, type(model)))
        
        for name, model_config in kwargs.items():
            models = _build_single_model(model_config)
            _set_single_model(models, name = name)
            
    def init_train_config(self, global_batch_size = None, ** kwargs):
        """
            Initialize training configuration
            Can take as argument whathever you put in the `training_hparams` property
        """
        
        self.stop_training  = False
        self.current_epoch  = tf.Variable(self.epochs, trainable = False, name = 'epoch')
        self.current_step   = tf.Variable(self.steps, trainable = False, name = 'step')
        self.global_batch_size  = -1
        
        mapper = self.training_hparams_mapper
        
        for k, v in self.training_hparams.items():
            val = kwargs.get(k, v)
            if k in mapper:
                mapper[k](val)
            elif not hasattr(self, k) or val is not None:
                setattr(self, k, val)
    
    def _update_augment_prct(self, augment_prct):
        self.augment_prct_scheduler = augment_prct if callable(augment_prct) else None
        if self.augment_prct_scheduler:
            self.augment_prct = tf.Variable(
                0., trainable = False, name = 'augment_prct', dtype = tf.float32
            )
        else:
            self.augment_prct = tf.constant(tf.cast(augment_prct, tf.float32))
    
    def update_train_config(self, epoch, step):
        """ Update training configuration after each step / epoch """
        self.current_epoch.assign(epoch)
        self.current_step.assign(step)
        
        if self.augment_prct_scheduler:
            self.augment_prct.assign(self.augment_prct_scheduler(
                epoch = self.current_epoch, step = self.current_step,
                old_value = self.augment_prct
            ))
    
    @property
    def folder(self):
        return os.path.join(_pretrained_models_folder, self.nom)
    
    @property
    def save_dir(self):
        return os.path.join(self.folder, "saving")
        
    @property
    def train_dir(self):
        return os.path.join(self.folder, "training-logs")
    
    @property
    def eval_dir(self):
        return os.path.join(self.folder, "eval")
    
    @property
    def pred_dir(self):
        return os.path.join(self.folder, "outputs")
    
    @property
    def ckpt_dir(self):
        return os.path.join(self.train_dir, "checkpoints")
    
    @property
    def train_test_dir(self):
        return os.path.join(self.train_dir, "eval")
        
    
    @property
    def config_file(self):
        return os.path.join(self.folder, "config.json")
    
    @property
    def history_file(self):
        return os.path.join(self.save_dir, "historique.json")
        
    @property
    def ckpt_format(self):
        return os.path.join(self.ckpt_dir, "checkpoint-{epoch:04d}.ckpt")
    
    @property
    def config_models_file(self):
        return os.path.join(self.save_dir, "config_models.json")
        
    @property
    def run_eagerly(self):
        return False
    
    @property
    def include_signature(self):
        strat = self.distribute_strategy
        return False if strat is not None and strat.num_replicas_in_sync > 1 else True
    
    @property
    def input_signature(self):
        shape = self.input_shape
        if isinstance(shape, dict):
            raise NotImplementedError("When you have multiple submodels you must define `input_signature`")
        elif isinstance(shape, list):
            return tuple([tf.TensorSpec(shape = s, dtype = tf.float32) for s in shape])
        return tf.TensorSpec(shape = shape, dtype = tf.float32)
            
    @property
    def output_signature(self):
        shape = self.output_shape
        if isinstance(shape, dict):
            raise NotImplementedError("When you have multiple submodels you must define `output_signature`")
        elif isinstance(shape, list):
            return (tf.TensorSpec(shape = s, dtype = tf.float32) for s in shape)
        return tf.TensorSpec(shape = shape, dtype = tf.float32)
    
    @property
    def distribute_strategy(self):
        if hasattr(self.get_model(name = self.model_names[0]), 'distribute_strategy'):
            return self.get_model(name = self.model_names[0]).distribute_strategy
        else:
            return None
    
    @property
    def inputs(self):
        if len(self.__models) == 1:
            return self.get_model().inputs
        else:
            return {name : model.inputs for name, model in self.models.items()}
    
    @property
    def outputs(self):
        if len(self.__models) == 1:
            return self.get_model().outputs
        else:
            return {name : model.outputs for name, model in self.models.items()}
    
    @property
    def input_shape(self):
        if len(self.__models) == 1:
            return self.get_model().input_shape
        else:
            return {name : model.input_shape for name, model in self.models.items()}
    
    @property
    def output_shape(self):
        if len(self.__models) == 1:
            return self.get_model().output_shape
        else:
            return {name : model.output_shape for name, model in self.models.items()}
        
    @property
    def model_names(self):
        return list(self.__models.keys())
        
    @property
    def models(self):
        return {name : infos['model'] for name, infos in self.__models.items()}
    
    @property
    def model_infos(self):
        return self.__models
    
    @property
    def default_metrics_config(self):
        return {}
    
    @property
    def layers(self):
        if len(self.__models) == 1:
            return self.get_model().layers
        else:
            return {name : model.layers for name, model in self.models.items()}
    
    @property
    def optimizers(self):
        return self.__optimizers
    
    @property
    def losses(self):
        return self.__losses
    
    @property
    def metrics(self):
        return list(self.__metrics.values())
            
    @property
    def metric_names(self):
        loss_names      = get_metric_names(self.losses)
        metric_names    = get_metric_names(self.__metrics)
        if len(loss_names) == 1:
            loss_names = ['loss']
        elif len(loss_names) > 1 and 'loss' not in loss_names:
            loss_names = ['loss'] + loss_names
        
        return loss_names + metric_names
    
    @property
    def history(self):
        return self.__history
    
    @property
    def step_history(self):
        return self.__history.step_history
    
    @property
    def epochs(self):
        return len(self.__history)
    
    @property
    def steps(self):
        return self.__history.steps
    
    @property
    def custom_objects(self):
        return custom_objects
    
    @property
    def training_hparams(self):
        return HParams(augment_prct = 0.25)
    
    @property
    def training_hparams_mapper(self):
        return {'augment_prct' : self._update_augment_prct}
    
    @property
    def trainable_variables(self):
        if len(self.__models) == 1:
            return self.get_model().trainable_variables
        else:
            return {name : model.trainable_variables for name, model in self.models.items()}
    
    @property
    def variables(self):
        var = []
        for _, model in self.models.items():
            var += model.variables
        return var
    
    @property
    def list_trainable_variables(self):
        var = []
        for _, model in self.models.items():
            var += model.trainable_variables
        return var
    
    @property
    def checkpoint(self):
        return self.__ckpt
    
    @property
    def ckpt_manager(self):
        return self.__ckpt_manager
    
    @property
    def latest_checkpoint(self):
        return self.__ckpt_manager.latest_checkpoint
    
    @property
    def latest_train_checkpoint(self):
        return tf.train.latest_checkpoint(self.ckpt_dir)
            
    def is_compiled(self, model_name):
        return self.__models[model_name]['compiled']
    
    def add_loss(self, loss, name = 'loss', **kwargs):
        if isinstance(loss, str):
            loss = get_loss(loss, **kwargs)
        setattr(self, name, loss)
    
    def add_optimizer(self, optimizer, name = 'optimizer', **kwargs):
        if isinstance(optimizer, str):
            optimizer = get_optimizer(optimizer, **kwargs)
        setattr(self, name, optimizer)
        
    def add_metric(self, metric, name = 'metric', prefix = None, **kwargs):
        if isinstance(metric, str):
            metric = get_metrics(metric, **kwargs)
        if isinstance(metric, list):
            if len(metric) == 1 and not isinstance(name, (list, tuple)): name = [name]
            for i, m in enumerate(metric):
                n = name[i] if isinstance(name, (list, tuple)) else m.__class__.__name__
                if prefix is not None: n = prefix + n
                self.add_metric(m, name = n)
        else:
            setattr(self, name, metric)
        
    def _init_trackable_variable(self, name, var):
        if not isinstance(var, _trackable_objects):
            raise ValueError("Le type '{}' n'est pas une variable trackee !".format(type(var)))
            
        if hasattr(self, name):
            raise ValueError("La variable '{}' existe déjà et est une variable trackee ! Elle ne peut donc pas être modifiée. ".format(name))
            
        if isinstance(var, tf.keras.Model):
            self._init_model(name, var)
        elif isinstance(var, tf.keras.optimizers.Optimizer):
            self._init_optimizer(name, var)
        elif isinstance(var, tf.keras.losses.Loss):
            self._init_loss(name, var)
        elif isinstance(var, tf.keras.metrics.Metric):
            self._init_metric(name, var)
        
    def _init_model(self, name, model):
        assert isinstance(model, tf.keras.Model), "'model' doit être un tf.keras.Model !"
        if name in self.__models:
            logging.warning("Submodel '{}' already exists !".format(name))
            return
        
        logging.info("Initializing submodel : {} !".format(name))
        
        config_model_file = os.path.join(self.save_dir, name + '.json')
        
        self.__models[name] = {
            'save_path' : config_model_file,
            'model'     : model,
            'compiled'  : False,
            'optimizer_name'    : None,
            'metrics_name'      : None,
            'loss_name'         : None
        }
        setattr(self.__ckpt, name, model)
        
    def _init_optimizer(self, name, optimizer):
        assert isinstance(optimizer, tf.keras.optimizers.Optimizer), "'optimizer' doit êtreun tf.keras.optimizers.Optimizer !"
        
        if name in self.__optimizers:
            logging.warning("Optimizer '{}' already exists !".format(name))
            return
        
        logging.info("Optimizer '{}' initilized successfully !".format(name))
        
        self.__optimizers[name] = optimizer
        setattr(self.__ckpt, name, optimizer)
                
    def _init_loss(self, name, loss):
        assert isinstance(loss, tf.keras.losses.Loss) or callable(loss), "'loss' doit êtreun tf.keras.losses.Loss ou un callable !"
        
        if name in self.__losses:
            logging.warning("Loss '{}' already exists !".format(name))
            return
        
        self.__losses[name] = loss
        
    def _init_metric(self, name, metric):
        assert isinstance(metric, tf.keras.metrics.Metric) or callable(metric), "'metric' doit êtreun tf.keras.metrics.Metric ou un callable !"
        
        if name in self.__metrics:
            logging.warning("Metric '{}' already exists !".format(name))
            return
        
        self.__metrics[name] = metric
        
    def get_model(self, name = None):
        if name is None and len(self.__models) > 1:
            raise ValueError("Pour récupérer un modèle quand il existe plusieurs sous-modèles, il faut l'identifier par son nom ! ou alors instancier un 'full_model' combinant les différents sous-modèles")
                
        name = name if name is not None else self.model_names[0]
        
        if name not in self.__models:
            raise ValueError("Le sous-modèle n'existe pas !\nReçu : {}\nAcceptés : {}".format(name, self.model_names))
            
        return self.__models[name]['model']
    
    def get_optimizer(self, model_name = None, opt_name = None):
        if opt_name is None:
            if model_name is not None:
                opt_name = self.__models[model_name].get('optimizer_name', None)
            elif len(self.optimizers) == 1:
                opt_name = list(self.optimizers.keys())[0]
        
        return self.optimizers[opt_name] if opt_name is not None else None
    
    def get_loss(self, model_name = None, loss_name = None):
        if loss_name is None:
            if model_name is not None:
                loss_name = self.__models[model_name].get('loss_name', None)
            elif len(self.losses) == 1:
                loss_name = list(self.losses.keys())[0]
        
        return self.losses[loss_name] if loss_name is not None else None
    
    def get_metric(self, model_name = None, met_name = None):
        if met_name is None:
            if model_name is not None:
                met_name = self.__models[model_name].get('metrics_name', None)
            elif len(self.losses) == 1:
                met_name = list(self.losses.keys())[0]
        
        if isinstance(met_name, list):
            return [self.__metrics[m] for m in met_name]
        return self.__metrics[met_name] if met_name is not None else None
    
    def get_compiled_metrics(self, new_metrics = None, add_loss = True):
        metrics = get_metrics(
            new_metrics, ** self.default_metrics_config
        ) if new_metrics else self.__metrics
        return MetricList(metrics, losses = self.__losses if add_loss else None)
    
    def __setattr__(self, name, value):
        if isinstance(value, _trackable_objects) and name != 'compiled_metrics':
            self._init_trackable_variable(name, value)
        
        super().__setattr__(name, value)
        
    def __str__(self):
        free_optimizer, free_loss, free_metric = [], [], []
        for _, model in self.__models.items():
            if model['compiled']:
                free_optimizer.append(model['optimizer_name'])
                free_loss.append(model['loss_name'])
                free_metric += model['metrics_name']
        
        free_optimizer = {n : opt for n, opt in self.optimizers.items() if n not in free_optimizer}
        free_loss = {n : loss for n, loss in self.losses.items() if n not in free_loss}
        free_metric = {n : met for n, met in self.__metrics.items() if n not in free_metric}
        
        des = "\n========== {} ==========\n".format(self.nom)
        for name in self.model_names:
            des += self.describe_model(name) + '\n'
            
        if len(self.model_names) > 1:
            des += "Total number of layers : {}\n".format(self.count_layers())
            des += "Total number of parameters : {:.3f} Millions\n".format(self.count_params() / 1000000)
            
        if self.pretrained_name is not None:
            des += "Transfer-learning from : {}\n".format(self.pretrained_name)
        des += "Already trained on {} epochs ({} steps)\n\n".format(self.epochs, self.steps)
            
        if len(free_optimizer) > 0:
            des += "Optimizers :\n"
            for name, opt in free_optimizer.items():
                des += "- {} :\t{}\n".format(name, opt.get_config())
            des += "\n"
        
        if len(free_loss) > 0:
            des += "Losses :\n"
            for name, loss in free_loss.items():
                str_loss = loss.get_config() if hasattr(loss, 'get_config') else loss.__class__.__name__
                des += "- {} :\t{}\n".format(name, str_loss)
            des += "\n"
                
        if len(free_metric) > 0:
            des += "Metrics :\n"
            for name, met in free_metric.items():
                des += "- {} :\t{}\n".format(name, met.get_config())
            des += "\n"
        
        if len(free_optimizer) + len(free_loss) + len(free_metric) > 0: des += '\n'
        return des
    
    def __call__(self, * inputs, training = False, ** kwargs):
        return self.call_fn(* inputs, training = training, ** kwargs)
    
    def summary(self, model = None, **kwargs):
        if model is None: model = self.model_names
        for name in model:
            if name not in self.__models:
                print("[WARNING]\tModel {} does not exist !\n".format(name))
                continue
            print("\n========== Summary of {} ==========\n".format(name))
            self.__models[name]['model'].summary(**kwargs)
            print()
    
    def describe_model(self, name):
        model   = self.__models[name]['model']
        opt     = self.get_optimizer(name)
        loss    = self.get_loss(name)
        metrics = self.get_metric(name)
        if opt is not None: opt = opt.get_config()
        if metrics is not None: metrics = [m.get_config() for m in metrics]
        if hasattr(loss, 'get_config'): loss = loss.get_config()
        
        des = "Sub model {}\n".format(name)
        try:
            des += "- Inputs \t: {}\n".format(model.input_shape)
            des += "- Outputs \t: {}\n".format(model.output_shape)
        except AttributeError:
            des += "- Inputs \t: unknown\n"
            des += "- Outputs \t: unknown\n"
        des += "- Number of layers \t: {}\n".format(len(model.layers))
        des += "- Number of parameters \t: {:.3f} Millions\n".format(model.count_params() / 1000000)
        if opt is None:
            des += "- Model not compiled\n"
        else:
            des += "- Optimizer \t: {}\n".format(opt)
            des += "- Loss \t : {}\n".format(loss)
            des += "- Metrics\t : {}\n".format(metrics)
        return des
    
    def count_layers(self):
        return sum([len(model.layers) for _, model in self.models.items()])
    
    def count_params(self):
        return sum([model.count_params() for name, model in self.models.items()])
    
    def compile(self, model_name = None, **kwargs):
        """
            compile sub model(s) with given loss, optimizer, metrics and their respective configuration 
            See `compile_model` for accepted kwargs names
            
            If you do not want to really `compile` a model but just add losses / optimizers to this class, you can call `self.add_{loss / optimizer / metrics}` instead
            Therefore the `fit` function will not be usable but in some cases it can be useful to have multiple losses / optimizers that are not attached to one specific submodel
        """
        if model_name is None or isinstance(model_name, (list, tuple)):
            if model_name is None: model_name = self.model_names
                
            for name in model_name:
                self.compile(model_name = name, ** kwargs)
        
        elif type(model_name) is dict:
            for name, kw in model_name.items():
                self.compile(model_name = name, ** kw)
        
        elif type(model_name) is str and model_name in self.__models:
            self.compile_model(model_name, ** kwargs)
        else:
            raise ValueError("Modele inconnu !\nReçu : {}\nAcceptés : {}".format(model_name, self.model_names))
            
    def compile_model(self,
                      model_name, 
                      loss          = 'mse',    loss_config         = {},
                      optimizer     = 'adam',   optimizer_config    = {},
                      metrics       = [],       metrics_config      = {},
                      verbose       = True
                     ):
        if self.is_compiled(model_name):
            logging.warning("Model {} is already compiled !".format(model_name))
            return
        
        loss_config.setdefault('reduction', tf.keras.losses.Reduction.NONE)

        loss    = get_loss(loss,            ** loss_config)
        opt     = get_optimizer(optimizer,  ** optimizer_config)
        metrics = get_metrics(metrics,      ** metrics_config)
        
        self.__models[model_name]['model'].compile(optimizer = opt, loss = loss, metrics = metrics)
        
        model_optimizer = self.__models[model_name]['model'].optimizer
        model_loss      = self.__models[model_name]['model'].loss
        if hasattr(self.__models[model_name]['model'], 'compiled_metrics'):
            model_metrics   = self.__models[model_name]['model'].compiled_metrics._metrics
        else:
            model_metrics   = metrics
        
        opt_name    = '{}_optimizer'.format(model_name)
        loss_name   = '{}_loss'.format(model_name)
        met_name    = ['{}_{}'.format(model_name, m.__class__.__name__) for m in model_metrics]
                
        self.__models[model_name]['compiled']       = True
        self.__models[model_name]['optimizer_name'] = opt_name
        self.__models[model_name]['loss_name']      = loss_name
        self.__models[model_name]['metrics_name']   = met_name
                
        self.add_optimizer(model_optimizer, opt_name)
        self.add_loss(model_loss, loss_name)
        self.add_metric(model_metrics, met_name)
        
        str_loss = model_loss.get_config() if isinstance(model_loss, tf.keras.losses.Loss) else model_loss.__name__
        logging.log(
            logging.INFO if verbose else DEV,
            "Submodel {} compiled !\n  Loss : {}\n  Optimizer : {}\n  Metrics : {}".format(
                model_name, 
                str_loss, 
                model_optimizer.get_config(), 
                [m.get_config() for m in model_metrics]
        ))
                
    def _get_default_callbacks(self):
        callbacks = get_callbacks(
            TerminateOnNaN  = {},
            CkptCallback    = {
                'checkpoint'    : self.__ckpt,
                'directory'     : self.ckpt_dir,
                'save_best_only'    : True,
                'monitor'       : 'val_loss'
            }
        )
        return callbacks
        
    def get_dataset_config(self,
                           batch_size       = 16,
                           train_batch_size = None,
                           valid_batch_size = None,

                           shuffle_size     = 1024,
                           is_validation    = True,
                           strategy     = None,
                           ** kwargs
                          ):
        """
            Function that returns dataset configuration that should be used in the `prepare_dataset` call
        """
        config = kwargs.copy()
        
        if train_batch_size: batch_size = train_batch_size
        if is_validation and valid_batch_size: 
            if isinstance(valid_batch_size, float):
                batch_size = int(valid_batch_size * batch_size)
            else:
                batch_size = valid_batch_size
        
        if isinstance(shuffle_size, float): shuffle_size = int(shuffle_size * batch_size)
        
        if strategy is not None:
            batch_size *= strategy.num_replicas_in_sync
        
        config['batch_size']    = batch_size
        config['shuffle_size']  = shuffle_size if not is_validation else 0
        
        if hasattr(self, 'encode_data'): config.setdefault('encode_fn', self.encode_data)
        if hasattr(self, 'filter_data'): config.setdefault('filter_fn', self.filter_data)
        if hasattr(self, 'augment_data') and not is_validation:
            config.setdefault('augment_fn', self.augment_data)
        if hasattr(self, 'preprocess_data'): 
            config.setdefault('map_fn', self.preprocess_data)
        if hasattr(self, 'memory_consuming_fn'):
            config.setdefault('memory_consuming_fn', self.memory_consuming_fn)
        
        return config
        
    def _get_train_config(self,
                          x     = None,
                          y     = None,
                          
                          train_size        = None,
                          valid_size        = None,
                          test_size         = 4,
                          
                          train_times       = 1,
                          valid_times       = 1,
                          
                          pre_shuffle       = False,
                          random_state      = 10,
                          labels            = None,
                          validation_data   = None,
                          
                          test_batch_size   = 1,
                          pred_step         = -1,
                          
                          epochs            = 5, 
                          relative_epoch    = True,
                          
                          verbose           = 1, 
                          callbacks         = [], 
                          ** kwargs
                          ):
        """
            Function that returns dataset configuration for training / evaluation data
            
            Arguments :
                - x / y     : training dataset that will be passed to `prepare_dataset`
                - train_size / valid_size / test_size   : sizes of datasets
                - random_state  : used for shuffling the same way accross trainings
                It allows to pass a complete dataset and the splits will **always** be the same accross trainings which is really interesting to not have problems with shuffling
                - labels        : use in the `sklearn.train_test_split`, rarely used
                - validation_data   : dataset used for validation
                    If not provided, will split training dataset by train_size and valid_size
                    Note that the `test` dataset is a subset of the validation dataset and is used in the `PredictorCallback`
                
                - test_batch_size / pred_step   : kwargs specific for `PredictorCallback`
                
                - epochs    : the number of epochs to train
                - relative_epochs   : whether `epochs` must be seen as absolute epochs (ie the final epoch to reach) or the number of epochs to train on
                
                - verbose   : verbosity level
                - callbacks : training callbacks (added to `self._get_default_callbacks`)
                
                - kwargs    : additional configuration passed to `get_dataset_config` (such as `cache`, `prefetch`, ...)
        """
        train_config = self.get_dataset_config(is_validation = False, ** kwargs)
        valid_config = self.get_dataset_config(is_validation = True, ** kwargs)
        valid_config.setdefault('cache', False)
        
        test_kwargs = kwargs.copy()
        test_kwargs.update({
            'batch_size' : test_batch_size, 'train_batch_size' : None, 'valid_batch_size' : None,
            'cache' : False, 'is_validation' : True
        })
        test_config  = self.get_dataset_config(** test_kwargs)
        
        dataset = x if y is None else (x, y)
        if isinstance(dataset, dict) and 'train' in dataset:
            validation_data = dataset.get('valid', dataset.get('test', validation_data))
            dataset         = dataset['train']
        
        if validation_data is None:
            train_dataset, valid_dataset = train_test_split(
                dataset, 
                train_size     = train_size,
                valid_size     = valid_size,
                random_state   = random_state,
                shuffle        = pre_shuffle,
                labels         = labels
            )
        else:
            train_dataset, _ = train_test_split(
                dataset, train_size = train_size,
                random_state = random_state, shuffle = pre_shuffle
            ) if train_size is not None else dataset, None
            valid_dataset, _ = train_test_split(
                validation_data, train_size = train_size,
                random_state = random_state, shuffle = pre_shuffle
            ) if valid_size is not None else validation_data, None
        
        if test_size > 0:
            test_dataset    = prepare_dataset(valid_dataset,  ** test_config)
            test_dataset    = test_dataset.take(test_size)
        
        train_dataset = prepare_dataset(train_dataset, ** train_config)
        valid_dataset = prepare_dataset(valid_dataset, ** valid_config)
        
        if train_times > 1: train_dataset = train_dataset.repeat(train_times)
        if valid_times > 1: valid_dataset = valid_dataset.repeat(valid_times)
        
        if relative_epoch: epochs += self.epochs
        
        callbacks = callbacks + self._get_default_callbacks()
        
        if hasattr(self, 'predict_with_target') and test_size > 0 and pred_step != 0:
            predictor_callback  = get_callbacks(PredictorCallback = {
                'method'    : self.predict_with_target,
                'generator' : test_dataset,
                'directory' : self.train_test_dir,
                'initial_step'  : self.steps,
                'pred_every'    : pred_step
            })
            callbacks += predictor_callback
        
        return {
            'x'                 : train_dataset,
            'epochs'            : epochs,
            'verbose'           : verbose,
            'callbacks'         : callbacks,
            'validation_data'   : valid_dataset,
            'shuffle'           : False,
            'initial_epoch'     : self.epochs,
            'global_batch_size' : train_config['batch_size']
        }
        
    def fit(self, * args, name = None, custom_config = True, ** kwargs):
        """
            Call `model.fit` for the specified model (or single model if it has only 1 submodel) 
            
            if custom_config is True, pass kwargs and args to self._get_train_config and keep track of training configuration
            Otherwise, just call `fit` by passing args and kwargs (bad idea because it will not include all processing function in the dataset preparation) so only use it if you already processed the datasets or do not want to process it
        """
        if name is None and len(self.model_names) > 1:
            if 'full_model' in self.__models:
                name = 'full_model'
            else:
                raise ValueError("Pour entrainer un modèle quand il existe plusieurs sous-modèles, il faut l'identifier par son nom ! (ou alors instancier la variable 'full_model' combinant les différents sous-modèles)")

        train_model = self.get_model(name)
        
        if custom_config:
            base_hparams    = HParamsTraining().extract(kwargs, pop = False)
            train_hparams   = self.training_hparams.extract(kwargs, pop = True)
            self.init_train_config(** train_hparams)

            config = self._get_train_config(*args, **kwargs)
            self.global_batch_size = config.pop('global_batch_size', -1)
            
            base_hparams.extract(config, pop = False)
            train_hparams.update(base_hparams)
            
            self.history.set_params(train_hparams)
            
            logging.info("Training config :\n{}\n".format(train_hparams))
        else:
            config = kwargs

        config.setdefault('callbacks', [])
        config['callbacks'].append(self.history)
        
        start = time.time()
        try:
            if custom_config:                
                _ = train_model.fit(** config)
            else:
                _ = train_model.fit(* args, ** config)
        except KeyboardInterrupt as e:
            logging.warning("Training interrupted ! Saving model...")
        
        logging.info("Training finished after {} !".format(time_to_string(time.time() - start)))
        
        self.save()
        return self.history
    
    def train(self, 
              * args, 
              verbose       = 1,
              eval_epoch    = 1,
              verbose_step  = 100,
              tqdm          = tqdm_progress_bar,
              strategy      = None,
              run_eagerly   = False,
              # custom functions for training and evaluation
              train_step    = None,
              eval_step     = None,
              ** kwargs
             ):
        """
            Train the model with given datasets and configuration bby keeping track of all training hyperparameters
            
            Arguments :
                - args  : arguments passed to `self._get_train_config`, often the datasets iteself
                - verbose   : verbosity level of the `ProgressBar` (1 to add it and 0 for not)
                    If tf.__version__ == 2.1.0, verbose can also be 2 to print every `verbose_step` because the `ProgressBar` callback is not handled correctly in this version
                - verbose_step / tqdm  : only relevant for tf.__version__ == 2.1.0
                
                - strategy  : the distribute strategy to use for training (therically working but not tested in practice)
                
                - train_step / eval_step    : callable that takes a batch as argument and returns metrics
                    By default, use self.train_step and self.test_step
                
                - kwargs    : all configuration to pass to self._get_train_config()
                    It can also be whathever training hparams you specified in the `training_hparams` property
                    All these training configuration will be tracked by the `History` callback
        """
        if strategy is None: strategy = self.distribute_strategy
        if verbose_step is not None or verbose < 2: tqdm = lambda x: x
        
        ########################################
        #     Initialisation des variables     #
        ########################################
        
        base_hparams    = HParamsTraining().extract(kwargs, pop = False)
        train_hparams   = self.training_hparams.extract(kwargs, pop = True)
        self.init_train_config(** train_hparams)
        
        config = self._get_train_config(* args, strategy = strategy, ** kwargs)
        self.global_batch_size = config.pop('global_batch_size', -1)
        
        base_hparams.extract(config, pop = False)
        train_hparams.update(base_hparams)
        
        logging.info("Training config :\n{}\n".format(train_hparams))
        
        ##############################
        #     Dataset variables      #
        ##############################
        
        train_dataset = config['x']
        valid_dataset = config['validation_data']
        
        assert isinstance(train_dataset, tf.data.Dataset)
        assert isinstance(valid_dataset, tf.data.Dataset) or valid_epoch <= 0
                
        if strategy is not None:
            logging.info("Running on {} GPU".format(strategy.num_replicas_in_sync))
            train_dataset   = strategy.experimental_distribute_dataset(train_dataset)
            valid_dataset   = strategy.experimental_distribute_dataset(valid_dataset)
        
        init_epoch  = config['initial_epoch']
        last_epoch  = config['epochs']
        
        ##############################
        #  Metrics + callbacks init  #
        ##############################
        
        # Get compiled metrics
        self.compiled_metrics = self.get_compiled_metrics()
        
        # Prepare callbacks
        callbacks   = config['callbacks']
        callbacks.append(self.__history)
        
        if tf.__version__ == '2.1.0':
            if verbose: verbose = 2
            callbacks   = CallbackList(callbacks)
        else:
            if verbose: verbose = 1
            callbacks   = CallbackList(callbacks, add_progbar = verbose > 0, model = self)
        
        callbacks.set_params(train_hparams)
        
        callbacks.on_train_begin()
        
        ####################
        #     Training     #
        ####################
        
        train_function  = self.make_train_function(strategy, train_step, run_eagerly = run_eagerly)
        eval_function   = self.make_eval_function(strategy, eval_step, run_eagerly = run_eagerly)
        
        start_training_time = time.time()
        last_print_time, last_print_step = start_training_time, int(self.current_step.numpy())
        
        try:
            for epoch in range(init_epoch, last_epoch):
                logging.info("\nEpoch {} / {}".format(epoch + 1, last_epoch))
                callbacks.on_epoch_begin(epoch)
                
                start_epoch_time = time.time()
                
                self.compiled_metrics.reset_states()
                
                for i, batch in enumerate(tqdm(train_dataset)):
                    callbacks.on_train_batch_begin(i)
                    
                    metrics = train_function(batch)
                    metrics = {
                        n : m for n, m in zip(self.compiled_metrics.metric_names, self.compiled_metrics.result())
                    }
                    
                    callbacks.on_train_batch_end(i, logs = metrics)

                    self.update_train_config(
                        step    = self.current_step + 1,
                        epoch   = self.current_epoch
                    )
                    if self.stop_training: break
                    
                    if verbose == 2 and (i+1) % verbose_step == 0:
                        logging.info("Epoch {} step {} (avg time : {}) :\n  {}".format(
                            epoch,
                            i+1, 
                            time_to_string((time.time() - last_print_time) / (int(self.current_step) - last_print_step)), 
                            self.__history.str_training()))
                        last_print_time, last_print_step = time.time(), int(self.current_step)

                if eval_epoch > 0 and epoch % eval_epoch == 0:
                    self._test_loop(
                        valid_dataset, eval_function, callbacks, tqdm, prefix = 'val_'
                    )

                epoch_time = time.time() - start_epoch_time
                
                self.update_train_config(
                    step    = self.current_step,
                    epoch   = self.current_epoch + 1
                )
                
                callbacks.on_epoch_end(epoch, logs = self.__history.training_logs)
                
                if verbose == 2:
                    logging.info("\nEpoch {} / {} - Time : {} - Remaining time : {}\n{}".format(
                        epoch, last_epoch,
                        time_to_string(epoch_time),
                        time_to_string(epoch_time * (last_epoch - epoch)),
                        self.__history.to_string(mode = 'all')
                    ))

        except KeyboardInterrupt:
            logging.warning("Training interrupted ! Saving model...")
        
        callbacks.on_train_end()
        
        total_training_time = time.time() - start_training_time
        logging.info("Training finished after {} !".format(time_to_string(total_training_time)))
        
        self.save()
        
        return self.__history
    
    def update_metrics(self, y_true, y_pred):
        self.compiled_metrics.update_state(y_true, y_pred)

        return self.compiled_metrics.result()
        
    def _test_loop(self, dataset, eval_function, callbacks, tqdm, prefix = None):
        """ Utility function used in `train` for evaluation phase and in `test` """
        if prefix is None:
            prefix = 'val_' if self.__history.training else 'test_'

        self.compiled_metrics.reset_states()

        callbacks.on_test_begin()
        for i, batch in enumerate(tqdm(dataset)):
            callbacks.on_test_batch_begin(i)
            
            val_metrics = eval_function(batch)
            val_metrics = {
                n : m for n, m in zip(self.compiled_metrics.metric_names, self.compiled_metrics.result())
            }
            
            val_metrics = {
                prefix + name : val for name, val in val_metrics.items()
            }
            
            callbacks.on_test_batch_end(i, logs = val_metrics)
        
        callbacks.on_test_end(logs = self.__history.training_logs)
    
    def test(self, 
             dataset,
             test_name      = 'test',
             metrics        = None,
             add_loss       = True,
             
             verbose        = 1,
             verbose_step   = 100,
             tqdm           = tqdm_progress_bar,
             strategy       = None,
             run_eagerly    = False,
             # custom functions for training and evaluation
             eval_step      = None,
             ** kwargs
            ):
        """
            Same as `train`, it is an equivalent to `tf.keras.Model.evaluate` but using `self.test_step` as testing function
        """
        if strategy is None: strategy = self.distribute_strategy
        if verbose_step is not None or verbose < 2: tqdm = lambda x: x
        
        prefix = test_name if test_name.endswith('_') else test_name + '_'
        ########################################
        #     Initialisation des variables     #
        ########################################
        
        base_hparams    = HParamsTesting().extract(kwargs, pop = False)
        test_hparams   = self.training_hparams.extract(kwargs, pop = True)
        self.init_train_config(** test_hparams)
        
        test_hparams.update(base_hparams)
        
        logging.info("Testing config :\n{}\n".format(test_hparams))
                
        ##################################
        #     Dataset initialization     #
        ##################################
        
        config  = self.get_dataset_config(is_validation = True, ** kwargs)
        dataset = prepare_dataset(dataset, ** config)
        
        assert isinstance(dataset, tf.data.Dataset)
        
        if strategy is not None:
            dataset   = strategy.experimental_distribute_dataset(dataset)
                        
        # Prepare metrics and logs
        if metrics is not None and not isinstance(metrics, list): metrics = [metrics]
        self.compiled_metrics    = self.get_compiled_metrics(metrics, add_loss = add_loss)
        
        # Prepare callbacks
        callbacks   = [self.__history]
        
        if tf.__version__ == '2.1.0':
            if verbose: verbose = 2
            callbacks   = CallbackList(callbacks)
        else:
            if verbose: verbose = 1
            callbacks   = CallbackList(callbacks, add_progbar = verbose > 0, model = self)
        
        test_hparams.update({
            'verbose'   : verbose,
            'epochs'    : self.epochs,
            'test_prefix'   : prefix
        })
        callbacks.set_params(test_hparams)
        
        ####################
        #     Testing      #
        ####################
        
        eval_function   = self.make_eval_function(strategy, eval_step, run_eagerly = run_eagerly)
        
        start_test_time = time.time()
        
        try:
            self._test_loop(
                dataset, eval_function, callbacks, tqdm, prefix = prefix
            )
        except KeyboardInterrupt:
            logging.warning("Testing interrupted !")
        
        self.save_history()
        
        total_test_time = time.time() - start_test_time
        logging.info("Testing finished after {} !".format(time_to_string(total_test_time)))
        
        return self.__history
    
    def make_train_function(self, strategy, train_step = None, run_eagerly = False):
        if train_step is not None: pass
        elif hasattr(self, 'train_step'): train_step = self.train_step
        else:
            variables = self.list_trainable_variables
            loss_fn     = self.get_loss()
            optimizer   = self.get_optimizer()
            
            nb_loss     = len(get_metric_names(self.losses))
            
            def train_step(batch):
                inputs, target = batch
                
                with tf.GradientTape() as tape:
                    y_pred = self(inputs, training = True)
                    
                    replica_loss = compute_distributed_loss(
                        loss_fn, target, y_pred,
                        global_batch_size = self.global_batch_size,
                        nb_loss = nb_loss
                    )

                grads = tape.gradient(replica_loss, variables)
                optimizer.apply_gradients(zip(grads, variables))

                self.update_metrics(target, y_pred)
                return replica_loss
                        
        def run_step(batch):
            if strategy is None:
                outputs = train_step(batch)
            else:
                outputs = strategy.run(train_step, args = (batch,))

                outputs = _reduce_per_replica(outputs, strategy, reduction = 'sum')
            
            return outputs
        
        return _compile_fn(
            run_step, 
            run_eagerly = self.run_eagerly if not run_eagerly else True,
            signature   = (self.input_signature, self.output_signature),
            include_signature   = self.include_signature
        )
        
    def make_eval_function(self, strategy, eval_step = None, run_eagerly = False):
        if eval_step is not None: pass
        elif hasattr(self, 'eval_step'): eval_step = self.eval_step
        else:
            model = self.get_model()

            def eval_step(batch):
                inputs, target = batch
                y_pred = self(inputs, training = False)

                return self.update_metrics(target, y_pred)
        
        def run_step(batch):
            if strategy is None:
                outputs = eval_step(batch)
            else:
                outputs = strategy.run(eval_step, args = (batch,))

                #outputs = _reduce_per_replica(outputs, strategy, reduction = 'first')
            
            return outputs
        
        return _compile_fn(
            run_step, 
            run_eagerly = self.run_eagerly if not run_eagerly else True,
            signature   = (self.input_signature, self.output_signature),
            include_signature   = self.include_signature
        )
    
    def predict(self, inputs, name = None, **kwargs):
        model = self.get_model(name)
        return model.predict(inputs, **kwargs)
        
    def evaluate(self, datas, name = None, **kwargs):
        model = self.get_model(name)
        dataset_config = self.get_dataset_config(** kwargs)
        dataset = prepare_dataset(datas, ** dataset_config)
        return model.evaluate(dataset)
    
    def _add_history(self, history, by_step = False):
        self.__history.append(history, by_step = by_step)
        
    def plot_history(self, ** kwargs):
        self.__history.plot(** kwargs)
        
    def copy(self, nom = None):
        """ Create a copy of this model """
        return self.__class__.clone(pretrained = self.nom, nom = nom)
    
    def restore_models(self, directory = None, checkpoint = None, compile = True):
        if directory is not None:
            logging.info("Model restoration from {}...".format(directory))
            filename = os.path.join(directory, "config_models.json")
        else:
            logging.info("Model restoration...")
            filename = self.config_models_file
        
        variables_to_restore = load_json(filename)
        
        for model_name, infos in variables_to_restore['models'].items():
            compile_infos = None
            if infos['compiled'] and compile:
                compile_infos = {
                    ** variables_to_restore['optimizers'].pop(infos['optimizer_name']),
                    ** variables_to_restore['losses'].pop(infos['loss_name']),
                    'metrics' : [variables_to_restore['metrics'].pop(met_name) for met_name in infos['metrics_name']]
                }
            self.load_model(model_name, infos['save_path'], compile_infos)
        
        if compile:
            for opt_name, infos in variables_to_restore['optimizers'].items():
                name, kw = infos['optimizer'], infos['optimizer_config']
                setattr(self, opt_name, get_optimizer(name, **kw))

            for loss_name, infos in variables_to_restore['losses'].items():
                name, kw = infos['loss'], infos['loss_config']
                setattr(self, loss_name, get_loss(name, **kw))

            for met_name, infos in variables_to_restore['metrics'].items():
                name, kw = infos['metric'], infos['metric_config']
                setattr(self, met_name, get_metrics(name, **kw))
        
        self.load_checkpoint(directory = directory, checkpoint = checkpoint)
    
    def load_checkpoint(self, directory = None, checkpoint = None):
        if directory is None and checkpoint is None:
            checkpoint = self.latest_checkpoint
        elif directory is not None:
            if is_model_name(directory):
                directory = os.path.join(_pretrained_models_folder, directory, 'saving')
            
            if checkpoint is None:
                checkpoint = tf.train.latest_checkpoint(directory)
            else:
                checkpoint = os.path.join(directory, checkpoint)
            logging.info('Loading checkpoint {}'.format(checkpoint))

        return self.checkpoint.restore(checkpoint)

    def load_training_checkpoint(self):
        self.load_checkpoint(checkpoint = self.latest_train_checkpoint)
        
    def get_config(self, with_trackable_variables = False):
        config = {
            "nom" : self.nom,
            'pretrained_name'   : self.pretrained_name,
            ** self.backend_kwargs
        }
        
        if with_trackable_variables:
            config['trackable_variables'] = self.get_trackable_variables_config()
        
        return config
    
    def get_trackable_variables_config(self):
        config = {
            'models'   : {},
            'losses'    : {},
            'metrics'   : {},
            'optimizers'    : {}
        }
        for model_name, model_infos in self.model_infos.items():
            config['models'][model_name] = {k : v for k, v in model_infos.items() if k != 'model'}
            
        for opt_name, opt in self.optimizers.items():
            config['optimizers'][opt_name] = {
                'optimizer'    : opt.__class__.__name__,
                'optimizer_config'  : opt.get_config()
            }
            
        for met_name, met in self.__metrics.items():
            config['metrics'][met_name] = {
                'metric'    : met.__class__.__name__,
                'metric_config'  : met.get_config()
            }
            
        for loss_name, loss in self.losses.items():
            name, kw = loss, {}
            if isinstance(loss, tf.keras.losses.Loss):
                name = loss.__class__
                kw = loss.get_config() 
            name = name.__name__
            
            config['losses'][loss_name] = {
                'loss'    : name,
                'loss_config'  : kw
            }
            
        return config
    
    def save(self, save_models = True, save_ckpt = True, 
             save_history = True, save_config = True):
        if save_models: self.save_models()
        if save_ckpt: self.ckpt_manager.save()
        if save_history: self.save_history()
        if save_config: self.save_config()
    
    def save_history(self):
        self.__history.save(self.history_file)
        
    def save_models(self, directory = None, **kwargs):
        for model_name in self.model_names:
            filename = None if directory is None else os.path.join(directory, model_name)
            self.save_model(
                model_name      = model_name, 
                filename        = filename,
                ** kwargs
            )
        
        
    def save_model(self, model_name, filename = None, save_weights = False, ** kwargs):
        if filename is None and type(model_name) is str: 
            if model_name not in self.models:
                raise ValueError("Le modèle est inconnu !\nReçu : {}\nExistants : {}".format(model_name, self.model_names))
            
            filename = self.model_infos[model_name]['save_path']
        
        config_filename = filename if '.json' in filename else filename + '.json'
        with open(config_filename, 'w', encoding = 'utf-8') as model_config_file:
            model_config = self.__models[model_name]['model'].to_json(indent = 4)
            model_config_file.write(model_config)
            
        if save_weights:
            self.__models[model_name]['model'].save_weights(filename, **kwargs)
            
        logging.info("Submodel {} saved in {} !".format(model_name, config_filename))
                
    def save_config(self, with_trackable_variables = True, directory = None):
        config = {
            'class_name'    : self.__class__.__name__,
            'config'        : self.get_config(with_trackable_variables = False)
        }
        
        config_file = self.config_file if directory is None else os.path.join(directory, 'config.json')
        dump_json(config_file, config, indent = 4)
        
        if with_trackable_variables:
            trackable_variables_config = self.get_trackable_variables_config()
            config_models_file = self.config_models_file if directory is None else os.path.join(directory, 'config_models_file.json')
            
            dump_json(config_models_file, trackable_variables_config, indent = 4)
        
    def load_model(self, name, filename, compile_infos = None, verbose = True):
        restored_model = None
        if '.json' in filename:
            with open(filename, 'r', encoding = 'utf-8') as fichier_config:
                config = fichier_config.read()
            restored_model = model_from_json(config, custom_objects = self.custom_objects)
        else:
            restored_model = load_model(
                filename, custom_objects = self.custom_objects, compile = False
            )
        
        setattr(self, name, restored_model)
        if compile_infos is not None:
            self.compile(model_name = name, ** compile_infos, verbose = False)

        logging.info("Successfully restored {} from {} !".format(name, filename))
    
    def destroy(self, ask = True):
        """ Destroy the model and all its folders """
        y = True
        if ask:
            y = input("Are you sure you want to destroy agent {} ? (y to validate)".format(self.nom))
            y = y == 'y'
        
        if y:
            shutil.rmtree(self.folder)
            del self
    
    @classmethod
    def clone(cls, pretrained, nom = None, compile = False):
        pretrained_dir = get_model_dir(pretrained)
        if not os.path.exists(pretrained_dir):
            raise ValueError("Pretrained model {} does not exist !".format(pretrained))
        
        if nom is None:
            n = len([
                n for n in os.listdir(_pretrained_models_folder) if n.startswith(pretrained)
            ])
            nom = '{}_copy_{}'.format(pretrained, n)
        elif nom in os.listdir(_pretrained_models_folder):
            raise ValueError("Model {} already exist, cannot create a copy !".format(nom))

        config = load_json(os.path.join(pretrained_dir, 'config.json'))
        if config['class_name'] != cls.__name__:
            raise ValueError("Model {} already exists but is not the expected class !\n  Expected : {}\n  Got : {}".format(nom, config['class_name'], cls.__name__))
        
        config = config['config']
        config.update({'nom' : nom, 'pretrained_name' : pretrained})
        config.setdefault('max_to_keep', 1)
        
        instance = cls(
            restore = {'directory' : pretrained, 'compile' : compile}, ** config
        )
        instance.save()
        return instance
    
    @classmethod
    def from_pretrained(cls, nom, pretrained_name, ** kwargs):
        from models import get_pretrained
        
        with tf.device('cpu') as d:
            pretrained = get_pretrained(pretrained_name)
        
        config = pretrained.get_config()
        config.update({'nom' : nom, 'pretrained_name' : pretrained_name, ** kwargs})
        
        instance = cls(max_to_keep = 1, ** config)
        
        partial_transfer_learning(instance.get_model(), pretrained.get_model())
        
        instance.save()
        
        return instance

    @classmethod
    def restore(cls, nom):
        folder = get_model_dir(nom)
        if not os.path.exists(folder): return None
        
        config = load_json(os.path.join(os.path.join(folder, 'config.json')))
        
        if config['class_name'] != cls.__name__:
            raise ValueError("Model {} already exists but is not the expected class !\n  Expected : {}\n  Got : {}".format(nom, config['class_name'], cls.__name__))
        
        return cls(** config['config'])
    
    @staticmethod
    def rename(nom, new_name):
        def _rename_in_file(filename):
            if os.path.isdir(filename):
                for f in os.listdir(filename): _rename_in_file(os.path.join(filename, f))
            elif filename.endswith('.json'):
                with open(filename, 'r', encoding = 'utf-8') as file:
                    text = file.read().replace(nom, new_name)
                with open(filename, 'w', encoding = 'utf-8') as file:
                    file.write(text)
            
        folder = get_model_dir(nom)
        if not os.path.exists(folder):
            raise ValueError("Pretrained model {} does not exist !".format(nom))
        
        if is_model_name(new_name):
            raise ValueError("Model {} already exist, cannot rename model !".format(new_name))

        _rename_in_file(folder)
        os.rename(folder, os.path.join(get_model_dir(new_name)))
        
def _can_restore(restore, config_file):
    if restore is True: return os.path.exists(config_file)
    elif isinstance(restore, str):
        return is_model_name(restore) or os.path.exists(os.path.join(restore, 'config.json'))
    elif isinstance(restore, dict):
        return is_model_name(restore['directory']) or os.path.exists(os.path.join(restore['directory'], 'config.json'))
    
def _compile_fn(fn, run_eagerly = False, signature = None, 
                include_signature = True, ** kwargs):
    if not run_eagerly:
        config = {
            'experimental_relax_shapes' : True,
            ** kwargs
        }
        if include_signature and signature is not None:
            config['input_signature'] = [signature]
        
        fn = def_function.function(fn, ** config)
    
    return fn
    
def _reduce_per_replica(values, strategy, reduction = 'mean'):
    def _reduce(v):
        if reduction == 'first':
            return strategy.unwrap(v)[0]
        elif reduction == 'mean':
            return strategy.reduce(tf.distribute.ReduceOp.MEAN, v, axis = None)
        elif reduction == 'sum':
            return strategy.reduce(tf.distribute.ReduceOp.SUM, v, axis = None)
        else:
            return v

    return nest.map_structure(_reduce, values)

def compute_distributed_loss(loss_fn, y_true, y_pred, global_batch_size = None, nb_loss = None):
    loss = loss_fn(y_true, y_pred)

    if nb_loss is not None and nb_loss > 1: loss = loss[0]
    return tf.nn.compute_average_loss(loss, global_batch_size = global_batch_size)
