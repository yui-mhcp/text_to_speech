# Copyright (C) 2025-now yui-mhcp project author. All rights reserved.
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
import time
import inspect
import logging

from functools import cached_property, partialmethod

from loggers import Timer, timer
from utils import JSONSaver, Stream, dump_json, load_json, time_to_string, copy_methods
from utils.keras import TensorSpec, ops, build_runtime, graph_compile
from utils.datasets import prepare_dataset, summarize_dataset
from custom_train_objects import CheckpointManager, History

from ..utils import get_saving_dir, get_model_dir, is_model_name, describe_model, loss_to_str, optimizer_to_str, metrics_to_str
from ..weights_converter import name_based_partial_transfer_learning

logger = logging.getLogger(__name__)

class ModelInstances(type):
    _instances = {}
    
    @timer(name = 'model loading', debug = True)
    def __call__(cls, * args, reload = False, name = None, ** kwargs):
        if not name: name = kwargs.pop('nom', cls.__name__)
        
        if reload or name not in cls._instances:
            config_file = get_model_dir(name, 'config.json')
            if os.path.exists(config_file):
                config = load_json(config_file)
                if config['class_name'] != cls.__name__:
                    raise ValueError("Model `{}` already exists with a non-matching class !\n  Expected : {}\n  Got : {}".format(name, config['class_name'], cls.__name__))

                args = ()
                kwargs = {** config['config'], ** kwargs}
            kwargs['name'] = name
            
            try:
                cls._instances[name] = super().__call__(* args, ** kwargs)
            except Exception as e:
                logger.critical('An error occured while initializing {} : {}'.format(name, e))
                raise e

        return cls._instances[name]

@copy_methods(
    'model',
    'inputs', 'outputs', 'input_shape', 'output_shape', 'summary',
    'variables', 'trainable_variables', 'non_trainable_variables',
    'save_weights', 'load_weights', 'set_weights', 'get_weights',
    'train_on_batch', 'test_on_batch', 'predict_on_batch',
)
@copy_methods(
    'checkpoint_manager',
    'checkpoint_format', 'latest_checkpoint', 'loaded_checkpoint',
    load_checkpoint = 'load', save_checkpoint = 'save', delete_checkpoint = 'delete',
    type = CheckpointManager
)
@copy_methods(
    'history',
    'epochs', 'steps', 'training_logs', 'training_config', 'training_infos', 'training_time',
    plot_history = 'plot',
    type = History
)
class BaseModel(metaclass = ModelInstances):
    """
        Abstract class wrapping a `keras.Model`, and defining the processing functions / parameters
        
        Each `BaseModel` is identified by its `name`, which **must** be unique
        
        Arguments :
            - name  : the model's name
            - root  : the root directory where to save configuration / weights files
            - save  : whether to save the model or not
            - max_to_keep   : maximum number of checkpoint files to keep
            
            - pretrained_name   : the original model used for transfer-learning / initialization
            - force_rebuild     : whether to force calling `build()` instead of `restore_models`
        
        Sub-classes have to override the `build` method in order to define the configuration to instanciate the `keras.Model` instances, then call `super().build(model = config)` in order to properly initialize the model
        The method can also directly sets the attribute : `self.model = model`
        
        The model defines multiple processing functions, executed in the following order :
            1) `augment_raw_data(data)` : data augmentation on raw unprepared data
            2) `prepare_data(data)`     : initial data preparation (e.g., text tokenization, image loading)
            3) `filter_data(inp, out)`  : return whether to keep the data or not
            4) `process_data(inp, out)` : data processing on single data
        // `cache` is applied here
        // `shuffle` is applied here
            5) `augment_data(inp, out)` : data augmentation on single data
        // `batch` is applied here
            6) `process_batch_data(inp, out)` : data processing on batched data
        // `prefetch` is applied here
        
        The `prepare_data` is expected to take a single argument (typically a `dict`), while all subsequent functions are expected to take 2 arguments : `inputs` and `output`
        The `prepare_data` aims to extract inputs and outputs from the raw initial data
        
        At prediction time, the pipeline is modified to only call the `{prepare / process / process_batch}_input` functions, as the output is unexpected for inference
        
        The `cache` is applied just before data augmentation / shuffling to not cache random operations (which would be applied only once if cached, breaking data augmentation principle)
        
        Each of the above methods internally calls their equivalent for `input` and `output`,
        except the `augment_raw_data`, which takes raw data as argument, and outputs raw data
        The `_output` function takes the `inputs` as kwargs to enable input-related processing (e.g., in AutoEncoder, `output == inputs`)
        ```python
        def prepare_data(self, data):
            inputs = self.prepare_input(data)
            return inputs, self.prepare_output(data, inputs = inputs)
        
        def augment_data(self, output):
            inputs = self.prepare_input(inputs)
            return inputs, self.augment_output(output, inputs = inputs)
        ...
        ```
    """
    _directories    = {
        'directory' : '{root}/{self.name}',
        'save_dir'  : '{root}/{self.name}/saving',
        'pred_dir'  : '{root}/{self.name}/predictions',
        'train_dir' : '{root}/{self.name}/training-logs'
    }
    _files  = {
        'config_file'   : '{self.directory}/config.json',
        'history_file'  : '{self.save_dir}/history.json',
        'config_models_file' : '{self.save_dir}/config_models.json'
    }
    
    def __init__(self,
                 *,
                 
                 name   = None,
                 pretrained_name    = None,
                 
                 save   = True,
                 max_to_keep    = 3,
                 force_rebuild  = False,
                 
                 run_eagerly    = None,
                 support_xla    = None,
                 graph_compile_config   = None,
                 
                 runtime   = 'keras',
                 
                 ** kwargs
                ):
        """ Constructor that initialize the model's configuration, architecture, folders, ... """
        assert name is not None

        # Allows to pass all kwargs to super-classes without forwarding everything to `build`
        kwargs  = {k : v for k, v in kwargs.items() if not hasattr(self, k)}
        
        self.name   = name
        self._save  = save
        self.runtime    = runtime
        self.build_kwargs   = kwargs
        self.pretrained_name    = pretrained_name
        
        self.max_to_keep    = max_to_keep
        self._run_eagerly   = run_eagerly
        self._support_xla   = support_xla
        self._graph_compile_config  = graph_compile_config
        
        self._init_dir_properties()
        self._init_processing_functions()

        self.__history  = None
        self.__checkpoint_manager   = None
        
        if not force_rebuild and os.path.exists(self.config_file):
            with Timer('{} restoration'.format(runtime), debug = True):
                if runtime == 'keras':
                    self._restore_model()
                else:
                    self.model = build_runtime(runtime, ** kwargs)
        else:
            if runtime == 'keras':
                self.build(** kwargs)
            else:
                self.model = build_runtime(runtime, ** kwargs)
        
            if save and not os.path.exists(self.config_file):
                self._init_directories()
                self.save()

        self._init_train_config()
        
        if hasattr(self, 'get_output'):
            if hasattr(self, 'prepare_input') and not hasattr(self, 'prepare_output'):
                self.prepare_output = self.get_output
            elif hasattr(self, 'process_input') and not hasattr(self, 'process_output'):
                self.process_output = self.get_output
        else:
            self.get_output = self._get_output
            
        self._compiled_call = None
        self._compiled_infer    = None
        
        logger.info("{} `{}` initialized successfully !".format(self.__class__.__name__, self.name))
    
    @timer(debug = True)
    def _init_dir_properties(self):
        def format_path(path_format):
            return property(
                lambda self: path_format.format(self = self, root = get_saving_dir())
            )
        
        all_paths = {** self._directories, ** self._files}
        for property_name, path_format in all_paths.items():
            setattr(self.__class__, property_name, format_path(path_format))

    @timer(debug = True)
    def _init_processing_functions(self):
        for prefix in ('prepare', 'augment', 'process', 'process_batch'):
            if not hasattr(self, f'{prefix}_data') and (
                hasattr(self, f'{prefix}_input') or hasattr(self, f'{prefix}_output')
            ):
                setattr(self, f'{prefix}_data', _build_generic_processing(
                    getattr(self, f'{prefix}_input', lambda inp, ** kwargs: inp),
                    getattr(self, f'{prefix}_output', lambda out, ** kwargs: out),
                    name = f'{prefix}_data'
                ))
        if hasattr(self, 'filter_input') or hasattr(self, 'filter_output'):
            self.filter_data = self._filter_data
            
    @timer(debug = True)
    def _init_directories(self):
        """ Initialize directory structure based on `self._directories` """
        for property_name in self._directories.keys():
            os.makedirs(getattr(self, property_name), exist_ok = True)
    
    @timer(debug = True)
    def _init_train_config(self, ** kwargs):
        """ Initializes custom training parameters """
        mapper = self.training_hparams_mapper
        
        for k, v in self.training_hparams.items():
            val = kwargs.get(k, v)
            if k in mapper:
                mapper[k](val)
            elif not hasattr(self, k) or val is not None:
                setattr(self, k, val)

    @timer(debug = True)
    def build(self, model = None, ** kwargs):
        """ Initializes the effective `keras.Model` `self._model` attribute """
        import keras
        
        from architectures import get_architecture

        if model is None: model = kwargs
        
        if isinstance(model, keras.Model):
            self.model = model
        elif isinstance(model, dict):
            self.model = get_architecture(** model)
        elif isinstance(model, str):
            self.model = get_architecture(model)
        
    
    model   = property(lambda self: self._model)
    loss    = property(lambda self: getattr(self, '_loss', self.model.loss))
    optimizer   = property(lambda self: self.model.optimizer)
    compiled    = property(
        lambda self: getattr(self.model, 'compiled', False) and self.loss is not None
    )
    
    input_signature     = property(lambda self: get_signature(self.input_shape))
    output_signature    = property(lambda self: get_signature(self.output_shape))

    @model.setter
    def model(self, value):
        self._model = value
        if self.runtime == 'keras':
            self.checkpoint_manager.model = value
    
    @loss.setter
    def loss(self, value):
        self._loss = value
    
    @property
    def history(self):
        if self.__history is None:
            if os.path.exists(self.history_file.replace('history', 'historique')):
                if not os.path.exists(self.history_file):
                    os.rename(self.history_file.replace('history', 'historique'), self.history_file)

            self.__history = History.load(self.history_file)
        return self.__history
    
    @property
    def checkpoint_manager(self):
        if self.__checkpoint_manager is None:
            self.__checkpoint_manager = CheckpointManager(self, max_to_keep = self.max_to_keep)
        return self.__checkpoint_manager
    
    @property
    def default_metrics_config(self):
        return {}
    
    @property
    def default_loss_config(self):
        return {}
    
    @property
    def training_hparams(self):
        return {'augment_prct' : 0.25} if hasattr(self, 'augment_data') else {}
    
    @property
    def training_hparams_mapper(self):
        return {}

    @cached_property
    def call_signature(self):
        fn = getattr(self.model, 'call', self.model.__call__)
        return set(
            list(inspect.signature(fn).parameters) + ['run_eagerly', 'use_xla']
        )
    
    @property
    def run_eagerly(self):
        return self._run_eagerly if self._run_eagerly is not None else False
    
    @property
    def support_xla(self):
        return self._support_xla if self._support_xla is not None else True
    
    @property
    def graph_compile_config(self):
        return self._graph_compile_config if self._graph_compile_config is not None else {
            'prefer_xla'        : self.support_xla,
            'prepare_for_xla'   : getattr(self, 'prepare_for_xla', None),
            'prepare_for_graph' : getattr(self, 'prepare_for_graph', None)
        }
    
    @property
    def infer_compile_config(self):
        config = self.graph_compile_config
        if hasattr(self.model, 'prepare_for_xla'):
            config['prepare_for_xla'] = self.model.prepare_for_xla
        elif hasattr(self, 'prepare_for_xla_inference'):
            config['prepare_for_xla'] = self.prepare_for_xla_inference
        if hasattr(self, 'prepare_for_graph_inference'):
            config['prepare_for_graph'] = self.prepare_for_graph_inference
        return config

    @property
    def compiled_call(self):
        if self._compiled_call is None:
            if self.runtime != 'keras':
                self._compiled_call = self.model
            else:
                self._compiled_call = graph_compile(self.model, ** self.graph_compile_config)
        return self._compiled_call

    @property
    def compiled_infer(self):
        if self._compiled_infer is None:
            if self.runtime != 'keras':
                self._compiled_infer = self.model
            elif hasattr(self.model, 'infer'):
                self._compiled_infer = graph_compile(self.model.infer, ** self.infer_compile_config)
            else:
                self._compiled_infer = graph_compile(self.model, ** self.infer_compile_config)
        return self._compiled_infer
    
    def __str__(self):
        des = "\n========== {} ==========\n".format(self.name)
        des += "Model :\n"
        if self.runtime == 'keras':
            des += describe_model(self.model) + '\n'
            if hasattr(self, '_loss'):
                des += loss_to_str(self._loss) + '\n'
        else:
            des += str(self.model) + '\n\n'
        
        if self.pretrained_name:
            des += "Transfer-learning from : {}\n".format(self.pretrained_name)
        if self.runtime == 'keras':
            des += "Already trained on {} epochs ({} steps)\n\n".format(self.epochs, self.steps)
        
        return des

    def __call__(self, * args, training = False, mask = None, ** kwargs):
        if not hasattr(self, 'call'):
            if self.run_eagerly or self.runtime != 'keras':
                self.call = self.model
            else:
                self.call = graph_compile(self.model, ** self.graph_compile_config)
        
        kwargs.update({'training' : training, 'mask' : mask})
        if 'kwargs' not in self.call_signature:
            kwargs = {k : v for k, v in kwargs.items() if k in self.call_signature}
        return self.call(* args, ** kwargs)
    
    def compile(self,
                loss        = None,
                optimizer   = None,
                metrics     = None,
                
                loss_config = {},
                metrics_config  = {},
                optimizer_config    = {},
                
                overwrite   = False,
                
                ** kwargs
               ):
        import keras

        if hasattr(self, '_compiled_config') or hasattr(self, '_serialized_compile_config'):
            if hasattr(self, '_serialized_compile_config'):
                self.model.compile(** self._serialized_compile_config)
            
            if hasattr(self, '_compiled_config'):
                config = self._compiled_config
                del self._compiled_config
                self.compile(** config['losses'], ** config['optimizers'], ** config['metrics'])
            return
        
        from custom_train_objects.losses import get_loss
        from custom_train_objects.metrics import get_metrics
        from custom_train_objects.optimizers import get_optimizer
        
        if self.compiled and not overwrite:
            logger.warning('The model is already compiled. To verwrite the current compilation, pass `overwrite = True` as `compile` argument')
            return

        if 'metric' in kwargs:
            metrics         = kwargs.pop('metric')
            metrics_config  = kwargs.pop('metric_config', metrics_config)
        
        if loss is None:    loss = getattr(self, '_default_loss', None)
        if metrics is None: metrics = getattr(self, '_default_metrics', [])
        if optimizer is None:   optimizer = getattr(self, '_default_optimizer', 'adam')
        
        loss_config = {** self.default_loss_config, ** loss_config}
        metrics_config  = {** self.default_metrics_config, ** metrics_config}
        
        loss   = get_loss(loss, ** loss_config)
        metrics    = get_metrics(metrics, ** metrics_config)
        
        if hasattr(loss, 'output_names'):
            self.loss = loss
            self.model.compute_loss     = self.compute_multi_loss
            self.model._tracker.unlock()
            self.model.loss_metrics = {
                name : keras.metrics.Mean(name = name)
                for name in self.loss.output_names[1:]
            }
            self.model._tracker.lock()
            loss = None # loss is not propagated to model compilation
        
        if not self.model.compiled or overwrite:
            if not isinstance(metrics, list): metrics = [metrics]
            if len(metrics) == 0: metrics = None
            
            optimizer = get_optimizer(optimizer, ** optimizer_config)
            self.model.compile(optimizer = optimizer, loss = loss, metrics = metrics, ** kwargs)
        
        logger.info('Model compiled successfully !\n{}{}{}'.format(
            optimizer_to_str(self.optimizer), loss_to_str(loss), metrics_to_str(metrics)
        ))

    def compute_multi_loss(self, x, y = None, y_pred = None, sample_weight = None, ** kwargs):
        losses = self.loss(y, y_pred, sample_weight = sample_weight)
        
        for name, metric in self.model.loss_metrics.items():
            metric.update_state(losses[name])
        
        return losses['loss']
    
    def get_input(self, data, ** kwargs):
        """ Sequentially calls `prepare_input` then `process_input` if defined """
        inputs = data
        if hasattr(self, 'prepare_input'): inputs = self.prepare_input(inputs, ** kwargs)
        if hasattr(self, 'process_input'): inputs = self.process_input(inputs, ** kwargs)
        return inputs
    
    def _get_output(self, data, ** kwargs):
        """ Sequentially calls `prepare_output` then `process_output` if defined """
        output = data
        if hasattr(self, 'prepare_output'): output = self.prepare_output(output, ** kwargs)
        if hasattr(self, 'process_output'): output = self.process_output(output, ** kwargs)
        return output

    def _filter_data(self, inputs, output):
        valid_inp = self.filter_input(inputs) if hasattr(self, 'filter_input') else True
        valid_out = self.filter_output(output) if hasattr(self, 'filter_output') else True
        return ops.logical_and(valid_inp, valid_out)
    
    def get_dataset_config(self, mode, ** kwargs):
        """ Prepares the arguments for `prepare_dataset` to build the processing pipeline """
        assert mode in ('train', 'valid', 'predict')
        
        suffix = 'input' if mode == 'predict' else 'data'
        for prefix in ('augment_raw', 'prepare', 'filter', 'process', 'augment', 'process_batch'):
            if 'augment' in prefix and mode != 'train': continue
            
            key = f'{prefix}_fn'
            if key in kwargs: continue
            
            if hasattr(self, f'{prefix}_{suffix}'):
                kwargs[key] = getattr(self, f'{prefix}_{suffix}')
        
        if mode == 'train':
            kwargs.setdefault('shuffle', True)
        elif mode == 'valid':
            kwargs.update({'shuffle' : False})
        elif mode == 'predict':
            kwargs.update({'shuffle' : False, 'cache' : False})
        
        kwargs.update({
            k.replace(mode + '_', '') for k, v in kwargs.items()
            if k.startswith(mode + '_') and v is not None
        })
        return kwargs
    
    def prepare_dataset(self, dataset, mode, ** kwargs):
        return prepare_dataset(dataset, ** self.get_dataset_config(mode, ** kwargs))
    
    def prepare_for_training(self,
                             x,
                             y = None,
                             *,

                             epochs = 1,
                             
                             train_size = None,
                             valid_size = None,
                             validation_data    = None,
                             validation_split   = 0.2,
                             random_state   = 10,
                             pre_shuffle    = False,
                             
                             train_times    = 1,
                             valid_times    = 1,
                          
                             add_checkpoint = None,
                             add_early_stopping = False,
                             terminate_on_nan   = True,
                             
                             add_dataset_infos = True,
                             summary_kwargs    = {},

                             ** kwargs
                            ):
        from custom_train_objects.callbacks import HistoryCallback, get_callbacks
        
        dataset = x if y is None else (x, y)
        if isinstance(dataset, dict) and 'train' in dataset:
            validation_data = dataset.get('valid', dataset.get('test', validation_data))
            dataset         = dataset['train']
        
        if validation_data is None:
            dataset, validation_data = train_test_split(
                dataset, 
                train_size  = train_size,
                valid_size  = valid_size,
                random_state   = random_state,
                shuffle     = pre_shuffle,
                ** kwargs
            )
        else:
            if train_size:
                dataset, _ = train_test_split(
                    dataset,
                    train_size  = train_size,
                    random_state    = random_state,
                    shuffle = pre_shuffle,
                    ** kwargs
                )
            
            if valid_size:
                validation_data, _ = train_test_split(
                    validation_data,
                    train_size  = valid_size,
                    random_state    = random_state,
                    shuffle = pre_shuffle,
                    ** kwargs
                )
        
        train_dataset   = dataset
        valid_dataset   = validation_data
        
        ds_infos    = {}
        if add_dataset_infos:
            ds_infos    = {
                'train' : summarize_dataset(train_dataset, ** summary_kwargs),
                'valid' : summarize_dataset(valid_dataset, ** summary_kwargs)
            }
        
        train_dataset = self.prepare_dataset(train_dataset, mode = 'train', ** kwargs)
        valid_dataset = self.prepare_dataset(valid_dataset, mode = 'valid', ** kwargs)
        for k in ('batch_size', 'shuffle'): kwargs.pop(k, None)

        if train_times > 1: train_dataset = train_dataset.repeat(train_times)
        if valid_times > 1: valid_dataset = valid_dataset.repeat(valid_times)
        
        callbacks = kwargs.pop('callbacks', []) + [HistoryCallback(self.history)]
        if terminate_on_nan:
            callbacks.append({'class_name' : 'TerminateOnNaN'})
        if add_early_stopping:
            if add_checkpoint is None: add_checkpoint = True
            monitor = kwargs.get('monitor', 'val_loss')
            callbacks.append({
                'class_name'    : 'EarlyStopping',
                'min_delta'     : kwargs.get('min_delta', 1e-3),
                'patience'      : kwargs.get('patience', 3),
                'baseline'      : self.history.get_best(monitor),
                'monitor'       : monitor
            })
        if add_checkpoint:
            monitor = kwargs.get('monitor', 'val_loss')
            callbacks.append({
                'class_name'    : 'CheckpointCallback',
                'checkpoint_manager'    : self.checkpoint_manager,
                'save_best_only'    : True,
                'save_weights_only' : True,
                'monitor'   : monitor,
                'initial_value_threshold'   : self.history.get_best(monitor)
            })

        _allowed_kwargs = inspect.signature(self.model.fit).parameters
        return {
            'x'     : train_dataset,
            'epochs'    : epochs + self.epochs,
            'callbacks' : get_callbacks(callbacks),
            'initial_epoch' : self.epochs,
            'validation_data'   : valid_dataset,
            'dataset_infos' : ds_infos,
            ** {k : v for k, v in kwargs.items() if k in _allowed_kwargs}
        }
    
    def fit(self, * args, ** kwargs):
        train_hparams   = self.training_hparams.copy()
        train_hparams.update({k : kwargs.pop(k) for k in train_hparams if k in kwargs})
        
        self._init_train_config(** train_hparams)

        config  = self.prepare_for_training(* args, ** kwargs)

        self.history.set_config(
            hparams = train_hparams,
            config  = config,
            dataset_infos   = config.pop('dataset_infos', {}),
            ** {k : v for k, v in kwargs.items() if k not in config}
        )
        
        logger.info("Training config :\n{}\n".format('\n'.join([
            '- {}\t:{}'.format(k, v) for k, v in {** config, ** train_hparams}.items()
        ])))
        
        start = time.time()
        try:
            _ = self.model.fit(** config)
        except KeyboardInterrupt as e:
            logger.warning("Training interrupted !")
        
        logger.info("Training finished after {} !".format(time_to_string(time.time() - start)))
        self.save()
        
        return self.history

    @timer
    def predict(self,
                inputs,
                *,
                
                predicted = None,
                callbacks = None,
                
                return_results  = True,
                return_output   = None,
                
                ** kwargs
               ):
        join_callbacks = predicted is None
        if predicted is None:
            predicted, callbacks = self.get_inference_callbacks(** kwargs)
        
        if return_output is None:
            return_output = not any(isinstance(callback, JSONSaver) for callback in callbacks)
        
        kwargs.update({
            'predicted' : predicted,
            'callbacks' : callbacks,
            'return_output' : return_output
        })
        
        results = []
        for text, output in Stream(self.infer, inputs, ** kwargs).items():
            if return_results:
                results.append(output if return_output else predicted[text])
        
        if join_callbacks:
            for callback in callbacks: callback.join()
        
        return results

    stream = partialmethod(predict, return_output = False, return_results = False)
    
    def get_config(self):
        return {
            'name'  : self.name,
            'runtime'   : self.runtime,
            'pretrained_name'   : self.pretrained_name,

            'run_eagerly'   : self._run_eagerly,
            'support_xla'   : self._support_xla,
            'graph_compile_config'  : self._graph_compile_config,
            
            ** self.build_kwargs
        }
        
    def save(self, ** kwargs):
        if not self._save: return
        
        if self.runtime == 'keras':
            self.save_models_config(** kwargs)
            self.save_checkpoint(** kwargs)
            self.save_history(** kwargs)
        self.save_config(** kwargs)
    
    def save_history(self, directory = None, ** _):
        filename = self.history_file if not directory else os.path.join(directory, 'history.json')
        self.history.save(filename)
    
    def save_models_config(self, directory = None, ** _):
        import keras
        
        config = {'model' : keras.saving.serialize_keras_object(self.model)}
        if getattr(self, '_loss', None) is not None:
            config['loss'] = keras.saving.serialize_keras_object(self._loss)
        
        config_file = self.config_models_file if not directory else os.path.join(
            directory, 'config_models.json'
        )
        dump_json(config_file, config, indent = 4)
    
    def save_config(self, directory = None, ** _):
        config_file = self.config_file if not directory else os.path.join(directory, 'config.json')
        config      = {
            'class_name'    : self.__class__.__name__,
            'config'        : self.get_config()
        }
        
        dump_json(config_file, config, indent = 4)
        
    def _restore_model(self, compile = False, ** _):
        config = load_json(self.config_models_file)

        if 'models' in config:
            self._restore_model_old(config, compile = compile)
            self.save_models_config()
            return
        
        import keras
        
        from architectures import get_custom_objects
        
        if config['model'].get('compile_config', {}) and not compile:
            self._serialized_compile_config = config['model'].pop('compile_config')

        self.model = keras.saving.deserialize_keras_object(
            config['model'], custom_objects = get_custom_objects()
        )
        if 'loss' in config:
            if compile:
                self._loss = keras.saving.deserialize_keras_object(config['loss'])
            else:
                self._serialized_loss = config['loss']
        self.checkpoint_manager.load()

    def _restore_model_old(self, config, compile = False, ** kwargs):
        import keras
        
        from architectures import get_custom_objects
        from custom_train_objects.losses import get_loss
        from custom_train_objects.metrics import get_metrics
        
        _load_weights   = True
        
        if len(config['models']) > 1:
            raise NotImplementedError('{} has more than 1 model ({}) which is not supported anymore'.format(self.name, tuple(config['models'].keys())))
        
        key, model_config = list(config['models'].items())[0]

        filename = self.checkpoint_manager.loaded_checkpoint
        if not filename: filename = os.path.join(self.save_dir, '{}.keras'.format(key))
        
        if filename.endswith('.keras') and os.path.exists(filename):
            logger.info('Loading `{}` from {}'.format(key, filename))
            self.model = keras.models.load_model(filename)
            _load_weights = False
        
        elif 'module' in model_config:
            logger.info('Deserializing `{}` from config'.format(key))
            model_config = keras.tree.map_structure(
                lambda k: k if not isinstance(k, str) or 'custom_' not in k else k
                    .replace('custom_architectures', 'architectures')
                    .replace('custom_layers.', 'architectures.layers.')
                    .replace('transformers_arch.', 'transformers.'),
                model_config
            )
            
            self.model = keras.saving.deserialize_keras_object(
                model_config, custom_objects = get_custom_objects()
            )
        elif os.path.exists(filename.replace('.keras', '.json')):
            filename = filename.replace('.keras', '.json')

            logger.info('Loading `{}` from {}'.format(key, filename))
            with open(filename, 'r', encoding = 'utf-8') as file:
                json_config = file.read()
            model_config    = json.loads(json_config)

            if 'module' not in model_config:
                logger.info('Updating keras 2 config')
                from architectures import deserialize_keras2_model
                self.model = deserialize_keras2_model(model_config)
            else:
                self.model = model_from_json(json_config)

        logger.info('`model` successfully restored !')
        
        if config['losses']:
            if '{}_optimizer'.format(key) in config.get('optimizers', {}):
                for k in ('optimizers', 'losses', 'metrics'):
                    if len(config[k]) > 0: config[k] = list(config[k].values())[0]
                config['losses'].get('loss_config', {}).pop('reduction', None)
            
            self.compile(** config['losses'], ** config['optimizers'], ** config['metrics'])

        if _load_weights:   self.checkpoint_manager.load()

    @classmethod
    def from_pretrained(cls, name, pretrained, ** kwargs):
        """
            Creates a copy of `pretrained` by using its configuration + transfering model weights
            
            Note : the transfer is *partial*, meaning that the new model architecture can be modified (by passing specific new `kwargs`)
            
            **Important note** : the pretrained model is loaded on CPU, so it is higly recommanded to restart the kernel before using the new instance to free memory
        """
        if isinstance(pretrained, str):
            if not is_model_name(pretrained):
                raise ValueError('The model `{}` is not available'.format(pretrained))
            
            import keras
            
            from .. import get_pretrained
            with keras.device('cpu'):
                pretrained = get_pretrained(pretrained)
        
        config = pretrained.get_config()
        config.update({'name' : name, 'pretrained_name' : pretrained.name, ** kwargs})
        
        instance = cls(max_to_keep = 1, ** config)
        
        name_based_partial_transfer_learning(instance.model, pretrained.model)
        
        instance.save()
        
        return instance

def _build_generic_processing(input_processing, output_processing, name):
    if name.startswith('prepare'):
        def inner(* data, ** kwargs):
            if len(data) == 1:
                if isinstance(data[0], tuple) and len(data[0]) == 2:
                    inputs, output = data[0][0], data[0][1]
                else:
                    inputs, output = data[0], data[0]
            elif len(data) == 2:
                inputs, output = data
            else:
                raise RuntimeError('input `data` should either be single element (`dict`) either 2-elements (`inputs` and `outputs`). Get {} elements instead :\n{}'.format(len(data), data))
            
            inputs = input_processing(inputs, ** kwargs)
            if has_input_kwarg: kwargs['inputs'] = inputs
            return inputs, output_processing(output, ** kwargs)
    else:
        def inner(inputs, output, ** kwargs):
            inputs = input_processing(inputs, ** kwargs)
            if has_input_kwarg: kwargs['inputs'] = inputs
            return inputs, output_processing(output, ** kwargs)

    has_input_kwarg = 'inputs' in inspect.signature(output_processing).parameters
    inner.__name__ = name
    return inner
