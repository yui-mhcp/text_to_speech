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

import time
import inspect
import logging

from loggers import timer
from utils import time_to_string
from ..utils import loss_to_str, optimizer_to_str, metrics_to_str

logger = logging.getLogger(__name__)

class TrainableModelMixin:
    """
        Training / compilation surface of a `BaseModel`. Only meaningful when the underlying runtime
        is trainable (`self.model.supports_training`, i.e. a `KerasRuntime`) ; on inference-only runtimes
        these methods will fail explicitly (keras internals / `self.model.engine` are unavailable).
    """

    @property
    def loss(self):
        return self._loss if hasattr(self, '_loss') else self.model.loss

    @loss.setter
    def loss(self, value):
        self._loss = value

    optimizer   = property(lambda self: self.model.optimizer)
    compiled    = property(
        lambda self: getattr(self.model, 'compiled', False) and self.loss is not None
    )

    @property
    def default_metrics_config(self):
        return {}

    @property
    def default_loss_config(self):
        return {}

    @property
    def training_hparams(self):
        return {'augment_prct' : 0.25} if hasattr(self, 'augment_data') else {}

    @timer(debug = True)
    def _apply_training_hparams(self, ** kwargs):
        """
            Applies the training hyper-parameters for the duration of a `fit` call : each
            `training_hparams` entry gives a parameter and its default, and the resolved value
            (`kwargs` override or default) overwrites the matching instance attribute.

            A `None` default means "only set it if the attribute does not exist yet" : it lets a
            real config attribute (e.g. `Tacotron2.max_input_length`) be *optionally* overridden at
            `fit` time without being clobbered when the caller does not pass it.

            Only ever called from `fit` (which requires a trainable runtime), so inference-only
            models are never touched.
        """
        for k, default in self.training_hparams.items():
            val = kwargs.get(k, default)
            if not hasattr(self, k) or val is not None:
                setattr(self, k, val)

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
        if not self.model.supports_training:
            raise RuntimeError("compile() requires a trainable runtime, but '{}' is inference-only".format(self.runtime))

        if self._compile_from_serialized_config():
            return

        import keras

        from custom_train_objects.losses import get_loss
        from custom_train_objects.metrics import get_metrics
        from custom_train_objects.optimizers import get_optimizer

        if self.compiled and not overwrite:
            logger.warning('The model is already compiled. To overwrite the current compilation, pass `overwrite = True` as `compile` argument')
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
            # writes must target the raw keras model (`engine`) : `KerasRuntime.__getattr__`
            # delegates reads, not attribute assignments, and keras internals use `engine.*`
            model   = self.model.engine
            model.compute_loss  = self.compute_multi_loss
            model._tracker.unlock()
            model.loss_metrics  = {
                name : keras.metrics.Mean(name = name)
                for name in self.loss.output_names[1:]
            }
            model._tracker.lock()
            loss = None # loss is not propagated to model compilation

        if not self.model.compiled or overwrite:
            if not isinstance(metrics, list): metrics = [metrics]
            if len(metrics) == 0: metrics = None

            optimizer = get_optimizer(optimizer, ** optimizer_config)
            self.model.compile(optimizer = optimizer, loss = loss, metrics = metrics, ** kwargs)

        logger.info('Model compiled successfully !\n{}{}{}'.format(
            optimizer_to_str(self.optimizer), loss_to_str(loss), metrics_to_str(metrics)
        ))

    def _compile_from_serialized_config(self):
        """ Re-applies a keras `compile_config` that was serialized at save time (deferred to the
            first explicit `compile` call). Returns whether it handled the compilation. """
        if self.serialized_compile_config is None:
            return False

        self.model.compile(** self.serialized_compile_config)
        return True

    def compute_multi_loss(self, x, y = None, y_pred = None, sample_weight = None, ** kwargs):
        losses = self.loss(y, y_pred, sample_weight = sample_weight)

        for name, metric in self.model.loss_metrics.items():
            metric.update_state(losses[name])

        return losses['loss']

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
        from utils.datasets import train_test_split
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
            from utils.datasets import summarize_dataset

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
        if not self.model.supports_training:
            raise RuntimeError("fit() requires a trainable runtime, but '{}' is inference-only".format(self.runtime))

        train_hparams   = self.training_hparams.copy()
        train_hparams.update({k : kwargs.pop(k) for k in train_hparams if k in kwargs})

        self._apply_training_hparams(** train_hparams)

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
