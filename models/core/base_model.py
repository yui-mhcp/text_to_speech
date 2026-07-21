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
import logging

from loggers import Timer, timer
from utils import Stream, copy_methods, copy_properties, load_json, map_structure
from utils.callbacks import apply_callbacks
from utils.keras import Runtime, TensorSpec, build_runtime
from custom_train_objects import CheckpointManager, History

from .mixins import ModelCheckpointMixin, TrainableModelMixin, ModelProcessingMixin
from .utils import get_saving_dir, get_model_dir, describe_model, loss_to_str, pop_legacy_model_name

logger = logging.getLogger(__name__)

class ModelInstances(type):
    _instances = {}

    @timer(name = 'model loading', debug = True)
    def __call__(cls, * args, reload = False, name = None, ** kwargs):
        if not name: name = pop_legacy_model_name(kwargs, cls.__name__)

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
    'summary',
    'save_weights', 'load_weights', 'set_weights', 'get_weights',
    'train_on_batch', 'test_on_batch', 'predict_on_batch',
)
@copy_properties(
    'model',
    'inputs', 'outputs',
    'variables', 'trainable_variables', 'non_trainable_variables',
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
class BaseModel(ModelCheckpointMixin, TrainableModelMixin, ModelProcessingMixin, metaclass = ModelInstances):
    """
        Abstract class wrapping a `Runtime` (either a `KerasRuntime` around a `keras.Model`, or an
        inference-only runtime such as TensorRT / ONNX), and defining the processing functions / parameters

        Each `BaseModel` is identified by its `name`, which **must** be unique

        The class is split across mixins for readability :
            - `ModelProcessingMixin`    : input / output processing pipeline
            - `TrainableModelMixin`     : compilation / training (only usable when `self.model.supports_training`)
            - `ModelCheckpointMixin`    : configuration / weights persistence and restoration

        Arguments :
            - name  : the model's name
            - root  : the root directory where to save configuration / weights files
            - save  : whether to save the model or not
            - max_to_keep   : maximum number of checkpoint files to keep

            - pretrained_name   : the original model used for transfer-learning / initialization
            - force_rebuild     : whether to force calling `build()` instead of `restore_models`

            - runtime   : the execution runtime (`keras` by default ; any key from `build_runtime`)

        Sub-classes have to override the `build` method in order to define the configuration to instanciate the `keras.Model` instances, then call `super().build(model = config)` in order to properly initialize the model
        The method can also directly sets the attribute : `self.model = model`

        **Initialization contract** : modality mixins (`TextModelMixin`, `AudioModelMixin`, ...) expose
        an `_init_<modality>(...)` hook that a leaf model **must** call **before** `super().__init__(...)`,
        because `build()` may depend on modality state (e.g. `vocab_size` derives from `tokenizer`).
        The core mixins' `_init_*` hooks, on the contrary, are called by `BaseModel.__init__` itself.

        **Shape / dtype contract** : `input_signature` / `output_signature` are the **primitive** that
        each sub-class (or modality mixin, via `<modality>_signature`) **must** define ; the default
        raises `NotImplementedError`. Everything else is *derived* from them : `input_shape` /
        `output_shape` / `input_dtype` / `output_dtype` (via `map_structure`, so nested signatures give
        nested shapes/dtypes) and `unbatched_{input,output}_signature`. Float signatures should use
        `self.base_dtype` (the runtime's real compute precision, e.g. `float16` for a fp16 TensorRT
        engine) so they avoid useless host casts ; integer signatures (token ids) keep `int32`.

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

        def augment_data(self, inputs, output):
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
    def __init_subclass__(cls, ** kwargs):
        """
            Builds the directory / file path properties once per sub-class (from `_directories` /
            `_files`), instead of re-creating them on every instantiation.
        """
        super().__init_subclass__(** kwargs)
        for property_name, path_format in {** cls._directories, ** cls._files}.items():
            setattr(cls, property_name, _make_path_property(path_format))

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

        # Drop kwargs already consumed by the sub-class `_init_*` hooks (they set them as instance
        # attributes *before* `super().__init__`), so they are not re-forwarded to `build` nor
        # persisted twice through `build_kwargs`. Filtering on `self.__dict__` targets *only* the
        # instance attributes actually set, unlike `hasattr` which also matches class
        # methods / properties (`@copy_methods` / `@copy_properties`) and would trigger their getters
        # mid-construction (e.g. `self.loss` -> `self.model.loss` while `self._model` is unset).
        kwargs  = {k : v for k, v in kwargs.items() if k not in self.__dict__}

        self.name   = name
        self._save  = save
        self.runtime    = runtime
        self.build_kwargs   = kwargs
        self.pretrained_name    = pretrained_name

        self.max_to_keep    = max_to_keep
        self._run_eagerly   = run_eagerly
        self._support_xla   = support_xla
        self._graph_compile_config  = graph_compile_config

        self._init_processing_functions()
        self._init_persistence()

        # Note : the `runtime == 'keras'` branches (here, `save`, `__str__`) are NOT capability checks
        # (`self.model.supports_training`) but require an actual `keras.Model` (build / (de)serialize /
        # `describe_model`), which only exists for the keras runtime.
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

        # Training hyper-parameters are applied lazily in `fit` (`_apply_training_hparams`), not at
        # construction : inference-only models (e.g. non-keras runtimes) must never get training
        # config set on them.

        logger.info("{} `{}` initialized successfully !".format(self.__class__.__name__, self.name))

    @timer(debug = True)
    def _init_directories(self):
        """ Initialize directory structure based on `self._directories` """
        for property_name in self._directories.keys():
            os.makedirs(getattr(self, property_name), exist_ok = True)

    @timer(debug = True)
    def build(self, model = None, ** kwargs):
        """ Initializes the effective `keras.Model` (wrapped into a `KerasRuntime` by the setter) """
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

    # Native compute *float* precision of the runtime, 
    base_dtype  = property(
        lambda self: getattr(getattr(self, '_model', None), 'base_dtype', 'float32')
    )

    @property
    def input_signature(self):
        raise NotImplementedError('`{}` must define `input_signature`'.format(
            self.__class__.__name__
        ))

    @property
    def output_signature(self):
        raise NotImplementedError('`{}` must define `output_signature`'.format(
            self.__class__.__name__
        ))

    input_shape     = property(lambda self: map_structure(_spec_shape, self.input_signature))
    output_shape    = property(lambda self: map_structure(_spec_shape, self.output_signature))
    input_dtype     = property(lambda self: map_structure(_spec_dtype, self.input_signature))
    output_dtype    = property(lambda self: map_structure(_spec_dtype, self.output_signature))

    # signatures without the leading (batch) axis, e.g. for per-sample dataset generators
    unbatched_input_signature   = property(lambda self: map_structure(_unbatch_spec, self.input_signature))
    unbatched_output_signature  = property(lambda self: map_structure(_unbatch_spec, self.output_signature))

    # Compiled version of `__call__` using e.g., `graph_compile` (runtime-specific)
    compiled_call   = property(lambda self: self.model.compiled_call)
    compiled_infer  = property(lambda self: self.model.compiled_infer)

    @model.setter
    def model(self, value):
        if not isinstance(value, Runtime):
            value = build_runtime(self.runtime, engine = value, ** self._exec_config)
        self._model = value
        if value.supports_training:
            self.checkpoint_manager.model = value.engine

    @property
    def _exec_config(self):
        """ Execution parameters forwarded to `KerasRuntime` (call / infer graph compilation). """
        return {
            'run_eagerly'   : self._run_eagerly,
            'support_xla'   : self._support_xla,
            'graph_compile_config'  : self._graph_compile_config,
            'prepare_for_xla'   : getattr(self, 'prepare_for_xla', None),
            'prepare_for_graph' : getattr(self, 'prepare_for_graph', None),
            'prepare_for_xla_inference'     : getattr(self, 'prepare_for_xla_inference', None),
            'prepare_for_graph_inference'   : getattr(self, 'prepare_for_graph_inference', None)
        }

    def __str__(self):
        des = "\n========== {} ==========\n".format(self.name)
        des += "Model :\n"
        # keras-only : `describe_model` introspects a real `keras.Model` (see `__init__` note)
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
        """ Calls `self.model` with the provided arguments """
        return self.model(* args, training = training, mask = mask, ** kwargs)

    def infer(self, data, *, callbacks = None, predicted = None, overwrite = False, return_output = True, ** kwargs):
        """
            Per-sample full prediction. **Must** be overriden by each sub-class ; must NOT batch.

            **Inference callbacks contract** : `predict` initializes `(predicted, callbacks)` once
            (via `get_inference_callbacks`), then forwards them to every `infer` call, which is
            responsible for applying them :
                - `callbacks` : `list` of `Callback` applied on the output. They run in the thread
                  executing `infer`, which may be a `Stream` worker (`max_workers > 0`) : callbacks
                  must therefore be thread-safe (`FileSaver` and its subclasses already are)
                - `predicted` : the persisted entries mapping (`{key : entry}`, i.e. the `map.json`
                  content), which is the **same** object as `JSONSaver.data` : new entries appear
                  in it once the `JSONSaver` callback has been applied
                - `overwrite` : when the data's key is already in `predicted` and `overwrite` is
                  False, `infer` should skip the computation and delegate to
                  `_finalize_cached_prediction` (so that display / post-processing callbacks are
                  still applied, while `saves_to_disk` ones are skipped)
                - `return_output` : whether to return the full (heavy) output or the lightweight
                  stored entry (see `_finalize_predictions`, which resolves it)

            A typical implementation therefore looks like :
            ```python
            def infer(self, data, *, callbacks = None, predicted = None, overwrite = False,
                      return_output = True, ** kwargs):
                if predicted and not overwrite and isinstance(data, str) and data in predicted:
                    return self._finalize_cached_prediction(
                        data, callbacks = callbacks, predicted = predicted
                    )

                output = ... # the effective inference
                return self._finalize_predictions(
                    output, callbacks = callbacks, predicted = predicted, return_output = return_output
                )
            ```

            Note : the duplicate check is **not** atomic across threads : two workers receiving
            the same key concurrently will both compute it. This is benign (`JSONSaver` serializes
            the persisted state under its mutex ; the only cost is duplicated work), and should not
            happen in practice.
        """
        raise NotImplementedError('`{}` must implement the `infer` method'.format(
            self.__class__.__name__
        ))

    def get_inference_callbacks(self, ** kwargs):
        """ Returns the `(predicted, callbacks)` couple used by `predict`. Sub-class responsibility. """
        raise NotImplementedError('`{}` must implement the `get_inference_callbacks` method'.format(
            self.__class__.__name__
        ))

    def filter_prediction_output(self, output):
        """
            Returns the lightweight persisted `infos` derived from a full `infer` output, i.e.
            without the heavy / non-serializable entries (raw image / audio / model output, ...)
        """
        raise NotImplementedError('`{}` must implement the `filter_prediction_output` method'.format(
            self.__class__.__name__
        ))

    def filter_returned_output(self, output):
        """
            The value returned by `infer` when `return_output == False and save = False`.
            By default equivalent to `filter_prediction_output`, but allows to keep some
            relevant parts of the output to return (e.g., raw audio for the Tacotron model)
        """
        return self.filter_prediction_output(output)

    def _finalize_predictions(self, output, *, callbacks = None, predicted = None, return_output = True, save = True):
        """
            Applies `callbacks` on a **newly computed** `infer` output, then resolves the value to
            return (see the `infer` contract)

            Arguments :
                - output    : the full `infer` output (`dict`)
                - callbacks / predicted : see `infer`
                - return_output : whether to return the full `output` or the persisted entry
                - save      : forwarded to `apply_callbacks` (`saves_to_disk` callbacks are skipped
                              when False)
            Return :
                - result    : `output` if `return_output`, otherwise the entry stored in
                              `predicted` (when a `provides_entry` callback persisted it), with
                              `filter_returned_output(output)` (the serializable in-memory view)
                              as fallback

            Note : callbacks may enrich `output` **in-place** (e.g. `initializer`s, such as the
            `detected` drawn image), so callers may rely on those keys after this call. The
            fallback is (re)computed from `output` — the source of truth — rather than the
            callback-mutated `infos`, so those enrichments are preserved in the returned value
        """
        entry = None
        if callbacks:
            infos = self.filter_prediction_output(output)
            entry = apply_callbacks(callbacks, infos, output, save = save)

        if return_output:
            return output
        elif predicted is not None and entry is not None and entry in predicted:
            return predicted[entry]
        else:
            return self.filter_returned_output(output)

    def _finalize_cached_prediction(self, key, *, callbacks = None, predicted = None):
        """
            Re-applies `callbacks` on an entry already stored in `predicted` (duplicate detection
            in `infer`), then returns it. `save = False` skips the `saves_to_disk` callbacks, while
            display / post-processing ones still run. The `output` argument is a **shallow copy**
            of the entry, so that callback `initializer`s can enrich it without polluting the
            persisted entry
        """
        infos = predicted[key]
        if callbacks:
            apply_callbacks(callbacks, infos, {}, save = False)
        return infos

    def _normalize_prediction_inputs(self, inputs):
        """
            Wraps single-sample `inputs` into a `list` for `predict`. Sub-classes may extend it
            for raw array inputs (e.g. rank-1 audio / rank-3 image), which cannot be distinguished
            from a batch at the `BaseModel` level
        """
        if isinstance(inputs, (str, dict)): return [inputs]
        return inputs

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
        """
            Runs `self.infer` on each item of `inputs` (any type supported by `Stream` : `list`,
            generator, `Queue`, ... ; `max_workers` in `kwargs` enables multi-threaded inference).
            The inference callbacks are initialized **once** (via `get_inference_callbacks`), then
            forwarded to every `infer` call, which applies them (see the `infer` contract)
        """
        inputs = self._normalize_prediction_inputs(inputs)

        join_callbacks = predicted is None
        if predicted is None:
            predicted, callbacks = self.get_inference_callbacks(** kwargs)

        if return_output is None:
            return_output = not any(callback.provides_entry for callback in callbacks)

        kwargs.update({
            'predicted' : predicted,
            'callbacks' : callbacks,
            'return_output' : return_output
        })

        # `infer` applies the callbacks and resolves `return_output` itself (see
        # `_finalize_predictions`), so its output is directly the expected result
        results = []
        for _, output in Stream(self.infer, inputs, ** kwargs).items():
            if return_results:
                results.append(output)

        if join_callbacks:
            with Timer('waiting callbacks'):
                for callback in callbacks: callback.join()

        return results

    def stream(self, inputs, ** kwargs):
        kwargs.setdefault('return_output', False)
        kwargs.setdefault('return_results', False)
        return self.predict(inputs, ** kwargs)

    def get_config(self):
        """
            `BaseModel`-level configuration (name / runtime / execution + `build_kwargs`).
            Sub-classes and modality mixins extend it additively, e.g. :
                `return {** super().get_config(), ** self.get_config_text()}`
        """
        return {
            'name'  : self.name,
            'runtime'   : self.runtime,
            'pretrained_name'   : self.pretrained_name,

            'run_eagerly'   : self._run_eagerly,
            'support_xla'   : self._support_xla,
            'graph_compile_config'  : self._graph_compile_config,

            ** self.build_kwargs
        }

def _make_path_property(path_format):
    return property(lambda self: path_format.format(self = self, root = get_saving_dir()))

# leaf helpers deriving `shape` / `dtype` / unbatched specs from a (possibly nested) signature
def _spec_shape(spec):
    return tuple(spec.shape)

def _spec_dtype(spec):
    return getattr(spec.dtype, 'name', spec.dtype)

def _unbatch_spec(spec):
    return TensorSpec(shape = tuple(spec.shape)[1:], dtype = _spec_dtype(spec))
