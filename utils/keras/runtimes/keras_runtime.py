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

import inspect
import logging

from functools import cached_property

from .. import timer, graph_compile
from .runtime import Runtime

logger  = logging.getLogger(__name__)

class KerasRuntime(Runtime):
    """
        Wraps a `keras.Model` behind the `Runtime` interface, making `keras` "just another
        runtime" for `BaseModel`.

        Unlike inference-only runtimes, the wrapped `keras.Model` is usually **injected** via
        `engine = ...` (built / restored by the `models` layer, which owns `architectures` and
        `custom_train_objects`) rather than loaded from `path`. This keeps `utils.keras.runtimes`
        free of any upward dependency on the `models` layer. `load_engine` is only used to reload
        a plain `.keras` artifact for pure inference.

        This runtime owns the whole **execution** concern (previously scattered in `BaseModel`) :
            - `__call__`            : signature-filtered forward pass
            - `compiled_call`       : graph/XLA-compiled forward pass
            - `compiled_infer`      : graph/XLA-compiled `infer` (generation) pass

        The XLA / graph hooks that live on the model sub-class (e.g. `prepare_for_xla_inference`
        in `WaveGlow`) cannot descend into this layer, so they are **injected** at construction,
        exactly like `SavedModelRuntime` does with `prepare_for_xla` / `prepare_for_graph`.
    """
    supports_training   = True

    def __init__(self,
                 path   = None,
                 *,

                 engine = None,

                 run_eagerly    = None,
                 support_xla    = None,
                 graph_compile_config   = None,

                 prepare_for_xla    = None,
                 prepare_for_graph  = None,
                 prepare_for_xla_inference      = None,
                 prepare_for_graph_inference    = None,

                 ** kwargs
                ):
        super().__init__(path, engine = engine, ** kwargs)

        self._run_eagerly   = run_eagerly
        self._support_xla   = support_xla
        self._graph_compile_config  = graph_compile_config

        # `call`-time hooks (used by `compiled_call`)
        self._prepare_for_xla   = prepare_for_xla
        self._prepare_for_graph = prepare_for_graph
        # `infer`-time hooks (used by `compiled_infer`)
        self._prepare_for_xla_inference     = prepare_for_xla_inference
        self._prepare_for_graph_inference   = prepare_for_graph_inference

        self._compiled_call     = None
        self._compiled_infer    = None

    def __repr__(self):
        return '<{} engine={}>'.format(self.__class__.__name__, self.engine.__class__.__name__)

    def __getattr__(self, name):
        """ Transparently delegates unknown attributes to the wrapped `keras.Model` (e.g.
            `variables`, `summary`, `layers`, `output_shape`, `embedding_dim`, ...). """
        # `engine` is set in `super().__init__`; guard against recursion if it is missing
        if name == 'engine':
            raise AttributeError(name)
        return getattr(self.engine, name)

    def __setattr__(self, name, value):
        """ Symmetric counterpart to `__getattr__` : writes that do not target the wrapper's own
            state are forwarded to the wrapped `keras.Model`, so that e.g. `runtime.trainable = False`
            actually freezes the keras model (keras and callers expect to set attributes on the model,
            not on the runtime wrapper).

            Wrapper-owned attributes (`path`, `engine` and every private `_*` one) stay local. The
            `'engine' not in self.__dict__` guard forces every write to stay local until the engine
            has been assigned (i.e. during `super().__init__`). """
        if name in ('path', 'engine') or name.startswith('_') or 'engine' not in self.__dict__:
            super().__setattr__(name, value)
        else:
            setattr(self.engine, name, value)

    @property
    def base_dtype(self):
        """ Compute dtype of the wrapped `keras.Model` (honours the mixed-precision policy). """
        return getattr(self.engine, 'compute_dtype', 'float32')

    @property
    def run_eagerly(self):
        return self._run_eagerly if self._run_eagerly is not None else False

    @property
    def support_xla(self):
        return self._support_xla if self._support_xla is not None else True

    @cached_property
    def call_signature(self):
        fn = getattr(self.engine, 'call', self.engine.__call__)
        return set(
            list(inspect.signature(fn).parameters) + ['run_eagerly', 'use_xla']
        )

    @property
    def graph_compile_config(self):
        return self._graph_compile_config if self._graph_compile_config is not None else {
            'prefer_xla'        : self.support_xla,
            'prepare_for_xla'   : self._prepare_for_xla,
            'prepare_for_graph' : self._prepare_for_graph
        }

    @property
    def infer_compile_config(self):
        # `dict(...)` : never mutate the (possibly shared) `graph_compile_config` dict
        config = dict(self.graph_compile_config)
        if hasattr(self.engine, 'prepare_for_xla'):
            config['prepare_for_xla'] = self.engine.prepare_for_xla
        elif self._prepare_for_xla_inference is not None:
            config['prepare_for_xla'] = self._prepare_for_xla_inference
        if self._prepare_for_graph_inference is not None:
            config['prepare_for_graph'] = self._prepare_for_graph_inference
        return config

    @property
    def compiled_call(self):
        if self._compiled_call is None:
            self._compiled_call = graph_compile(self.engine, ** self.graph_compile_config)
        return self._compiled_call

    @property
    def compiled_infer(self):
        if self._compiled_infer is None:
            fn = self.engine.infer if hasattr(self.engine, 'infer') else self.engine
            self._compiled_infer = graph_compile(fn, ** self.infer_compile_config)
        return self._compiled_infer

    @timer(name = 'Keras runtime inference')
    def __call__(self, * args, training = False, mask = None, ** kwargs):
        kwargs.update({'training' : training, 'mask' : mask})
        if 'kwargs' not in self.call_signature:
            kwargs = {k : v for k, v in kwargs.items() if k in self.call_signature}

        call_fn = self.engine if self.run_eagerly else self.compiled_call
        return call_fn(* args, ** kwargs)

    @staticmethod
    def load_engine(path, ** _):
        """
            Reloads a plain `.keras` artifact (pure-inference deployment). Training / restoration
            from the model directory is orchestrated by the `models` layer and injected via
            `engine = ...`.
        """
        import keras

        return keras.models.load_model(path)
