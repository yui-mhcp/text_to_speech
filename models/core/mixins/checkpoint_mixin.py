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

from utils import dump_json, load_json
from custom_train_objects import CheckpointManager, History

from ..utils import is_model_name, migrate_legacy_history_file, restore_legacy_model
from ...weights_converter import name_based_partial_transfer_learning

logger = logging.getLogger(__name__)

class ModelCheckpointMixin:
    """
        Persistence surface of a `BaseModel` : configuration (`config.json`), keras model definition
        (`config_models.json`), checkpoints and `History`. Everything keras-specific is guarded by
        `self.runtime == 'keras'` (only the config is saved for inference-only runtimes).
    """

    def _init_persistence(self):
        """ Initializes the persistence / restoration state owned by this mixin. Called by
            `BaseModel.__init__` **before** build / restore, since `_restore_model` populates part
            of it (the deferred-compilation configs). """
        self._history                   = None
        self._checkpoint_manager        = None
        # Deferred compilation : the keras native `compile_config` is produced here at restore time
        # and consumed by `TrainableModelMixin.compile` on the first explicit `compile()` call.
        # Exposed read-only via the property below so the training mixin does not reach into the private.
        self._serialized_compile_config = None

    @property
    def serialized_compile_config(self):
        """ keras `compile_config` deferred at restore time, applied on the first `compile()` (or `None`). """
        return self._serialized_compile_config

    @property
    def history(self):
        if self._history is None:
            migrate_legacy_history_file(self.history_file)
            self._history = History.load(self.history_file)
        return self._history

    @property
    def checkpoint_manager(self):
        if self._checkpoint_manager is None:
            self._checkpoint_manager = CheckpointManager(self, max_to_keep = self.max_to_keep)
        return self._checkpoint_manager

    def save(self, ** kwargs):
        if not self._save: return

        # keras-only : (de)serializing weights / config / history needs an actual `keras.Model`
        # (not a capability check ; inference-only runtimes only persist `config.json`)
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

        config = {'model' : keras.saving.serialize_keras_object(self.model.engine)}
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
            restore_legacy_model(self, config, compile = compile)
            self.save_models_config()
            return

        import keras

        from architectures import get_custom_objects

        if config['model'].get('compile_config', {}) and not compile:
            self._serialized_compile_config = config['model'].pop('compile_config')

        self.model = keras.saving.deserialize_keras_object(
            config['model'], custom_objects = get_custom_objects()
        )
        if 'loss' in config and compile:
            self._loss = keras.saving.deserialize_keras_object(config['loss'])
        self.checkpoint_manager.load()

    @classmethod
    def from_pretrained(cls, name, pretrained, ** kwargs):
        """
            Creates a copy of `pretrained` by using its configuration + transfering model weights

            Note : the transfer is *partial*, meaning that the new model architecture can be modified (by passing specific new `kwargs`)

            **Important note** : the pretrained model is loaded on CPU, so it is highly recommended to restart the kernel before using the new instance to free memory
        """
        if isinstance(pretrained, str):
            if not is_model_name(pretrained):
                raise ValueError('The model `{}` is not available'.format(pretrained))

            import keras

            from ... import get_pretrained
            with keras.device('cpu'):
                pretrained = get_pretrained(pretrained)

        config = pretrained.get_config()
        config.update({'name' : name, 'pretrained_name' : pretrained.name, ** kwargs})

        instance = cls(max_to_keep = 1, ** config)

        name_based_partial_transfer_learning(instance.model.engine, pretrained.model.engine)

        instance.save()

        return instance
