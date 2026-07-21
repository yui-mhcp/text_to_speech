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
import logging

__all__ = ['pop_legacy_model_name', 'migrate_legacy_history_file', 'restore_legacy_model']

logger = logging.getLogger(__name__)

def pop_legacy_model_name(kwargs, default):
    """ Backward-compat : early configs stored the model name under the French key `nom`. """
    return kwargs.pop('nom', default)

def migrate_legacy_history_file(history_file):
    """ Renames a legacy French `historique.json` to `history.json` when needed. """
    legacy = history_file.replace('history', 'historique')
    if os.path.exists(legacy) and not os.path.exists(history_file):
        os.rename(legacy, history_file)

def restore_legacy_model(model, config, compile = False):
    """
        Restores a model saved with the legacy (multi-model / keras 2) `config_models.json` format.

        `model` is the `BaseModel` instance being restored. This is kept out of `BaseModel` so the
        class is not bloated with backward-compatibility code that only old checkpoints trigger.
    """
    import keras

    from architectures import get_custom_objects

    _load_weights   = True

    if len(config['models']) > 1:
        raise NotImplementedError('{} has more than 1 model ({}) which is not supported anymore'.format(
            model.name, tuple(config['models'].keys())
        ))

    key, model_config = list(config['models'].items())[0]

    filename = model.checkpoint_manager.loaded_checkpoint
    if not filename: filename = os.path.join(model.save_dir, '{}.keras'.format(key))

    if filename.endswith('.keras') and os.path.exists(filename):
        logger.info('Loading `{}` from {}'.format(key, filename))
        model.model = keras.models.load_model(filename)
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

        model.model = keras.saving.deserialize_keras_object(
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
            model.model = deserialize_keras2_model(model_config)
        else:
            model.model = keras.models.model_from_json(json_config)

    logger.info('`model` successfully restored !')

    if config['losses']:
        if '{}_optimizer'.format(key) in config.get('optimizers', {}):
            for k in ('optimizers', 'losses', 'metrics'):
                if len(config[k]) > 0: config[k] = list(config[k].values())[0]
            config['losses'].get('loss_config', {}).pop('reduction', None)

        model.compile(** config['losses'], ** config['optimizers'], ** config['metrics'])

    if _load_weights:   model.checkpoint_manager.load()
