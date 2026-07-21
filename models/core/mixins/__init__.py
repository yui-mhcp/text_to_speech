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

import importlib

# `core` mixins are the always-on capabilities folded into every `BaseModel` (imported eagerly by
# `base_model`). `modality` mixins are opt-in : each leaf model mixes in only the ones it needs, and
# each pulls a heavy modality package at import time (`utils.{text,audio,image}`). They are therefore
# exposed **lazily** (PEP 562 `__getattr__`) so that importing a text model never drags in `utils.audio`.
_CORE_MIXINS = {
    'ModelCheckpointMixin' : 'checkpoint_mixin',
    'TrainableModelMixin'  : 'training_mixin',
    'ModelProcessingMixin' : 'processing_mixin',
}
_MODALITY_MIXINS = {
    'TextModelMixin'           : 'text_mixin',
    'AudioModelMixin'          : 'audio_mixin',
    'ImageModelMixin'          : 'image_mixin',
    'ClassificationModelMixin' : 'classification_mixin',
}
_MIXINS = {** _CORE_MIXINS, ** _MODALITY_MIXINS}

def __getattr__(name):
    """ Lazily imports the module owning `name` on first access (keeps modality utils lazy). """
    if name in _MIXINS:
        module = importlib.import_module('.' + _MIXINS[name], __name__)
        return getattr(module, name)
    raise AttributeError('module {!r} has no attribute {!r}'.format(__name__, name))

__all__ = list(_MIXINS)
