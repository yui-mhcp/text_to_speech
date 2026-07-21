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

from utils import Stream, setup_environment
from utils.generic_utils import import_submodules

from .waveglow import WaveGlow
from ..core import BaseModel
from ..core.utils import get_model_config

_models = {}
for module in import_submodules(__package__):
    _models.update({
        k : v for k, v in vars(module).items() if isinstance(v, type) and issubclass(v, BaseModel)
    })
globals().update(_models)

_default_vocoder = 'WaveGlow'

def get_model_lang(model):
    """ Returns the language of a model """
    return get_model_config(model).get('lang', None)

def set_pretrained_model(model, lang):
    _pretrained[lang] = model

def get_pretrained_model(lang):
    return _pretrained.get(lang, None)

def get_models(model = None, lang = None, vocoder = None):
    assert model or lang
    
    if not model: model = get_pretrained_model(lang)
    
    if not isinstance(model, BaseModel):
        from .. import get_pretrained
        model = get_pretrained(model)
    
    if not isinstance(vocoder, BaseModel):
        if vocoder is None: vocoder = _default_vocoder
        
        if isinstance(vocoder, dict):
            vocoder = WaveGlow(** vocoder)
        else:
            from .. import get_pretrained
            vocoder = get_pretrained(vocoder)
    
    return model, vocoder


def tts(text, *, model = None, lang = None, vocoder = None, add_model_name = False, ** kwargs):
    """
        Performs TTS on `text` (single `str` / `dict`, or a list of them) with the model resolved
        by `get_models` (explicit `model`, or the default one for `lang`)

        Return : the result of `model.predict(...)` — for each input, a `dict` :
            - text      : the raw input text
            - cleaned   : the cleaned text
            - splitted  : the synthesized text segments (aligned with `mel` / `attention`)
            - mel       : mel-spectrogram (list, one per segment)
            - attention : attention weights (list, one per segment)
            - audio / rate / time   : the full raw audio (only when a `vocoder` is used)
        When saving is enabled and `return_output` is False, the lightweight `map.json` entry is
        returned instead (heavy keys replaced by the saved filenames)

        For a single `str` / `dict` input, the single result is returned directly (not a list)
    """
    model, vocoder = get_models(model = model, lang = lang, vocoder = vocoder)
    
    if add_model_name and 'directory' in kwargs:
        kwargs['directory'] = os.path.join(kwargs['directory'], model.name)
    
    res = model.predict(text, vocoder = vocoder, ** kwargs)
    return res[0] if isinstance(text, (str, dict)) else res

def stream(stream, *, preload = None, model = None, lang = None, vocoder = None, ** kwargs):
    """
        Runs TTS on a stream of items (see `utils.threading.Stream`), in one of 3 modes :
            1) `model` / `lang` is provided : the resolved synthesizer handles the whole stream
               (`model.stream(...)`), items are texts
            2) `preload` : the given model names (or all `_pretrained` if True) are pre-compiled,
               then mode 3) applies
            3) Fallback : each item is forwarded to `tts(...)`, and **must** be a `dict` providing
               `model` or `lang` (e.g., a `Queue` of dicts, unpacked as kwargs via
               `Stream(dict_as_kwargs = True)`). This enables switching the model dynamically on a
               per-item basis, which stays cheap since all `BaseModel` instances are cached
               singletons (see `get_pretrained`)
    """
    setup_environment(** kwargs)
    
    if model or lang:
        model, vocoder = get_models(model = model, lang = lang, vocoder = vocoder)
        return model.stream(stream, vocoder = vocoder, ** kwargs)
    
    if preload:
        if preload is True: preload = _pretrained.values()
        elif not isinstance(preload, (list, tuple)): preload = [preload]
        for name in preload:
            synthesizer, _ = get_models(model = name, vocoder = vocoder)
            synthesizer.precompile_for_stream(** kwargs)
        
    elif vocoder:
        if isinstance(vocoder, dict):
            vocoder = WaveGlow(** vocoder)
        else:
            from .. import get_pretrained
            vocoder = get_pretrained(vocoder)
    
    return list(Stream(tts, stream, vocoder = vocoder, ** kwargs))

_pretrained = {
    'en'    : 'pretrained_tacotron2',
    'fr'    : 'sv2tts_siwis_v3'
}