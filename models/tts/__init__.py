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
import importlib

from utils import Stream, setup_environment

from .waveglow import WaveGlow
from ..interfaces import BaseModel
from ..utils import get_model_dir, get_model_config, is_model_name

for module in os.listdir(__package__.replace('.', os.path.sep)):
    if module.startswith(('.', '_')) or '_old' in module: continue
    module = importlib.import_module(__package__ + '.' + module[:-3])
    
    globals().update({
        k : v for k, v in vars(module).items() if isinstance(v, type) and issubclass(v, BaseModel)
    })

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
        Perform TTS and return result of Tacotron2.predict(...)
        Return : list of tuple (sentence, infos) whe infos is a dict
            `infos` contains :
            - splitted  : the splitted original phrase
            - mels      : mel spectrogram files for each splitted part
            - audios    : audio files for each part
            - audio     : raw audio (if directory is None) or filename of the full audio
    """
    model, vocoder = get_models(model = model, lang = lang, vocoder = vocoder)
    
    if add_model_name and 'directory' in kwargs:
        kwargs['directory'] = os.path.join(kwargs['directory'], model.name)
    
    res = model.predict(text, vocoder = vocoder, ** kwargs)
    return res[0] if isinstance(text, (str, dict)) else res

def stream(stream, *, preload = None, model = None, lang = None, vocoder = None, ** kwargs):
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