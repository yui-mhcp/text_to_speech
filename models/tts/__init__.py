# Copyright (C) 2022-now yui-mhcp project author. All rights reserved.
# Licenced under a modified Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import logging
import numpy as np

from loggers import timer
from utils.keras_utils import ops
from utils import load_json, create_stream, limit_gpu_memory, set_memory_growth
from utils.text import parse_document, get_encoder
from utils.audio import read_audio, write_audio
from models.utils import get_model_dir, get_model_config, is_model_name

from .waveglow import WaveGlow, PtWaveGlow
from .tacotron2 import Tacotron2
from .sv2tts_tacotron2 import SV2TTSTacotron2
from .vocoder import Vocoder

logger  = logging.getLogger(__name__)

_vocoder        = None
_text_encoders  = {}

_default_vocoder = 'WaveGlow' if is_model_name('WaveGlow') else None

_compiled = set()

def clean_text(text, model = None, lang = None):
    """ Cleans the `text` given a model or language """
    if model is None: model = get_model_name(lang)
    
    if hasattr(model, 'clean_text'): return model.clean_text(text)
    
    global _text_encoders
    if model not in _text_encoders:
        logger.debug('Loading text encoder for model {}'.format(model))
        _text_encoders[model] = get_encoder(text_encoder = model)
    
    return _text_encoders[model].clean_text(text)

@timer
def compile_vocoder(vocoder, win_len = -1, ** kwargs):
    if vocoder.name in _compiled: return
    _compiled.add(vocoder.name)
    
    if 'vocoder_config' in kwargs: kwargs = {** kwargs, ** kwargs.pop('vocoder_config')}
    if win_len > 0:
        if isinstance(win_len, float):
            for i in range(1, 3):
                if win_len * i > kwargs.get('max_win_len', float('inf')): break
                vocoder.infer(ops.zeros((1, int(win_len * i), vocoder.n_mel_channels)), ** kwargs)
        else:
            vocoder.infer(ops.zeros((1, win_len, vocoder.n_mel_channels)), ** kwargs)

@timer
def load_tts_models(model = None, compile = False, ** kwargs):
    """ Loads all default models (in `_pretrained`) """
    from models import get_pretrained
    
    global _pretrained, _default_vocoder
    
    if model is None:                   model = list(_pretrained.values())
    elif not isinstance(model, list):   model = [model]
    
    if compile and isinstance(_default_vocoder, str):
        compile_vocoder(get_pretrained(_default_vocoder), ** kwargs)

    kwargs.update({
        'max_length' : 10, 'max_trial' : 1, 'save' : False, 'play' : False, 'display' : False
    })
    for name in model:
        synthesizer = get_pretrained(name)
        if compile and name not in _compiled:
            _compiled.add(name)
            logger.debug('Call `{}.predict` to compile the model...'.format(name))
            for txt in ('A', 'AB'): synthesizer.infer(txt, ** kwargs)
    

def get_model_lang(model):
    """ Returns the language of a model """
    return get_model_config(model).get('lang', None)

def get_model_name(lang):
    """ Returns the model's name associated to a given language (in `_pretrained`) """
    global _pretrained
    if lang not in _pretrained:
        raise ValueError('Unknown language for pretrained TTS model\n  Accepted : {}\n  Got : {}'.format(
            tuple(_pretrained.keys()), lang
        ))
    return _pretrained[lang]

def get_audio_dir(model = None, lang = None, directory = None, add_model_name = True):
    """ Returns the directory used to save audios for a given model """
    if directory is None:
        if model is None: model = get_model_name(lang)
        directory = get_model_dir(model, 'output')
    elif add_model_name and (lang is not None or model is not None):
        if model is None: model = get_model_name(lang)
        elif not isinstance(model, str): model = model.name
        directory = os.path.join(directory, model)
    
    return directory

def get_audio_file(text, model = None, lang = None, should_clean = True, ** kwargs):
    directory = get_audio_dir(model = model, lang = lang, ** kwargs)
    
    if (model or lang) and should_clean: text = clean_text(text, model = model, lang = lang)
    
    return load_json(os.path.join(directory, 'map.json')).get(text, {}).get('audio', None)

def get_tts_model(model = None, lang = None, vocoder = None, ** kwargs):
    global _vocoder, _default_vocoder
    
    if vocoder is None: vocoder = _default_vocoder

    # Get pretrained information from '_pretrained'
    if model is None:
        assert lang is not None, "You must specify either the model, either the language !"
        model = get_model_name(lang = lang)
    
    # Create Vocoder class (if necessary)
    if _vocoder is None:
        _vocoder = Vocoder()
    # Set new synthesizer / vocoder
    _vocoder.set_vocoder(vocoder)
    _vocoder.set_synthesizer(model)
    
    return _vocoder

def tts_document(filename,
                 output_dir = None,
                 save_page_audios   = True,
                 page_audio_format  = 'page_{}.mp3',
                 save_mel   = False,
                 ** kwargs
                ):
    if output_dir is None: output_dir = os.path.splitext(filename)[0] + '_audios'
    os.makedirs(output_dir, exist_ok = True)
    
    parsed  = parse_document(filename, save_images = False)

    flattened = []
    for _, paragraphs in parsed.items():
        flattened.extend([p['text'] for p in paragraphs if 'text' in p])

    result  = tts(flattened, directory = output_dir, save_mel = save_mel, ** kwargs)
    
    text_to_audio = {}
    for text, infos in result: text_to_audio.setdefault(text, infos)
    
    for page_nb, paragraphs in parsed.items():
        for para in paragraphs:
            if 'text' not in para or not para['text'].strip() or para['text'] not in text_to_audio:
                continue
            
            para.update(text_to_audio[para['text']])
        
        if save_page_audios:
            page_audios = [para['audio'] for para in paragraphs if 'audio' in para]
            audio = [
                read_audio(audio) for audio in page_audios
            ]
            rate    = audio[0][0]
            audio   = np.concatenate([a[1] for a in audio])
            filename    = os.path.join(output_dir, page_audio_format.format(page_nb))
            write_audio(audio = audio, filename = filename, rate = rate)
    
    return parsed
    
def tts(text, model = None, lang = None, vocoder = _default_vocoder, ** kwargs):
    """
        Perform TTS and return result of Tacotron2.predict(...)
        Return : list of tuple (sentence, infos) whe infos is a dict
            `infos` contains :
            - splitted  : the splitted original phrase
            - mels      : mel spectrogram files for each splitted part
            - audios    : audio files for each part
            - audio     : raw audio (if directory is None) or filename of the full audio
    """
    model = get_tts_model(
        lang = lang, model = model, vocoder = vocoder
    )
    return model.predict(text, ** kwargs)

def fast_tts(text, model = None, lang = None, vocoder = _default_vocoder, ** kwargs):
    """ Perform TTS and return result of `Vocoder.fast_predict(...)`, i.e. the raw audio """
    model = get_tts_model(
        lang = lang, model = model, vocoder = vocoder
    )
    return model.fast_predict(text, ** kwargs)

def tts_stream(stream = None, save = False, display = True, play = True, ** kwargs):
    if 'gpu_memory' in kwargs:  limit_gpu_memory(kwargs.pop('gpu_memory'))
    if 'gpu_growth' in kwargs:  set_memory_growth(kwargs.pop('gpu_growth'))
    
    if 'model' in kwargs: load_tts_models(compile = True, ** kwargs)
    if stream is None:
        model = get_tts_model(
            lang    = kwargs.pop('lang', None),
            model   = kwargs.pop('model', None),
            vocoder = kwargs.pop('vocoder', _default_vocoder)
        )
        return model.stream(save = save, display = display, play = play, ** kwargs)
        
    return create_stream(
        tts, stream, logger = logger, save = save, display = display, play = play, ** kwargs
    )

_pretrained = {
    'en'    : 'pretrained_tacotron2',
    'fr'    : 'sv2tts_siwis_v3'
}
