
# Copyright (C) 2022 yui-mhcp project's author. All rights reserved.
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
import shutil
import logging
import numpy as np

from utils import load_json, create_stream
from utils.text import parse_document, get_encoder
from utils.audio import read_audio, write_audio
from models.model_utils import get_model_dir, get_model_config, is_model_name

from models.tts.waveglow import WaveGlow, PtWaveGlow
from models.tts.tacotron2 import Tacotron2
from models.tts.sv2tts_tacotron2 import SV2TTSTacotron2
from models.tts.vocoder import Vocoder

logger  = logging.getLogger(__name__)

_vocoder        = None
_text_encoders  = {}

_default_vocoder = 'WaveGlow' if is_model_name('WaveGlow') else None


def clean_text(text, model = None, lang = None):
    """ Cleans the `text` given a model or language """
    if model is None: model = get_model_name(lang)
    
    if hasattr(model, 'clean_text'): return model.clean_text(text)
    
    global _text_encoders
    if model not in _text_encoders:
        logger.debug('Loading text encoder for model {}'.format(model))
        _text_encoders[model] = get_encoder(text_encoder = model)
    
    return _text_encoders[model].clean_text(text)

def load_tts_models(model = None, tf_compile = False, ** kwargs):
    """ Loads all default models (in `_pretrained`) """
    from models import get_pretrained
    
    global _pretrained, _default_vocoder
    
    if model is None:                   model = list(_pretrained.values())
    elif not isinstance(model, list):   model = [model]
    
    vocoder = get_pretrained(_default_vocoder) if isinstance(_default_vocoder, str) else None
    
    kwargs.update({
        'max_length' : 10, 'max_trial' : 1, 'save' : False, 'play' : False, 'display' : False
    })
    for name in model:
        synthesizer = get_pretrained(name)
        if tf_compile:
            logger.debug('Call `predict` to compile the model...')
            for txt in ('A', 'AB'):
                synthesizer.infer(txt, vocoder = vocoder, ** kwargs)[1]

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
        elif not isinstance(model, str): model = model.nom
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

_models = {
    'SV2TTSTacotron2'   : SV2TTSTacotron2,
    'Tacotron2'     : Tacotron2,
    'PtWaveGlow'    : PtWaveGlow,
    'WaveGlow'      : WaveGlow
}

_pretrained = {
    'en'    : 'pretrained_tacotron2',
    'fr'    : 'sv2tts_siwis_v2'
}
