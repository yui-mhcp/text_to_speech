import os

from utils import load_json
from models.model_utils import get_model_dir, get_model_config, is_model_name

from models.tts.waveglow import WaveGlow, PtWaveGlow
from models.tts.tacotron2 import Tacotron2
from models.tts.sv2tts_tacotron2 import SV2TTSTacotron2
from models.tts.vocoder import Vocoder

_vocoder = None

_default_vocoer = 'WaveGlow' if is_model_name('WaveGlow') else None

def get_model_lang(model):
    return get_model_config(model).get('lang', None)
    
def get_model_name(lang):
    if lang not in _pretrained:
        raise ValueError('Unknown language for pretrained TTS model\n  Accepted : {}\n  Got : {}'.format(tuple(_pretrained.keys()), lang))
    return _pretrained[lang]

def get_tts_model(lang = None, model = None, vocoder = _default_vocoer, ** kwargs):
    global _vocoder
    
    # Get pretrained information from '_pretrained'
    if model is None:
        assert lang is not None, "You must specify either the model or the language !"
        model = get_model_name(lang = lang)
    # Create Vocoder class (if necessary)
    if _vocoder is None:
        _vocoder = Vocoder()
    
    # Set new synthesizer / vocoder
    _vocoder.set_vocoder(nom = vocoder)
    _vocoder.set_synthesizer(nom = model)
    
    return _vocoder

def get_audio_dir(lang = None, model = None, directory = None, add_model_name = True):
    if directory is None:
        if model is None: model = get_model_name(lang)
        directory = get_model_dir(model, 'output')
    elif add_model_name and (lang is not None or model is not None):
        if model is None: model = get_model_name(lang)
        directory = os.path.join(directory, model)
    
    return directory
    
def get_audio_file(text, * args, ** kwargs):
    directory = get_audio_dir(* args, ** kwargs)
    
    return load_json(os.path.join(directory, 'map.json')).get(text, {}).get('audio', None)

def tts_stream(lang = None, model = None, vocoder = _default_vocoer, ** kwargs):
    model = get_tts_model(
        lang = lang, model = model, vocoder = vocoder
    )
    model.stream(** kwargs)

def tts(sentences, lang = None, model = None, vocoder = _default_vocoer, ** kwargs):
    """
        Perform tts and return result of Waveglow.predict(...)
        Return : list of tuple (phrase, infos) whe infos is a dict
            infos contains :
            - splitted  : the splitted original phrase
            - mels  : mel spectrogram files for each splitted part
            - audio : raw audio (if directory is None) or filename of the full audio
    """
    model = get_tts_model(
        lang = lang, model = model, vocoder = vocoder
    )
    return model.predict(sentences, ** kwargs)


_models = {
    'SV2TTSTacotron2'   : SV2TTSTacotron2,
    'Tacotron2'     : Tacotron2,
    'PtWaveGlow'    : PtWaveGlow,
    'WaveGlow'      : WaveGlow
}

_pretrained = {
    'en'    : 'pretrained_tacotron2',
    'fr'    : 'tacotron2_siwis'
}
