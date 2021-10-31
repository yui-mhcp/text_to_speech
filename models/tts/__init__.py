import os

from models.model_utils import infer_model_class

from models.tts.waveglow import WaveGlow, PtWaveGlow
from models.tts.tacotron2 import Tacotron2
from models.tts.sv2tts_tacotron2 import SV2TTSTacotron2
from models.tts.vocoder import Vocoder

_vocoder = None

_default_vocoer = {'name' : 'WaveGlow', 'model_class' : WaveGlow} if os.path.exists(os.path.join('pretrained_models', 'WaveGlow')) else None

def get_tts_model(lang = None, model = None, vocoder = _default_vocoer, vocoder_class = None,
                  ** kwargs):
    assert lang is not None or model is not None, "You must specify either the name of the model or the language of the model !"

    global _vocoder, _models
    
    # Get pretrained information from '_pretrained'
    if model is None:
        if lang not in _pretrained:
            raise ValueError("No pretrained model for this language !!\n  Supported : {}\n   Got : {}".format(lang, list(_pretrained.keys())))
        model = _pretrained[lang]
    # Convert 'model' to a dict
    if not isinstance(model, dict): model = {'nom' : model}
    model.setdefault('model_class', infer_model_class(model['nom'], _models))
    
    if isinstance(vocoder, dict):
        vocoder, vocoder_class = vocoder['name'], vocoder['model_class']
    # Create Vocoder class (if necessary)
    if _vocoder is None:
        _vocoder = Vocoder()
    
    # Set new synthesizer / vocoder
    _vocoder.set_vocoder(nom = vocoder, model_class = vocoder_class)
    _vocoder.set_synthesizer(** model)
    
    return _vocoder

def tts_stream(lang = None, model = None, vocoder = _default_vocoer, vocoder_class = None,
               ** kwargs):
    model = get_tts_model(
        lang = lang, model = model, vocoder = vocoder, vocoder_class = vocoder_class
    )
    model.stream(** kwargs)

def tts(sentences, lang = None, model = None, vocoder = _default_vocoer, vocoder_class = None,
        ** kwargs):
    """
        Perform tts and return result of Waveglow.predict(...)
        Return : list of tuple (phrase, infos) whe infos is a dict
            infos contains :
            - splitted  : the splitted original phrase
            - mels  : mel spectrogram files for each splitted part
            - audio : raw audio (if directory is None) or filename of the full audio
    """
    model = get_tts_model(
        lang = lang, model = model, vocoder = vocoder, vocoder_class = vocoder_class
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
