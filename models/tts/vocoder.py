
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

from models.interfaces import BaseModel

from utils.stream_utils import text_input_stream

_stream_msg = {
    'fr'    : 'Entrez le texte à lire :',
    'en'    : 'Enter text to read :'
}
_end_msg = {
    'fr'    : 'Au revoir, à la prochaine !',
    'en'    : 'Goodbye, see you soon !'
}

class Vocoder(object):
    def __init__(self):
        self.__synthesizer  = None
        self.__vocoder      = None
    
    def set_synthesizer(self, model):
        if isinstance(model, BaseModel):
            self.__synthesizer = model
        elif isinstance(model, str):
            from models import get_pretrained

            self.__synthesizer = get_pretrained(model)
        else:
            raise ValueError("Unknown synthesizer type : {}\n  {}".format(type(model), model))

    def set_vocoder(self, model = None, model_class = None):
        if  model is None:
            from models.tts import PtWaveGlow

            self.__vocoder = PtWaveGlow()
        elif isinstance(model, str):
            from models import get_pretrained

            self.__vocoder = get_pretrained(model)
        elif isinstance(model, BaseModel):
            self.__vocoder = model
        else:
            raise ValueError("Unknown vocoder type : {}\n  {}".format(type(model), model))
    
    @property
    def synthesizer(self):
        return self.__synthesizer
    
    @property
    def vocoder(self):
        return self.__vocoder
    
    @property
    def lang(self):
        return self.synthesizer.lang
    
    @property
    def rate(self):
        return self.synthesizer.audio_rate
    
    def predict(self, sentences, ** kwargs):
        """ See `help(self.synthesizer.predict)` for complete information """
        return self.synthesizer.predict(sentences, vocoder = self.vocoder, ** kwargs)
    
    def fast_predict(self, text, filename = None, display = True, play = False, ** kwargs):
        from utils.audio import display_audio, write_audio
        
        mel     = self.synthesizer.infer(text, ** kwargs)[1]
        audio   = self.vocoder.infer(mel)[0]

        if display: display_audio(audio, rate = self.rate, play = play)
        
        if filename:
            write_audio(filename = filename, audio = audio, rate = self.rate)
        
        return audio
    
    def stream(self, stream = None, save = False, display = True, play = True, ** kwargs):
        """ Run a streaming TTS procedure where you can enter text and the model reads it ! """
        use_input_stream = stream is None
        if use_input_stream:
            stream = text_input_stream(_stream_msg.get(self.lang, _stream_msg['en']))
        
        res = self.synthesizer.stream(
            stream, vocoder = self.vocoder, play = play, save = save, display = display, ** kwargs
        )
        if use_input_stream and self.lang in _end_msg:
            self.predict(_end_msg[self.lang], play = play, save = save, display = display, ** kwargs)
        
        return res

