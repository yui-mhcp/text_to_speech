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
import time
import logging
import numpy as np

from loggers import Timer, timer
from utils.text import split_sentences, split_text
from utils import Stream, time_to_string, load_json
from utils.callbacks import AudioSaver, AudioPlayer, FunctionCallback, QueueCallback, JSONSaver, SpectrogramSaver, apply_callbacks
from utils.keras import TensorSpec, ops
from ..interfaces.base_text_model import BaseTextModel
from ..interfaces.base_audio_model import BaseAudioModel

logger = logging.getLogger(__name__)

DEFAULT_MAX_MEL_LENGTH  = 1024
DEFAULT_MAX_TEXT_LENGTH = 150

class Tacotron2(BaseTextModel, BaseAudioModel):
    """
        Tacotron2 is a model introduced in this paper [https://arxiv.org/abs/1712.05884]. 
        It takes as input a text and produces a mel-spectrogram of the corresponding audio. 
        
        This class inherits from `BaseTextModel` and `BaseAudioModel` as it combines both types of data (text and audio). It means that it handles all the features available in those 2 interfaces.
    """
    
    _default_loss   = 'TacotronLoss'
    
    infer_signature = BaseTextModel.text_signature
    prepare_input   = BaseTextModel.encode_text
    
    def __init__(self,
                 lang,
                 
                 *,
                 
                 mel_fn = 'TacotronSTFT',
                 audio_format   = 'mel',
                 
                 max_input_length   = DEFAULT_MAX_TEXT_LENGTH,
                 max_output_length  = DEFAULT_MAX_MEL_LENGTH,
                 
                 ** kwargs
                ):
        self._init_text(lang = lang, ** kwargs)
        self._init_audio(audio_format = audio_format, mel_fn = mel_fn, ** kwargs)
        
        self.max_input_length   = max_input_length
        self.max_output_length  = max_output_length

        super().__init__(** kwargs)
        
    def build(self, *, model = None, tts_model = None, ** kwargs):
        if tts_model is not None: model = tts_model
        elif model is None:
            model = {
                'architecture'  : kwargs.pop('architecture', 'tacotron2'),
                'pad_token'     : self.blank_token_idx,
                'vocab_size'        : self.vocab_size,
                'n_mel_channels'    : self.n_mel_channels,
                ** kwargs
            }
        
        return super().build(model = model)
    
    @property
    def input_signature(self):
        return (
            self.text_signature,
            self.audio_signature,
            TensorSpec(shape = (None, ), dtype = 'int32')
        )
    
    @property
    def output_signature(self):
        return (
            self.audio_signature, TensorSpec(shape = (None, None), dtype = 'float32')
        )
        
    @property
    def training_hparams(self):
        return {
            ** super().training_hparams,
            ** self.training_hparams_audio,
            'max_input_length'  : None,
            'max_output_length' : None
        }
    
    def __str__(self):
        return super().__str__() + self._str_text() + self._str_audio()
    
    @timer(name = 'inference')
    def infer(self,
              text,
              *,

              embeddings    = None,
              
              callbacks = None,
              predicted = None,
              overwrite = False,
              return_output = True,
              
              max_length  = 10.,
              max_text_length = -1,
                    
              max_trial   = 5,
              min_fpt_ratio   = 2.,
              max_fpt_ratio   = 10.,

              vocoder = None,
              silence_time    = 0.15,
              vocoder_config  = {},

              ** kwargs
             ):
        if predicted and not overwrite and text in predicted:
            if callbacks: apply_callbacks(callbacks, predicted[text], {}, save = False)
            return predicted[text]
        
        with Timer('processing'):
            if max_text_length == -1:
                splitted    = [self.clean_text(text)]
            elif max_text_length == -2:
                splitted    = [self.clean_text(sent) for sent in split_sentences(text)]
            else:
                splitted    = [self.clean_text(sent) for sent in split_text(text, max_text_length)]
            splitted    = [s for s in splitted if any(c.isalnum() for c in s)]
            cleaned     = '\n\n'.join(splitted)

            encoded     = [self.encode_text(text, cleaned = True) for text in splitted]

            splitted    = [splitted[i] for i in range(len(splitted)) if len(encoded[i]) > 0]
            encoded     = [enc for enc in encoded if len(enc) > 0]

        synth_gen_time, vocoder_gen_time = 0., 0.

        mels, attention_weights, audios = [], [], []
        for inp in encoded:
            start_synth_infer_time = time.time()

            length  = len(inp)
            success = False
            inputs  = inp[None] if embeddings is None else (inp[None], embeddings[None])
            for trial in range(max_trial):
                with Timer('inference {}'.format(self.__class__.__name__)):
                    outputs = self.compiled_infer(
                        inputs, max_length = max_length, ** kwargs
                    )
                ratio = outputs.lengths[0] / length
                if ratio > min_fpt_ratio and ratio < max_fpt_ratio:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug('Inference succeeded with a ratio of {:.2f}'.format(ratio))
                    success = True
                    break

                logger.info('Inference failed (lengths : {}, frape/token ratio : {:.2f}) !'.format(
                    outputs.lengths, ratio
                ))

            synth_gen_time   += time.time() - start_synth_infer_time

            if not success:
                logger.warning('Inference failed too much time ! Result is probably not perfect')

            mels.append(outputs.mel[0, : outputs.lengths[0]])
            attention_weights.append(outputs.attention_weights[0, : outputs.lengths[0]])

            if vocoder is not None:
                start_vocoder_time = time.time()

                audio = vocoder(mels[-1], ** {** kwargs, ** vocoder_config})
                if len(audio.shape) == 2: audio = audio[0]
                audios.append(ops.convert_to_numpy(audio))

                vocoder_gen_time += time.time() - start_vocoder_time

        audio_infos = {}
        if vocoder is not None:
            if len(audios) > 0:
                audios = audios[0] if len(audios) == 1 else np.concatenate(audios, axis = 0)
                audio_infos = {
                    'audio' : audios,
                    'rate'  : self.rate,
                    'time'  : len(audios) / self.rate
                }

                logger.info('{} generated in {} ({} generated / sec) : {} synthesizer + {} vocoder'.format(
                    time_to_string(audio_infos['time']),
                    time_to_string(synth_gen_time + vocoder_gen_time),
                    time_to_string(audio_infos['time'] / (synth_gen_time + vocoder_gen_time)),
                    time_to_string(synth_gen_time),
                    time_to_string(vocoder_gen_time)
                ))
            else:
                audio_infos = {
                    'audio' : np.zeros((int(silence_time * self.rate), ), dtype = 'float32'),
                    'rate'  : self.rate,
                    'time'  : silence_time
                }

        output = {
            'text'  : text,
            'cleaned'   : cleaned,
            'splitted'  : splitted,
            
            'mel'   : mels,
            'attention' : attention_weights,
            ** audio_infos
        }
        
        entry = None
        if callbacks:
            if text not in predicted:
                predicted[text] = {
                    k : v for k, v in output.items() if k not in ('mel', 'attention', 'audio')
                }

            entry = apply_callbacks(callbacks, predicted[text], output, save = True)
        
        if return_output:
            return output
        elif vocoder is None or 'audio' in predicted.get(text, {}):
            return predicted.get(text, {})
        else:
            return {k : v for k, v in output.items() if k not in ('mel', 'attention')}

    def prepare_output(self, data):
        mel = self.get_audio(data)
        mel = ops.pad(mel, [(1, 0), (0, 0)])

        gate = ops.concatenate([
            ops.zeros((ops.shape(mel)[0] - 1, ), dtype = 'float32'),
            ops.ones((1,), dtype = 'float32')
        ], axis = 0)
        
        return mel, gate
    
    def prepare_data(self, data):
        encoded_text = self.prepare_input(data)
        
        mel, gate = self.prepare_output(data)
        
        return (encoded_text, mel[:-1], len(mel) - 1), (mel[1:], gate[1:])
        
    def filter_data(self, inputs, outputs):
        return ops.logical_and(
            ops.shape(inputs[0])[-1] <= self.max_input_length,
            inputs[-1] <= self.max_output_length
        )
    
    def get_dataset_config(self, * args, ** kwargs):
        kwargs['pad_kwargs']    = {
            'padding_values'    : (
                (self.blank_token_idx, self.pad_mel_value, 0), (self.pad_mel_value, 1.)
            )
        }
        
        return super().get_dataset_config(* args, ** kwargs)
    
    def get_inference_callbacks(self,
                                *,

                                vocoder    = None,

                                save   = None,
                                save_mel   = None,
                                save_audio = None,

                                directory  = None,
                                mel_dir    = None,
                                audio_dir  = None,

                                mel_filename   = 'mel-{}.npy',
                                audio_filename = 'audio-{}.mp3',

                                play   = False,
                                display    = None,

                                post_processing = None,

                                save_in_parallel    = False,

                                ** kwargs
                               ):
        """ Return a list of `utils.callbacks.Callback` instances that handle data saving/display """
        if vocoder is None:     play, display, save_audio = False, False, False
        elif save_audio is None:save_audio = save is not False
        if save is None:        save = directory or vocoder is None
        if save_mel is None:    save_mel = save and vocoder is None
        
        save = save_mel or save_audio
        if vocoder is not None:
            if save:                save_audio = True
            elif display is None:   display = not play

        predicted, callbacks = {}, []
        if save:
            # get saving directory
            if directory is None: directory = self.pred_dir
            map_file = os.path.join(directory, 'map.json')

            predicted   = load_json(map_file, {})

            if save_mel:
                if mel_dir is None: mel_dir = os.path.join(directory, 'mels')
                callbacks.append(SpectrogramSaver(
                    file_format     = os.path.join(mel_dir, mel_filename),
                    save_in_parallel    = save_in_parallel
                ))
            
            if save_audio:
                if audio_dir is None: audio_dir = os.path.join(directory, 'audios')
                callbacks.append(AudioSaver(
                    file_format     = os.path.join(audio_dir, audio_filename),
                    save_in_parallel    = save_in_parallel
                ))
        
            callbacks.append(JSONSaver(
                data    = predicted,
                filename    = map_file,
                primary_key = 'text',
                save_in_parallel = save_in_parallel
            ))
        
        if display or play:
            callbacks.append(AudioPlayer(display = display, play = play))

        if post_processing is not None:
            if not isinstance(post_processing, list): post_processing = [post_processing]
            for fn in post_processing:
                if callable(fn):
                    callbacks.append(FunctionCallback(fn))
                elif hasattr(fn, 'put'):
                    callbacks.append(QueueCallback(fn))
        
        return predicted, callbacks

    def precompile_for_stream(self, ** kwargs):
        for m in (64, 128):
            self.infer('hello {}'.format(m), max_trial = 1, padding_multiple = m, ** kwargs)

    @timer
    def predict(self, inputs, ** kwargs):
        if isinstance(inputs, (str, dict)): inputs = [inputs]
        
        return super().predict(inputs, ** kwargs)
    
    def stream(self, stream, *, vocoder, ** kwargs):
        self.precompile_for_stream(** kwargs)
        
        return super().stream(stream, vocoder = vocoder, ** kwargs)
    
    def get_config(self):
        return {
            ** super().get_config(),
            ** self.get_config_text(),
            ** self.get_config_audio(),
            
            'max_input_length'  : self.max_input_length,
            'max_output_length' : self.max_output_length
        }
    
