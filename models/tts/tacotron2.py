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
import glob
import time
import logging
import numpy as np

from utils import *
from utils.callbacks import *
from loggers import timer, time_logger
from utils.keras_utils import TensorSpec, ops
from custom_architectures import get_architecture
from models.utils import prepare_prediction_results
from utils.text import default_english_encoder, split_text, split_sentences
from models.interfaces.base_text_model import BaseTextModel
from models.interfaces.base_audio_model import BaseAudioModel
from utils.audio import write_audio, load_audio, display_audio, play_audio

logger      = logging.getLogger(__name__)

DEFAULT_MAX_MEL_LENGTH  = 1024
DEFAULT_MAX_TEXT_LENGTH = 150

class Tacotron2(BaseTextModel, BaseAudioModel):
    """
        Tacotron2 is a model introduced in this paper [https://arxiv.org/abs/1712.05884]. 
        It takes as input a text and produces a mel-spectrogram of the corresponding audio. 
        
        This class inherits from `BaseTextModel` and `BaseAudioModel` as it combines both types of data (text and audio). It means that it handles all the features available in those 2 interfaces.
    """
    
    _default_loss   = 'TacotronLoss'
    _default_metrics    = []
    
    infer_signature = BaseTextModel.text_signature
    
    prepare_input   = BaseTextModel.encode_text
    
    def __init__(self,
                 lang,
                 audio_rate     = 22050,
                 audio_format   = 'mel',
                 
                 max_input_length   = DEFAULT_MAX_TEXT_LENGTH,
                 max_output_length  = DEFAULT_MAX_MEL_LENGTH,
                 
                 ** kwargs
                ):
        self._init_text(lang = lang, ** kwargs)
        self._init_audio(audio_rate = audio_rate, audio_format = audio_format, ** kwargs)
        
        self.max_input_length   = max_input_length
        self.max_output_length  = max_output_length

        super().__init__(** kwargs)
    
    def init_train_config(self, ** kwargs):
        super().init_train_config(** kwargs)

        if self.max_train_frames > 0:
            raise NotImplementedError("max_train_frames > 0 not working yet !")
        if self.pad_to_multiple and self.max_train_frames <= 0:
            raise ValueError("If pad_to_multiple is True, max_train_frames must be > 0 !")
        
        if not self.trim_mel: self.trim_mel_method = None
        
    def build(self, tts_model = None, ** kwargs):
        if tts_model is None:
            tts_model = {
                'architecture'  : kwargs.pop('architecture', 'tacotron2'),
                'pad_token'     : self.blank_token_idx,
                'vocab_size'        : self.vocab_size,
                'n_mel_channels'    : self.n_mel_channels,
                'init_step'     : self.steps,
                ** kwargs
            }
        return super().build(tts_model = tts_model)
    
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
        return super().training_hparams(
            ** self.training_hparams_audio,
            max_input_length    = None,
            max_output_length   = None,
            
            max_train_frames = -1,
            pad_to_multiple  = False
        )
    
    @property
    def go_frame(self):
        return ops.zeros((1, self.n_mel_channels), dtype = 'float32')
    
    def __str__(self):
        return super().__str__() + self._str_text() + self._str_audio()
    
    @timer(name = 'inference')
    def infer(self,
              inputs,
              *,
              
              max_trial = 5,
              min_fpt_ratio = -1,
              max_fpt_ratio = -1,
              
              ** kwargs
             ):
        kwargs.setdefault('max_length', self.max_output_length)
        
        if not isinstance(inputs, (list, tuple)):
            if isinstance(inputs, str):
                inputs = ops.expand_dims(self.get_input(inputs), axis = 0)
            elif ops.ndim(inputs) == 1:
                inputs = ops.expand_dims(inputs, axis = 0)

        seq_len = inputs.shape[1] if not isinstance(inputs, (list, tuple)) else inputs[0].shape[1]
        
        for trial in range(max_trial):
            outputs = self.compiled_infer(inputs, ** kwargs)
            
            ratio = ops.max(outputs.lengths) / seq_len
            if (min_fpt_ratio == -1 or ratio > min_fpt_ratio) and (max_fpt_ratio == -1 or ratio < max_fpt_ratio):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('Inference succeeded with a ratio of {:.2f}'.format(ratio))
                break
            
            trial += 1
            if trial == max_trial:
                logger.warning('Inference failed too much time ! Result is probably not perfect')
            else:
                logger.info('Inference failed (lengths : {}, frape / token ratio : {:.2f}), re-executing it !'.format(outputs.lengths, ratio))
        
        return outputs.mel, outputs.lengths, outputs.stop_tokens, outputs.attention_weights
    
    def prepare_output(self, data):
        mel = self.get_audio(data)
        
        mel = ops.concatenate([self.go_frame, mel], axis = 0)

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
        if self.max_train_frames > 0: return True
        return ops.logical_and(
            ops.shape(inputs[0])[-1] <= self.max_input_length, 
            inputs[-1] <= self.max_output_length
        )
    
    def augment_input(self, inputs):
        mel_input, mel_length = inputs[-2:]
        mel_input = self.augment_audio(mel_input)
        
        return inputs[:-2] + (mel_input, mel_length)
    
    def get_dataset_config(self, * args, ** kwargs):
        kwargs.update({
            'pad_kwargs'    : {
                'padding_values'    : (
                    (self.blank_token_idx, self.pad_mel_value, 0), (self.pad_mel_value, 1.)
                )
            }
        })
        
        return super().get_dataset_config(* args, ** kwargs)
    
    def get_prediction_callbacks(self,
                                 *,

                                 use_cleaned_as_key = True,
                                 
                                 vocoder    = None,
                                 
                                 save   = None,
                                 save_mel   = None,
                                 save_plot  = False,
                                 save_audio = None,
                                 
                                 directory  = None,
                                 mel_dir    = None,
                                 plot_dir   = None,
                                 audio_dir  = None,
                                 
                                 mel_filename   = 'mel-{}.npy',
                                 plot_filename  = 'plot-{}.png',
                                 audio_filename = 'audio-{}.mp3',
                                 
                                 play   = False,
                                 display    = None,
                                 show_plot  = False,
                                 
                                 post_processing    = None,
                                 
                                 use_multithreading = False,

                                 ** kwargs
                                ):
        """
            Return a list of `utils.callbacks.Callback` instances that handle data saving/display
            
            Arguments :
                - save  : whether to save detection results
                          Set to `True` if `save_boxes` or `save_detected` is True
                - save_empty    : whether to save raw images if no object has been detected
                - save_detected : whether to save the image with detected objects
                - save_boxes    : whether to save boxes as individual images (not supported yet)
                
                - directory : root directory for saving (see below for the complete tree)
                - raw_img_dir   : where to save raw images (default `{directory}/images`)
                - detected_dir  : where to save images with detection (default `{directory}/detected`)
                - boxes_dir     : where to save individual boxes (not supported yet)
                
                - filename  : raw image file format
                - detected_filename : image with detection file format
                - boxes_filename    : individual boxes file format
                
                - display   : whether to display image with detection
                              If `None`, set to `True` if `save == False`
                - verbose   : verbosity level (cumulative, i.e., level 2 includes level 1)
                              - 1 : displays the image with detection
                              - 2 : displays the individual boxes
                              - 3 : logs the boxes position
                                 
                - post_processing   : callback function applied on the results
                                      Takes as input all kwargs returned by `self.predict`
                                      - image   : the raw original image (`ndarray / Tensor`)
                                      - boxes   : the detected objects (`dict`)
                                      * filename    : the image file (`str`)
                                      * detected    : the image with detection (`ndarray`)
                                      * output      : raw model output (`Tensor`)
                                      * frame_index : the frame index in a stream (`int`)
                                      Entries with "*" are conditionned and are not always provided
                
                - use_multithreading    : whether to multi-thread the saving callbacks
                
                - kwargs    : mainly ignored
            Return : (predicted, required_keys, callbacks)
                - predicted : the mapping `{filename : infos}` stored in `{directory}/map.json`
                - required_keys : expected keys to save (see `models.utils.should_predict`)
                - callbacks : the list of `Callback` to be applied on each prediction
        """
        if vocoder is None:     play, display, save_audio = False, False, False
        elif save_audio is None:save_audio = save is not False
        if save is None:        save = directory or vocoder is None
        if save_mel is None:    save_mel = save and vocoder is None
        if save_plot is None:   save_plot = save
        
        save = save_mel or save_plot or save_audio
        if vocoder is not None:
            if save:            save_audio = True
            elif not play:      display = True

        # get saving directory
        if directory is None: directory = self.pred_dir
        map_file = os.path.join(directory, 'map.json')
        
        predicted   = {}
        callbacks   = []
        required_keys = {'text', 'cleaned', 'mel' if vocoder is None else 'audio'}
        if save:
            predicted   = load_json(map_file, {})

            if save_mel:
                required_keys.add('mel')
                if mel_dir is None: mel_dir = os.path.join(directory, 'mels')
                callbacks.append(SpectrogramSaver(
                    key = 'mel',
                    name    = 'saving mel',
                    file_format = os.path.join(mel_dir, mel_filename),
                    use_multithreading  = use_multithreading
                ))
            
            if save_plot:
                required_keys.add('plot')
                if plot_dir: plot_dir = os.path.join(directory, 'plots')
                raise NotImplementedError()
            
            if save_audio:
                required_keys.add('audio')
                if audio_dir is None: audio_dir = os.path.join(directory, 'audios')
                callbacks.append(AudioSaver(
                    key = 'audio',
                    name    = 'saving audio',
                    file_format = os.path.join(audio_dir, audio_filename),
                    use_multithreading  = use_multithreading
                ))
        
            callbacks.append(JSonSaver(
                data    = predicted,
                filename    = map_file,
                primary_key = 'cleaned' if use_cleaned_as_key else 'text',
                use_multithreading = use_multithreading
            ))
        
        if display:
            callbacks.append(AudioDisplayer(play = play == 1))
        
        if show_plot:
            raise NotImplementedError()

        if post_processing is not None:
            callbacks.append(FunctionCallback(post_processing))
        
        return predicted, required_keys, callbacks

    @timer(name = 'text encoding')
    def split_and_encode(self, cleaned_text, max_text_length, cleaned = True):
        if isinstance(cleaned_text, list): splitted = cleaned_text
        elif isinstance(max_text_length, str):
            splitted = [p for p in cleaned_text.split(max_text_length) if p]
        elif max_text_length == -1:        splitted = [cleaned_text]
        elif max_text_length == -2:        splitted = split_sentences(cleaned_text)
        else:                              splitted = split_text(cleaned_text, max_text_length)

        if not cleaned:
            splitted    = [self.clean_text(s) for s in splitted]
        
        splitted    = [s for s in splitted if any(c.isalnum() for c in s)]
        encoded     = [self.encode_text(text, cleaned = True) for text in splitted]
        
        splitted    = [splitted[i] for i in range(len(splitted)) if len(encoded[i]) > 0]
        encoded     = [enc for enc in encoded if len(enc) > 0]
        return cleaned_text, splitted, encoded

    @timer
    def predict(self,
                texts,
                batch_size  = 1,
                max_text_length = -1,
                *,
                
                overwrite   = False,
                return_output   = True,
                
                cleaners    = {},
                expand_acronyms = True,
                use_cleaned_as_key  = False,
                
                max_length  = 10.,
                min_fpt_ratio = 2.,
                max_fpt_ratio = 10.,
                
                vocoder = None,
                silence_time    = 0.15,
                vocoder_config  = {},
                
                part_post_processing = None,
                
                predicted   = None,
                _callbacks  = None,
                required_keys   = None,
                
                ** kwargs
               ):
        ####################
        #  Initialization  #
        ####################

        now = time.time()
        with time_logger.timer('initialization'):
            join_callbacks = _callbacks is None
            if _callbacks is None:
                predicted, required_keys, _callbacks = self.get_prediction_callbacks(
                    vocoder = vocoder, use_cleaned_as_key = use_cleaned_as_key, ** kwargs
                )

            results, inputs, indexes, entries, duplicates, filtered = prepare_prediction_results(
                texts,
                predicted,
                
                rank    = 1,
                primary_key = 'text',
                expand_files    = False,
                normalize_entry = self.clean_text if use_cleaned_as_key else None,
                
                overwrite   = overwrite,
                required_keys   = required_keys,
            )
            
        ####################
        #  Inference loop  #
        ####################
        
        show_idx = apply_callbacks(results, 0, _callbacks)
        
        silence = np.zeros((int(self.audio_rate * silence_time), ), dtype = 'float32')
        
        for result_idx, data, key in zip(indexes, inputs, entries):
            cleaned, splitted, encoded = self.split_and_encode(
                key, max_text_length, cleaned = use_cleaned_as_key
            )

            synth_gen_time, vocoder_gen_time = 0., 0.

            outputs = {
                'text'  : data['text'] if isinstance(data, dict) else data,
                'parts' : splitted,
                'cleaned'   : cleaned,
                'timestamp' : now,
                
                'mel'   : [],
                'attention' : [],
            }
            if vocoder is not None:
                outputs.update({'audio' : [], 'rate' : self.audio_rate})
            
            for s in range(0, len(encoded), batch_size):
                batch = stack_batch(
                    encoded[s : s + batch_size],
                    dtype   = 'int32',
                    pad_value   = self.blank_token_idx,
                    maybe_pad   = True
                )
                batch = ops.convert_to_tensor(batch, 'int32')

                start_synth_infer_time = time.time()

                mels, lengths, gates, attentions = self.infer(
                    batch,
                    max_length  = max_length,
                    min_fpt_ratio   = min_fpt_ratio,
                    max_fpt_ratio   = max_fpt_ratio,
                    ** kwargs
                )

                synth_gen_time   += time.time() - start_synth_infer_time
                
                for idx in range(len(batch)):
                    mel, attn = mels[idx, : lengths[idx]], attentions[idx, : lengths[idx]]
                    outputs['mel'].append(mel)
                    outputs['attention'].append(attn)

                    audio = None
                    if vocoder is not None:
                        start_vocoder_time = time.time()

                        audio = vocoder(mel, ** {** kwargs, ** vocoder_config})
                        if len(audio.shape) == 2: audio = audio[0]
                        outputs['audio'].append(ops.convert_to_numpy(audio))

                        vocoder_gen_time += time.time() - start_vocoder_time

                    if part_post_processing is not None:
                        part_post_processing(
                            text = splitted[s + idx], mel = mel, attention = attn, audio = audio
                        )

            if vocoder is not None:
                if len(outputs['audio']) > 0:
                    outputs.update({
                        'audio' : np.concatenate(outputs['audio'], 0) if len(outputs['audio']) > 1 else outputs['audio'][0],
                        'time'  : sum(len(a) for a in outputs['audio']) / self.audio_rate
                    })
                    
                    logger.info('{} generated in {} ({} generated / sec) : {} synthesizer + {} vocoder'.format(
                        time_to_string(outputs['time']),
                        time_to_string(synth_gen_time + vocoder_gen_time),
                        time_to_string(outputs['time'] / (synth_gen_time + vocoder_gen_time)),
                        time_to_string(synth_gen_time),
                        time_to_string(vocoder_gen_time)
                    ))
                else:
                    outputs.update({
                        'audio' : silence,
                        'time'  : silence_time
                    })
                

            if duplicates.get(key, ()):
                for duplicate_idx in duplicates[key]:
                    results[duplicate_idx] = (predicted.get(key, {}), outputs)
            else:
                results[result_idx] = (predicted.get(key, {}), outputs)

            show_idx = apply_callbacks(results, show_idx, _callbacks)

        if join_callbacks:
            for callback in _callbacks: callback.join()
        
        return [
            {** out, ** stored} if return_output else stored
            for stored, out in results
        ]

    def get_config(self):
        config = super().get_config()
        config.update({
            ** self.get_config_text(),
            ** self.get_config_audio(),
            
            'max_input_length'  : self.max_input_length,
            'max_output_length' : self.max_output_length
        })
        
        return config
    
    @classmethod
    def from_nvidia_pretrained(cls, name = 'pretrained_tacotron2', ** kwargs):
        kwargs.update({'audio_format' : 'mel', 'audio_rate' : 22050})
        kwargs.setdefault('lang', 'en')
        kwargs.setdefault('text_encoder', default_english_encoder())
        
        with keras.device('cpu') as device:
            instance = cls(
                name = name, max_to_keep = 1, pretrained_name = 'pytorch_nvidia_tacotron2', ** kwargs
            )
        
        nvidia_model = get_architecture('nvidia_tacotron', to_gpu = False)
        
        pt_convert_model_weights(nvidia_model, instance.tts_model)
        
        instance.save()
        
        return instance
