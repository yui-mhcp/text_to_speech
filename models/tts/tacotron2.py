
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
import time
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm

from loggers import timer
from custom_architectures import get_architecture
from models.interfaces.base_text_model import BaseTextModel
from models.interfaces.base_audio_model import BaseAudioModel
from models.weights_converter import pt_convert_model_weights
from utils import load_json, dump_json, plot_spectrogram, pad_batch
from utils.audio import write_audio
from utils.text import default_english_encoder, split_text
from utils.thread_utils import Producer, Consumer, Pipeline, ThreadedDict

logger      = logging.getLogger(__name__)
time_logger = logging.getLogger('timer')

DEFAULT_MAX_MEL_LENGTH  = 1024
DEFAULT_MAX_TEXT_LENGTH = 150

class Tacotron2(BaseTextModel, BaseAudioModel):
    """
        Tacotron2 is a model introduced in this paper [https://arxiv.org/abs/1712.05884]. 
        It takes as input a text and produces a mel-spectrogram of the corresponding audio. 
        
        This class inherits from `BaseTextModel` and `BaseAudioModel` as it combines both types of data (text and audio). It means that it handles all the features available in those 2 interfaces.
    """
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
        
        if hasattr(self.tts_model, '_build'): self.tts_model._build()
        if hasattr(self.tts_model, 'set_step'): self.tts_model.set_step(self.steps)
    
    def init_train_config(self, ** kwargs):
        super().init_train_config(** kwargs)

        if self.max_train_frames > 0:
            raise NotImplementedError("max_train_frames > 0 not working yet !")
        if self.pad_to_multiple and self.max_train_frames <= 0:
            raise ValueError("If pad_to_multiple is True, max_train_frames must be > 0 !")
        
        if not self.trim_mel: self.trim_mel_method = None
        
    def _build_model(self, **kwargs):
        super()._build_model(
            tts_model = {
                'architecture_name' : kwargs.pop('architecture_name', 'tacotron2'),
                'pad_token'     : self.blank_token_idx,
                'vocab_size'        : self.vocab_size,
                'n_mel_channels'    : self.n_mel_channels,
                'init_step'     : self.steps,
                ** kwargs
            }
        )
    
    @property
    def input_signature(self):
        return self.text_signature[:1] + (
            self.audio_signature, tf.TensorSpec(shape = (None,), dtype = tf.int32)
        )
    
    @property
    def output_signature(self):
        return (
            self.audio_signature, tf.TensorSpec(shape = (None, None), dtype = tf.float32)
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
        return tf.zeros((1, self.n_mel_channels), dtype = tf.float32)
    
    def __str__(self):
        return super().__str__() + self._str_text() + self._str_audio()
    
    def call(self, inputs, training = False, **kwargs):
        pred = self.tts_model(inputs, training = training, **kwargs)
        return pred if len(pred) != 2 else pred[0]
    
    @timer(name = 'inference')
    def infer(self, text, * args, early_stopping = True, max_length = -1, ** kwargs):
        if max_length <= 0: max_length = self.max_output_length
        
        if not isinstance(text, (list, tuple)):
            if isinstance(text, str):
                text    = tf.expand_dims(self.encode_text(text), axis = 0)
            elif len(tf.shape(text)) == 1:
                text    = tf.expand_dims(text, axis = 0)
        
        return self.tts_model.infer(
            text, early_stopping = early_stopping, max_length = max_length, return_state = False
        )
    
    def compile(self, loss = 'tacotronloss', metrics = [], **kwargs):
        super().compile(loss = loss, metrics = metrics, ** kwargs)
    
    def get_input(self, data, ** kwargs):
        return self.tf_encode_text(data)
    
    def get_mel_gate(self, data):
        mel = self.get_audio(data)
        
        mel = tf.concat([self.go_frame, mel], axis = 0)

        gate = tf.zeros((tf.shape(mel)[0] - 1,), dtype = tf.float32)
        gate = tf.concat([gate, tf.ones((1,), dtype = tf.float32)], axis = 0)
        
        return mel, gate
    
    def encode_data(self, data):
        encoded_text = self.get_input(data)
        
        mel, gate = self.get_mel_gate(data)
        
        return (encoded_text, mel, len(mel)), (mel, gate)
        
    def filter_data(self, inputs, outputs):
        if self.max_train_frames > 0: return True
        return tf.logical_and(
            tf.shape(inputs[0])[-1] <= self.max_input_length, 
            inputs[-1] <= self.max_output_length
        )
    
    def augment_data(self, inputs, outputs):
        mel_input, mel_length = inputs[-2:]
        mel_input = self.augment_audio(mel_input)
        
        return inputs[:-2] + (mel_input, mel_length), outputs
    
    def preprocess_data(self, inputs, outputs):
        mel_input, mel_length   = inputs[-2:]
        mel_output, gate        = outputs
        
        if self.pad_to_multiple and self.max_train_frames > 0:
            to_pad = tf.shape(gate)[1] % self.max_train_frames
            padding = self.max_train_frames - to_pad + 1
            
            if padding > 0:
                mel_input   = tf.pad(
                    mel_input, [(0,0), (0,padding), (0,0)], constant_values = self.pad_mel_value
                )
                mel_output  = tf.pad(
                    mel_output, [(0,0), (0,padding), (0,0)], constant_values = self.pad_mel_value
                )
                gate        = tf.pad(gate, [(0,0), (0, padding)], constant_values = 1.)
        
        mel_input   = mel_input[:, :-1]
        mel_output  = mel_output[:, 1:]
        gate        = gate[:, 1:]
        
        return inputs[:-2] + (mel_input, mel_length), (mel_output, gate)
    
    def get_dataset_config(self, ** kwargs):
        kwargs.update({
            'batch_before_map'  : True,
            'padded_batch'  : True,
            'pad_kwargs'    : {
                'padding_values'    : (
                    (self.blank_token_idx, self.pad_mel_value, 0), (self.pad_mel_value, 1.)
                )
            }
        })
        
        return super().get_dataset_config(**kwargs)
    
    def predict_with_target(self, batch, step, prefix, directory = None, 
                            max_pred = 4, **kwargs):
        if directory is None: directory = self.train_test_dir
        mel_dir     = os.path.join(directory, 'mels')
        plot_dir    = os.path.join(directory, 'plot')
        
        os.makedirs(mel_dir, exist_ok = True)
        os.makedirs(plot_dir, exist_ok = True)
        
        inputs, outputs = batch
        inputs          = [inp[:max_pred] for inp in inputs]
        mel_out, gate   = [out[:max_pred] for out in outputs]
        
        text    = inputs[0]
        mel, mel_length = inputs[-2:]
        batch_size  = len(mel)
        
        kwargs.setdefault('x_size', 10)
        kwargs.setdefault('y_size', 4)
        kwargs.setdefault('show', False)
        
        pred  = self.tts_model(inputs, training = False)
        infer = self.infer(* inputs[:-2])

        _, pred_mel, _, pred_attn = [p.numpy() for p in pred]
        _, infer_mel, infer_gate, infer_attn = [i.numpy() for i in infer]
        
        text = text.numpy()
        for i in range(batch_size):
            txt, target, target_gate    = text[i], mel_out[i], gate[i]
            p_mel, p_attn               = pred_mel[i], pred_attn[i]
            i_mel, i_attn, i_gate       = infer_mel[i], infer_attn[i], infer_gate[i]
            txt = self.decode_text(txt)
            
            
            length = mel_length[i]
            target, p_mel, p_attn = target[:length], p_mel[:length], p_attn[:length]
            
            if batch_size > 1:
                i_length = np.where(i_gate > 0.5)[0]
                if len(i_length) > 0:
                    i_length = i_length[0]
                    i_mel, i_attn = i_mel[:i_length], i_attn[:i_length]
            
            pred_title = 'Prediction at step {} for text :\n{}'.format(step, txt)
            inf_title = 'Inference at step {} for text :\n{}'.format(step, txt)
            
            prefix_i = prefix + '_{}'.format(i) if len(pred_mel) > 1 else prefix
            
            plot_spectrogram(
                target = target, predicted = p_mel, prediction_attention = p_attn,
                title = pred_title,  filename = os.path.join(plot_dir, prefix_i + '_pred.png'), 
                ** kwargs
            )
            plot_spectrogram(
                target = target, inference = i_mel, inference_attention = i_attn,
                title = inf_title,  filename = os.path.join(plot_dir, prefix_i + '_infer.png'), 
                ** kwargs
            )
            
            np.save(os.path.join(mel_dir, prefix_i + '_target.npy'), target)
            np.save(os.path.join(mel_dir, prefix_i + '_pred.npy'), p_mel)
            np.save(os.path.join(mel_dir, prefix_i + '_infer.npy'), i_mel)

        
    def get_pipeline(self,
                     max_text_length    = -1,
                     expand_acronyms    = True,
                     
                     save    = True,
                     show_mel   = False,
                     save_mel   = None,
                     save_plot  = None,
                     save_audio = None,
                     
                     directory   = None,
                     
                     expected_keys  = 'mels',
                     
                     batch_size = 8,
                     post_group = None,
                     pre_processing     = None,
                     post_processing    = None,
                     
                     ** kwargs
                    ):
        """
            Creates an *inference pipeline* (i.e. cleaning + text splitting + inference)
            
            Arguments :
                - max_text_length   : kwarg for `split_text`
                - expand_acronyms   : kwarg for `self.clean_text` call

                - save / save_mel / save_plot / save_audio  : whether to save results or not
                    If one of save_{} is None, it takes the value of `save`
                - directory : where to save the results
                    `save_mel`      creates a `/mels` sub-directory
                    `save_plot`     creates a `/plots` subdirectory
                    `save_audio`    creates a `/sentence_audio` subdirectory
                
                - expected_keys : (list of) keys that are required in the resulting output
                    For instance if `expected_keys = 'audio'` and the sentence has already a saved `audio` key but no `mel` key, it will not generate it as `mel` is not required
                
                - batch_size    : number of sentences to give for the inference (it is a maximum, not a strict value ! cf `Consumer` batch behavior)
                - pre_processing    : takes a text and returns a cleaned version of it that will be used as ID (see the note on saving)
                - post_processing   : `Consumer`-like applied after inference and before saving (useful for `Vocoder` inference)
                    It takes a single argument `infos` (dict), the result of the inference
                - post_group    : `Consumer`-like applied after grouping the results (for a given text)
                    It takes a single argument `infos` (dict), the grouped results
                
                - kwargs    : propagated to `self.infer`
            Returns : pipeline
                a `thread_utils.Consumer` class (already started) that takes a text as input and outputs its resulting dict {mels, audios, plots, gates, attn_weights}
                Note that some of these keys are optional and can be missing (only `expected_keys` must be in the result)
            
            Pipeline process :
                1) Receives a text and pre-process it with `pre_processing` (default `self.clean_text`)
                2) Split the text into sentences
                3) Encode the sentences
                4) Performs inference
                5) Apply `post_processing` (if provided)
                6) Saving
                    6.1) If `save_mel` : saves the mel-spectrogram and set the `mel` key to its filename
                    6.2) If `save_plot` : saves the attention / mel plot and set the `plot` key
                    6.3) If `save_audio` and `audio` key is in the result : saves the audio and sets the `audio` key to the saving filename
                7) Groups each sentence's result for a given text
                8) Apply `post_group` (if provided)
            
            Note that `Pipeline` tracks inputs and can restore an already processed ID.
            It means that :
                - If 2 t
        """
        @timer
        def _filter_mel(mel, gate, attn, ** kwargs):
            """ Converts a padded mel output to the right shape (according to `gate`) """
            stop_gate   = np.where(gate > 0.5)[0]
            mel_length  = stop_gate[0] if len(stop_gate) > 0 else len(gate)

            return {
                ** kwargs,
                'mel'           : mel[: mel_length],
                'gate'          : gate[: mel_length],
                'attn_weights'  : attn[: mel_length]
            }

        @timer
        def preprocess(text, ** kwargs):
            kwargs.setdefault('to_expand_acronyms', expand_acronyms)
            return self.clean_text(text, ** kwargs)
        
        @timer
        def sentence_splitter(text, ** kwargs):
            """ Splits `text` in sentences of at most `max_text_length` caracters """
            kwargs.setdefault('to_expand_acronyms', expand_acronyms)
            max_length = kwargs.pop('max_text_length', max_text_length)
            splitted    = [
                self.clean_text(s, to_expand_acronys = expand_acronyms, ** kwargs)
                for s in split_text(text, max_length)
            ]
            return (text, splitted if splitted else [''])
        
        @timer
        def inference(sent, ** kw):
            """
                Get a (list of) str : the text(s) to read
                Returnsa (list of) dict containing {mel, attn_weights, gate}
            """
            inputs = sent if isinstance(sent, list) else [sent]
            
            should_skip = [
                False if s and any(c.isalnum() for c in s) else True for s in inputs
            ]
            
            if any(not skip for skip in should_skip):
                batch   = tf.cast(pad_batch([
                    self.get_input(s) for s, skip in zip(inputs, should_skip) if not skip
                ], pad_value = self.blank_token_idx, dtype = np.int32), tf.int32)

                _, mels, gates, attn_weights = [
                    out.numpy() for out in self.infer(batch, ** kwargs)
                ]
            
            result, idx = [], 0
            for i, (txt, skip) in enumerate(zip(inputs, should_skip)):
                res = {'text' : txt}
                if not skip:
                    res = _filter_mel(mels[idx], gates[idx], attn_weights[idx], text = txt)
                    idx += 1
                result.append(res)
            
            return result if isinstance(sent, list) else result[0]
        
        @timer
        def save_plot_fn(result, overwritten_data = {}, ** kwargs):
            if 'mel' not in result: return result
            
            filename = None
            if save_mel:
                if 'plot' in overwritten_data:
                    filename    = overwritten_data['plot']
                else:
                    num_pred    = len(os.listdir(plot_dir))
                    filename    = os.path.join(plot_dir, 'attn_{}.png'.format(num_pred))
            
            audio = {} if 'audio' not in result else {'audio' : {
                'x': result['audio'], 'plot_type' : 'plot'
            }}
            plot_spectrogram(
                mel = result['mel'], attention = result['attn_weights'], ** audio,
                title = "Spectrogram for :\n{}".format(result['text']),
                filename = filename, show = show_mel
            )
            result['plot'] = filename
            
            return result

        @timer
        def save_mel_fn(result, overwritten_data = {}, ** kwargs):
            if 'mel' not in result: return result

            if 'mel' in overwritten_data:
                filename    = overwritten_data['mel']
            else:
                num_pred    = len(os.listdir(mel_dir))
                filename    = os.path.join(mel_dir, 'mel_{}.npy'.format(num_pred))
            
            np.save(filename, result['mel'])
            result['mel'] = filename
            
            return result

        @timer
        def maybe_save_audio(result, overwritten_data = {}, ** kwargs):
            if 'audio' not in result: return result

            if 'audio' in overwritten_data:
                filename    = overwritten_data['audio']
            else:
                num_pred    = len(os.listdir(audio_dir))
                filename    = os.path.join(audio_dir, 'audio_{}.mp3'.format(num_pred))
            
            write_audio(audio = result['audio'], filename = filename, rate = self.audio_rate)
            result['audio'] = filename
            
            return result
        
        # get saving directory
        if max_text_length <= 0:    max_text_length = self.max_input_length
        if save_mel is None:    save_mel    = save
        if save_plot is None:   save_plot   = save
        if save_audio is None:  save_audio  = save
        
        if pre_processing is None: pre_processing = preprocess
        if post_group is not None and not isinstance(post_group, list):
            post_group = [post_group]
        if post_processing is not None and not isinstance(post_processing, list):
            post_processing = [post_processing]

        
        save = save_mel or save_plot or save_audio
        
        mel_dir, plot_dir, audio_dir, map_file = None, None, None, None
        if save:
            if directory is None: directory = self.pred_dir
            mel_dir     = os.path.join(directory, 'mels')
            plot_dir    = os.path.join(directory, 'plots')
            audio_dir   = os.path.join(directory, 'sentence_audios')
            
            text_map_file   = os.path.join(directory, 'map.json')
            sent_map_file   = os.path.join(directory, 'map_sentences.json')
            
            if save_mel:    os.makedirs(mel_dir, exist_ok = True)
            if save_plot:   os.makedirs(plot_dir, exist_ok = True)
            if save_audio:  os.makedirs(audio_dir, exist_ok = True)

        saving_functions    = [] if post_processing is None else post_processing
        do_not_save_keys    = ['attn_weights', 'gate', 'gates']
        
        if save_plot:
            saving_functions.append({'consumer' : save_plot_fn, 'allow_multithread' : False})
        
        if save_mel:
            saving_functions.append({'consumer' : save_mel_fn, 'allow_multithread' : False})
        else:
            do_not_save_keys.extend(['mel', 'mels'])
        
        if save_audio:
            saving_functions.append({'consumer' : maybe_save_audio, 'allow_multithread' : False})
        else:
            do_not_save_keys.append('audios')
        
        pipeline = Pipeline(** {
            ** kwargs,
            'name'  : 'tts_pipeline',
            'filename'  : None,
            'track_items'   : False,
            
            'tasks' : [
                {'consumer' : preprocess, 'name' : 'pre_processing'},
                {
                    'name'      : 'text_inference',
                    'filename'  : None if not save else text_map_file,
                    'expected_keys' : expected_keys,
                    'do_not_save_keys'  : do_not_save_keys,

                    'tasks' : [
                        {'consumer' : sentence_splitter, 'splitter' : True, 'name' : 'text_splitter'},
                        {
                            'name'      : 'sentence_inference',
                            'filename'  : None if not save else sent_map_file,
                            'expected_keys' : [
                                k[:-1] if k.endswith('s') else k for k in expected_keys
                            ],
                            'do_not_save_keys'  : do_not_save_keys,

                            'tasks' : [
                                {
                                    'consumer'      : inference,
                                    'batch_size'    : batch_size,
                                    'allow_multithread' : False
                                },
                            ] + saving_functions
                        },
                        {
                            'consumer'  : 'grouper',
                            'nested_group' : True,
                            'suffix'    : 's',
                            'name'  : 'text_grouper'
                        }
                    ] + (post_group if post_group is not None else [])
                }
            ]
        })
        pipeline.start()
        return pipeline
    
    @timer
    def predict(self, sentences, ** kwargs):
        """
            Performs prediction on `sentences`, a (list of) str
            See `help(self.get_pipeline)` for more information about the configuration and procedure
        """
        if not isinstance(sentences, (list, pd.DataFrame)): sentences = [sentences]
        pipe    = self.get_pipeline(** kwargs)
        
        return pipe.extend_and_wait(sentences, stop = True, ** kwargs)

    def get_config(self, * args, ** kwargs):
        config = super(Tacotron2, self).get_config(* args, ** kwargs)
        config.update({
            ** self.get_config_text(),
            ** self.get_config_audio(),
            
            'max_input_length'  : self.max_input_length,
            'max_output_length' : self.max_output_length
        })
        
        return config
    
    @classmethod
    def build_from_nvidia_pretrained(cls, nom = 'pretrained_tacotron2', ** kwargs):
        kwargs.update({'audio_format' : 'mel', 'audio_rate' : 22050})
        kwargs.setdefault('lang', 'en')
        kwargs.setdefault('text_encoder', default_english_encoder())
        
        with tf.device('cpu') as device:
            instance = cls(
                nom = nom, max_to_keep = 1, pretrained_name = 'pytorch_nvidia_tacotron2', ** kwargs
            )
        
        nvidia_model = get_architecture('nvidia_tacotron', to_gpu = False)
        
        pt_convert_model_weights(nvidia_model, instance.tts_model)
        
        instance.save()
        
        return instance
