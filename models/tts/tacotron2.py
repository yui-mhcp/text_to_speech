
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
import glob
import time
import logging
import collections
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm

from loggers import timer
from custom_architectures import get_architecture
from models.interfaces.base_text_model import BaseTextModel
from models.interfaces.base_audio_model import BaseAudioModel
from models.weights_converter import pt_convert_model_weights
from utils import time_to_string, load_json, dump_json, plot_spectrogram, pad_batch
from utils.audio import write_audio, load_audio, display_audio
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
        
        if hasattr(self.tts_model, '_build'):   self.tts_model._build()
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
        return (
            self.text_signature,
            self.audio_signature,
            tf.TensorSpec(shape = (None, ), dtype = tf.int32)
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
    
    def compile(self, loss = 'tacotronloss', metrics = [], ** kwargs):
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
        
        return (encoded_text, mel[:-1], len(mel) - 1), (mel[1:], gate[1:])
        
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
            to_pad  = tf.shape(gate)[1] % self.max_train_frames
            padding = self.max_train_frames - to_pad + 1
            
            if padding > 0:
                mel_input   = tf.pad(
                    mel_input, [(0, 0), (0, padding), (0, 0)], constant_values = self.pad_mel_value
                )
                mel_output  = tf.pad(
                    mel_output, [(0, 0), (0, padding), (0, 0)], constant_values = self.pad_mel_value
                )
                gate        = tf.pad(gate, [(0, 0), (0, padding)], constant_values = 1.)
        
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

    @timer
    def predict(self,
                sentences,
                max_text_length = -1,
                batch_size  = 1,
                
                vocoder = None,
                display = True,
                play    = False,
                silence_time    = 0.15,
                
                save    = None,
                save_mel    = None,
                save_plot   = False,
                save_audio  = None,
                show_plot   = False,
                directory   = None,
                overwrite   = False,
                timestamp   = -1,
                required_key    = None,
                
                filename    = 'audio_{}.mp3',
                
                expand_acronyms = True,
                
                tqdm    = lambda x: x,
                
                ** kwargs
               ):
        """
            Perform Tacotron-2 inference on all phrases
            Arguments :
                - sentences   : (list of) string, the sentence(s) to read.
                - max_text_length   : the maximum length of a single text part (if a sentence is longer, it will be splitted in multiple parts)
                - batch_size    : the number of sentences to infer in parallel
                    /!\ if `vocoder` is provided, the same `batch_size` is used
                
                - vocoder   : a function that takes a `mel-spectrogram` as input and returns an audio (if it has an `infer` method, its `infer` method is called)
                - display   : whether to display the audio (only relevant in notebooks)
                - play      : whether to play the audio (currently only supported in combination with `display`, i.e. in notebooks)
                - silence_time  : the time of silence to add between each part of a splitted sentence
                
                - save      : whether to save the results
                - save_mel  : whether to save the produced mel-spectrogram
                - save_plot : whether to save plots of the mel-spectrogram + attention + audio
                - save_audio    : whether to save the produced audio (only relevant if `vocoder` is provided)
                - show_plot : whether to show the produced plots
                - directory : the directory in which the produced mapping file + files will be saved
                - overwrite : whether to overwrite if a sentence has already been predicted
                - timestamp : if provided and `overwrite = True`, only overwrites if `timestamp` is smaller than the timestamp of the prediction (i.e. the time at which it has been generated)
                - required_key  : this key is expected to be in the resulting output (see Notes)
                
                - filename  : the audio filename format (must contain a '{}')
                
                - expand_acronyms   : kwarg for the `clean_text` method
                
                - tqdm  : progress bar
                
                - kwargs    : forwarded to `infer`
            Returns :
                - results   : a list of tuple (sentence, infos)
                    - sentences : the sentence read
                    - infos     : a `dict` containing the information about the sentence
                        - parts     : the splitted version of the sentence
                        - timestamp : the time at which the sentence has been read by the model
                        - mels      : the mel-spectrograms for each part
                        - plots (if `save_plot`)    : the plot filenames for each part
                        - audios (if `vocoder`)     : the audios for each part
                        - audio (if `vocoder`)      : the concatenation of each audio parts
            
            Note : in the returned `infos`, the `mels`, `audios` andàudio  may contain either a string (the filename), either the raw mel / audio, depending on the `save_{mel / audio}` value. 
            Note that only the filenames are saved in the mapping file : if `save_mel = False` and `save_audio = True`, the mapping file will only contain the `audios` and `audio` keys (not `mels`) while, in the result, `mels` will be a list of np.ndarray
            
            If the sentence was already generated and is not overwritten, it is possible that some keys will not be in the output, only the previously saved ones will be there. 
            However, if `required_key` was not saved, the sentence is overwritten, even if `overwrite = False`, in order to have the expected key in the output result
        """
        ####################
        # Saving functions #
        ####################
        @timer(name = 'saving plot')
        def save_plot_fn(text, mel, attn, audio):
            if text in saved_sents:
                filename    = saved_sents[text].get('plot', saved_sents[text].pop('old_plot', None))
                if not filename:
                    filename    = os.path.join(
                        plot_dir, 'attn_{}.png'.format(len(os.listdir(plot_dir)))
                    )
            
            audio = {} if audio is None else {'audio' : {
                'x': audio, 'rate' : self.audio_rate, 'plot_type' : 'audio'
            }}
            plot_spectrogram(
                mel = mel, attention = attn, ** audio,
                title = "Spectrogram for :\n{}".format(text),
                filename = filename, show = show_plot
            )
            
            return filename

        @timer(name = 'saving mel')
        def save_mel_fn(text, mel):
            if text in saved_sents:
                filename    = saved_sents[text].get('mel', saved_sents[text].pop('old_mel', None))
                if not filename:
                    filename    = os.path.join(
                        mel_dir, 'mel_{}.npy'.format(len(os.listdir(mel_dir)))
                    )
            
            np.save(filename, mel)
            
            return filename

        @timer(name = 'saving audio')
        def save_audio_fn(text, audio, audio_dir, saved_infos):
            if text in saved_infos:
                audio_filename    = saved_infos[text].get('audio', saved_infos[text].pop('old_audio', None))
                if not audio_filename or os.path.dirname(audio_filename) != audio_dir.replace(os.path.sep, '/'):
                    audio_filename    = os.path.join(audio_dir, filename.format(
                        len(glob.glob(os.path.join(audio_dir, filename.replace('{}', '*'))))
                    ))
            
            write_audio(audio = audio, filename = audio_filename, rate = self.audio_rate)
            
            return audio_filename

        def maybe_save_silence():
            silence_file = None
            if save_audio:
                silence_file = os.path.join(audio_dir, 'silence.{}'.format(filename.split('.')[-1]))
                if not os.path.exists(silence_file):
                    write_audio(audio = silence, filename = silence_file, rate = self.audio_rate)
            return {'raw' : silence, 'filename' : silence_file}
        
        def should_predict(text, infos):
            if required_key in infos.get(text, {}):
                if not overwrite or (timestamp != -1 and timestamp <= infos[text].get('timestamp', -1)):
                    return False
            return True
        
        ####################
        #  Initialization  #
        ####################
        
        time_logger.start_timer('initialization')

        now = time.time() # used to check whether we should overwrite or not (if `timestamp != -1`)
        
        if max_text_length <= 0: max_text_length = self.max_input_length
        
        if save is None:        save = True if vocoder is None else False
        if save_mel is None:    save_mel   = save
        if save_plot is None:   save_plot  = save
        if save_audio is None:  save_audio = save
        if vocoder is None:     save_audio = False
        save = save_mel or save_plot or save_audio
        if vocoder is not None:
            if save:    save_audio = True
            else:       display = True
        
        if isinstance(sentences, pd.DataFrame):      sentences = sentences.to_dict('records')
        if not isinstance(sentences, (list, tuple)): sentences = [sentences]
        if len(sentences) > 1: play = False

        if required_key is None:
            required_key    = 'mels' if vocoder is None else 'audio'
        
        # get saving directory
        if directory is None: directory = self.pred_dir
        mel_dir, plot_dir, audio_dir, text_audio_dir = None, None, None, None

        map_file      = os.path.join(directory, 'map.json')
        map_sent_file = os.path.join(directory, 'map_sentences.json')

        if save:
            if save_mel:
                mel_dir = os.path.join(directory, 'mels')
                os.makedirs(mel_dir, exist_ok = True)

            if save_plot:
                plot_dir = os.path.join(directory, 'plots')
                os.makedirs(plot_dir, exist_ok = True)

            if save_audio:
                sent_audio_dir  = os.path.join(directory, 'audios_sent')
                os.makedirs(sent_audio_dir, exist_ok = True)

                text_audio_dir = os.path.join(directory, 'audios')
                os.makedirs(text_audio_dir, exist_ok = True)
        # load previous generated (if any)
        saved_texts = load_json(map_file, default = {})
        saved_sents = load_json(map_sent_file, default = {})
        
        time_logger.stop_timer('initialization')
        
        ####################
        #  Pre-processing  #
        ####################
        time_logger.start_timer('pre-processing')

        # maps the sentence to read to its cleaned representation
        cleaned = collections.OrderedDict()
        for sent in sentences:
            if isinstance(sent, (dict, pd.Series)):
                if 'text' not in sent:
                    raise ValueError('`text` must be in the `dict` !\n  Got : {}'.format(sent))
                sent = sent['text']
            
            if sent not in cleaned: cleaned[sent] = self.clean_text(sent, expand_acronyms = expand_acronyms)
        logger.debug('Cleaned : {}'.format(cleaned))
        # maps each unique cleaned sentence to its splitted version
        # Note : multiple sentences may have the same cleaned representation
        splitted = collections.OrderedDict()
        for sent, clean in cleaned.items():
            if clean not in splitted and should_predict(clean, saved_texts):
                splitted[clean] = split_text(clean, max_text_length)

        logger.debug('Splitted : {}'.format(splitted))
        # filters unique parts with at least 1 alphanumeric caracter (others will be "silence")
        flattened   = set()
        for sent, parts in splitted.items():
            flattened.update(
                p for p in parts if any(c.isalnum() for c in p) and should_predict(p, saved_sents)
            )
        # Useful to sort according to the length for batching
        flattened   = list(sorted(flattened, key = len))
        
        encoded     = [self.encode_text(text, cleaned = True) for text in flattened]
        # filters empty encoded sentences
        # theorically nothing is filtered as sentences are already cleaned, just for check
        flattened   = [txt for i, txt in enumerate(flattened) if len(encoded[i]) > 0]
        encoded     = [enc for enc in encoded if len(enc) > 0]
        
        logger.debug('Flattened : {}'.format(flattened))
        
        # pre-compute the silence : used for empty sentences + between each sub-parts of a long text
        silence = np.zeros((int(self.audio_rate * silence_time), ))
        
        time_logger.stop_timer('pre-processing')
        
        # matches a text sub-part to its spect / audio / plot
        # In the case of mels / audios, it stores a dict {filename: , raw:} to avoid multiple reloading / saving
        pred_mels   = {}
        pred_plots  = {}
        pred_audios = {}
        
        # for each batch : computes the mel (+ audio if `vocoder` is provided)
        total_infer_time, total_gen_time = 0., 0.
        for start_idx in tqdm(range(0, len(encoded), batch_size)):
            texts   = flattened[start_idx : start_idx + batch_size]
            batch   = encoded[start_idx : start_idx + batch_size]
            batch   = tf.cast(pad_batch(
                batch, pad_value = self.blank_token_idx, dtype = np.int32
            ), tf.int32) if len(batch) > 1 else tf.expand_dims(batch[0], axis = 0)

            ####################
            #    Inference     #
            ####################
            
            start_infer = time.time()
            
            _, mels, gates, attns = self.infer(batch, ** kwargs)
            #_, mels, gates, attns = [tf.random.normal((len(texts), 4, 4)) for _ in range(4)]
            gates, attns = gates.numpy(), attns.numpy()

            if vocoder is not None:
                audios = vocoder(mels)
                #audios = tf.random.normal((len(texts), 22050))
                if hasattr(audios, 'numpy'): audios = audios.numpy()
            else:
                audios = [None] * len(texts)
            
            total_infer_time += (time.time() - start_infer)
            
            mels = mels.numpy()
            for i, (text, mel, gate, attn, audio) in enumerate(zip(texts, mels, gates, attns, audios)):
                time_logger.start_timer('post-processing')
                
                if len(texts) > 1:
                    stop_gate    = np.where(gate > 0.5)[0]
                    mel_length   = stop_gate[0] if len(stop_gate) > 0 else len(gate)

                    mel     = mel[: mel_length]
                    attn    = attn[: mel_length]
                
                audio_length   = len(mel) * self.mel_fn.hop_length
                total_gen_time += audio_length / self.audio_rate
                
                pred_mels[text]     = {'raw' : mel, 'filename' : None}
                pred_plots[text]    = attn
                if audio is not None:
                    if len(texts) > 1: audio   = audio[: audio_length]
                    pred_audios[text]   = {'raw' : audio, 'filename' : None}
                
                time_logger.stop_timer('post-processing')
                
                if save:
                    saved_sents.setdefault(text, {})
                    saved_sents[text]['timestamp'] = now
                    
                    if save_mel:
                        pred_mels[text]['filename'] = save_mel_fn(text, mel)
                        saved_sents[text]['mel']    = pred_mels[text]['filename']
                    elif 'mel' in saved_sents[text]:
                        saved_sents[text]['old_mel'] = saved_sents[text].pop('mel')

                    if save_plot:
                        pred_plots[text] = save_plot_fn(text, mel, attn, audio)
                        saved_sents[text]['plot']   = pred_plots[text]
                    elif 'plot' in saved_sents[text]:
                        saved_sents[text]['old_plot'] = saved_sents[text].pop('plot')

                    if save_audio:
                        pred_audios[text]['filename'] = save_audio_fn(
                            text, audio, sent_audio_dir, saved_sents
                        )
                        saved_sents[text]['audio']    = pred_audios[text]['filename']
                    elif 'audio' in saved_sents[text]:
                        saved_sents[text]['old_audio'] = saved_sents[text].pop('audio')

                    time_logger.start_timer('saving json')
                    dump_json(map_sent_file, saved_sents, indent = 4)
                    time_logger.stop_timer('saving json')
        
        time_logger.start_timer('finalization')

        for text, parts in splitted.items():
            saved_texts.setdefault(text, {})
            if 'audio' in saved_texts[text]:
                if len(saved_texts[text].get('audios', [])) <= 1:
                    saved_texts[text].pop('audio')
                else:
                    saved_texts[text]['old_audio'] = saved_texts[text].pop('audio')

            for k in ('mels', 'plots', 'audios'):
                if k in saved_texts[text]: saved_texts[text][k] = []
            
            saved_texts[text].update({
                'timestamp' : min(
                    saved_sents.get(p, {}).get('timestamp', now) for p in parts
                ) if len(parts) > 0 else now,
                'parts' : parts
            })
            
            if save:
                if save_mel:
                    saved_texts[text]['mels']   = [
                        saved_sents[p]['mel'] for p in parts if p in saved_sents
                    ]
                
                if save_plot:
                    saved_texts[text]['plots']  = [
                        saved_sents[p]['plot'] for p in parts if p in saved_sents
                    ]
                
                if save_audio:
                    saved_texts[text]['audios'] = [
                        saved_sents[p]['audio'] for p in parts if p in saved_sents
                    ]
            elif overwrite or text not in saved_texts:
                saved_texts[text] = {}
                for p in parts:
                    if p in pred_mels:
                        saved_texts[text].setdefault('mels', []).append(pred_mels[p]['raw'])
                    elif 'mel' in saved_sents.get(p, {}):
                        saved_texts[text].setdefault('mels', []).append(saved_sents[p]['mel'])
                    
                    if p in pred_plots:
                        saved_texts[text].setdefault('plots', []).append(pred_plots[p])
                    elif 'plot' in saved_sents.get(p, {}):
                        saved_texts[text].setdefault('mels', []).append(saved_sents[p]['plot'])
                    
                    if vocoder is not None:
                        if p in pred_audios:
                            saved_texts[text].setdefault('audios', []).append(pred_audios[p]['raw'])
                        elif 'audio' in saved_sents.get(p, {}):
                            saved_texts[text].setdefault('audios', []).append(saved_sents[p]['audio'])
            
            if vocoder is not None:
                audios = []
                for p in parts:
                    if p not in pred_audios:
                        if p in saved_sents:
                            pred_audios[p] = {'raw' : None, 'filename' : saved_sents[p]['audio']}
                        else:
                            pred_audios[p] = maybe_save_silence()
                    
                    if len(parts) > 1:
                        if pred_audios[p]['raw'] is None:
                            pred_audios[p]['raw'] = load_audio(
                                pred_audios[p]['filename'], self.audio_rate
                            )

                        audios.extend([pred_audios[p]['raw'], silence])
                    else:
                        audios.append(
                            pred_audios[p]['raw'] if pred_audios[p]['filename'] is None else pred_audios[p]['filename']
                        )
                
                if len(audios) > 1:
                    audios = np.concatenate(audios[:-1], axis = 0)
                    
                    if save_audio:
                        audios = save_audio_fn(
                            text, audios, text_audio_dir, saved_texts
                        )
                else:
                    if len(audios) == 1:
                        audios = audios[0]
                    else: #len(audios) == 0
                        audios = maybe_save_silence()
                        audios = audios['raw'] if audios['filename'] is None else audios['filename']
                
                # either `save_audio == True`, either `save == False`
                # In both cases, we have to add `audio` to saved_texts
                saved_texts[text]['audio'] = audios
                if text not in pred_audios:
                    pred_audios[text] = {'raw' : audios, 'filename' : saved_texts.get(text, {}).get('audio', None)}

        if display and vocoder is not None:
            time_logger.start_timer('display')
            for text, clean in cleaned.items():
                audio = pred_audios.get(clean, {}).get('raw', saved_texts.get(clean, {}).get('audio', None))
                if audio is None: raise RuntimeError('Audio for {} is None !'.format(clean))
                
                print('Text : {}'.format(text))
                display_audio(audio, rate = self.audio_rate, play = play)
            time_logger.stop_timer('display')

        if save and len(splitted) > 0:
            time_logger.start_timer('saving json')
            dump_json(map_file, saved_texts, indent = 4)
            time_logger.stop_timer('saving json')

        time_logger.stop_timer('finalization')

        if total_infer_time > 0:
            logger.info('{} generated in {} ({} generated / sec)'.format(
                time_to_string(total_gen_time),
                time_to_string(total_infer_time),
                time_to_string(total_gen_time / total_infer_time)
            ))

        return [(sent, saved_texts.get(cleaned[sent], {})) for sent in sentences]

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
    def from_nvidia_pretrained(cls, nom = 'pretrained_tacotron2', ** kwargs):
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
