
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

time_logger = logging.getLogger('timer')

DEFAULT_MAX_MEL_LENGTH  = 1024
DEFAULT_MAX_TEXT_LENGTH = 150

class Tacotron2(BaseTextModel, BaseAudioModel):
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
                'vocab_size'        : self.vocab_size,
                'n_mel_channels'    : self.n_mel_channels,
                'init_step'     : self.steps,
                ** kwargs
            }
        )
    
    @property
    def input_signature(self):
        return self.text_signature + (
            self.audio_signature,
            tf.TensorSpec(shape = (None,), dtype = tf.int32)
        )
    
    @property
    def output_signature(self):
        return (
            self.audio_signature,
            tf.TensorSpec(shape = (None, None), dtype = tf.float32)
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
        des = super().__str__()
        des += self._str_text()
        des += self._str_audio()
        return des
    
    def call(self, inputs, training = False, **kwargs):
        pred = self.tts_model(inputs, training = training, **kwargs)
        return pred if len(pred) != 2 else pred[0]
    
    @timer(name = 'inference')
    def infer(self, text, text_length = None, * args, ** kwargs):
        if not isinstance(text, list):
            if isinstance(text, str):
                text        = tf.expand_dims(self.encode_text(text), axis = 0)
                text_length = tf.cast([len(text[0])], tf.int32)
            elif len(tf.shape(text)) == 1:
                text        = tf.expand_dims(text, axis = 0)
        
        if text_length is None and tf.shape(text)[0] == 1:
            text_length = tf.cast([tf.shape(text)[1]], tf.int32)
        
        assert text_length is not None, "You must specify `text_length` or pas a `list`"
        
        pred = self.tts_model.infer(text, text_length, * args, ** kwargs)
        return pred if len(pred) != 2 else pred[0]
    
    def compile(self, loss = 'tacotronloss', metrics = [], **kwargs):
        super().compile(loss = loss, metrics = metrics, ** kwargs)
    
    def get_mel_gate(self, data):
        mel = self.get_audio(data)
        
        mel = tf.concat([self.go_frame, mel], axis = 0)

        gate = tf.zeros((tf.shape(mel)[0] - 1,), dtype = tf.float32)
        gate = tf.concat([gate, tf.ones((1,), dtype = tf.float32)], axis = 0)
        
        return mel, gate
    
    def encode_data(self, data):
        encoded_text = self.tf_encode_text(data)
        
        mel, gate = self.get_mel_gate(data)
        
        return encoded_text, len(encoded_text), mel, len(mel), mel, gate
        
    def filter_data(self, text, text_length, mel_input, mel_length, mel_output, gate):
        if self.max_train_frames > 0: return True
        return tf.logical_and(
            text_length <= self.max_input_length, 
            mel_length <= self.max_output_length
        )
    
    def augment_data(self, text, text_length, mel_input, mel_length, mel_output, gate):
        mel_input = self.augment_audio(mel_input)
        
        return text, text_length, mel_input, mel_length, mel_output, gate
    
    def preprocess_data(self, text, text_length, mel_input, mel_length, mel_output, gate):
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
        
        return (text, text_length, mel_input, mel_length), (mel_output, gate)
    
    def get_dataset_config(self, **kwargs):
        kwargs['pad_kwargs']    = {
            'padding_values'    : (
                self.blank_token_idx, 0, self.pad_mel_value, 0, self.pad_mel_value, 1.
            )
        }
        kwargs['batch_before_map']  = True
        kwargs['padded_batch']      = True
        
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
        
        text, text_length       = inputs[:2]
        mel, mel_length         = inputs[-2:]
        batch_size              = len(mel)
        
        kwargs.setdefault('x_size', 10)
        kwargs.setdefault('y_size', 4)
        kwargs.setdefault('show', False)
        
        pred = self.tts_model(inputs, training = False)
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
                title = pred_title, 
                filename = os.path.join(plot_dir, prefix_i + '_pred.png'), 
                ** kwargs
            )
            plot_spectrogram(
                target = target, inference = i_mel, inference_attention = i_attn,
                title = inf_title, 
                filename = os.path.join(plot_dir, prefix_i + '_infer.png'), 
                ** kwargs
            )
            
            np.save(os.path.join(mel_dir, prefix_i + '_target.npy'), target)
            np.save(os.path.join(mel_dir, prefix_i + '_pred.npy'), p_mel)
            np.save(os.path.join(mel_dir, prefix_i + '_infer.npy'), i_mel)

        
    def get_pipeline(self,
                     save    = True,
                     save_mel   = None,
                     save_plot  = None,
                     save_audio = None,
                     
                     directory   = None,
                     
                     required_keys  = 'mel',
                     
                     expand_acronyms    = True,
                     
                     batch_size     = 8,
                     max_workers    = 0,
                     post_processing    = None,
                     
                     ** kwargs
                    ):
        """
            Creates an `inference pipeline` with possibly saving the result
            
            Arguments :
                - save / save_mel / save_plot / save_audio  : whether to save results or not
                    If one of save_{} is None, it takes the value of `save`
                - directory : where to save the results
                    `save_mel`      creates a `/mels` sub-directory
                    `save_plot`     creates a `/plots` subdirectory
                    `save_audio`    creates a `/sentence_audio` subdirectory (see the important note below for audio saving)
                
                - required_keys : (list of) keys that are required in the resulting output
                    For instance if `required_keys = 'audio'` and the sentence has already a saved `audio` key but no `mel` key, it will not generate it as `mel` is not required
                
                - expand_acronyms   : kwarg for `self.clean_text` call
                
                - batch_size    : number of sentences to give for the inference (it is a maximum, not a strict value ! cf `Consumer` batch behavior)
                - max_workers   : max_workers argument for the `Consumer`'s (maximum 1)
                - post_processing   : `Consumer`-like applied after inference and before saving (useful for `Vocoder` inference)
                
                - kwargs    : propagated to `self.infer`
            Return : (sent_mapping, pipeline)
                - sent_mapping  : `ThreadedDict` that contains information about generated sentences
                - pipeline      : list of 3 dict representing the inference pipeline (can be given to a `Pipeline` object or to `add_consumer`
            
            Important Note : this function does not generate any audio ! the `save_audio` is therefore only used if the `post_processing` adds an `audio` key to the sentence's information
            
            Pipeline process :
                1) Receives sentences' information (sent, overwrite, timestamp) or sent (simple str)
                2) Filters duplacted sentences to avoid generating multiple times the same sentence
                3) Performs inference on kept sentences
                4) Maps the filtered and kept sentences to their respective result (mel / attn_weights / ...)
                5) Add each sentence in `in_pipeline` such that if the same sentence comes before the end, it will not be re-generated
                6) Apply `post_processing` (identity function if `post_processing is None`)
                7) Save mel / plot / audio (if required and provided and not already done*)
                8) Removes the sentence from `in_pipeline`
                
                * A sentence already generated with `overwrite = False` or a duplicata sentence will have a `mel` key which is already a filename (and not the raw generated mel) so it will not save it again.
            
            Notice that the pipeline does not perform any processing on `sentences` meaning that you have to split them before sending them to the pipeline.
            
            The `timestamp` value is used to determine if the already generated sentence is recent enough to be sent back as is or if it is expected to re-generate it. 
            For instance if you send 2 identical sentences, the 1st one will enter the pipeline and go out of it and will be saved with a more recent `timestamp`. Therefore, the 2nd one (with an older timestamp) will not be re-generated as the saved version is more recent (so has already been overwritten). 
        """
        @timer
        def _filter_sent(sent_infos):
            """
                Filter sentences to return those to predict
                
                Arguments :
                    - sent_infos : (list of) sentence informations
                        str : the sentence to read
                        tuple (sent, overwrite, timestamp)  : the sentence with its last timestamp and whether to overwrite it or not
                Returns : (list of) str or None (None if we should not predict the text)
                    If the input was a list and no sentences should be predicted, the output is an empty list
            """
            if isinstance(sent_infos, list):
                sents = [_filter_sent(s) for s in sent_infos]
                return list(set(s for s in sents if s is not None and s.strip()))
            
            if isinstance(sent_infos, str):
                sent, overwrite, timestamp = sent_infos, False, -1
            else:
                sent, overwrite, timestamp = sent_infos

            if sent in in_pipeline: return None
            infos = sent_mapping.get(sent, {})
            
            if not infos or (overwrite and infos.get('timestamp', 0) <= timestamp) or any(
                infos.get(k, None) is None for k in required_keys):
                return sent
            return None
        
        @timer
        def _filter_mel(mel, gate, attn):
            """ Converts a padded mel output to the right shape (according to `gate`) """
            stop_gate   = np.where(gate > 0.5)[0]
            mel_length  = stop_gate[0] if len(stop_gate) > 0 else len(gate)

            return {
                'mel'           : mel[: mel_length],
                'gate'          : gate[: mel_length],
                'attn_weights'  : attn[: mel_length]
            }
        
        @timer
        def _map_sent_to_mel(sent_infos, sents, mels, gates, attn_weights, timestamp = None):
            """
                Maps each input sentence to its corresponding mel / gate / attn
                
                Arguments :
                    - sent_infos    : the original input, the expected sentences
                    - sents         : the filtered sentences to predict (from `_filter_sent`)
                    - mels          : mel output
                
                Note : `len(sent_infos) ?= (len(sents) == len(mel) == len(gates) == len(attn))`
                    It means that each `sent` (in sents) has its corresponding output mel but some sent in `sent_infos` can be duplicates / already predicted / already in `in_pipeline`
            """
            def _get_mel(sent_info):
                sent = sent_info[0] if not isinstance(sent_info, str) else sent_info
                
                if sent not in mapping:
                    with in_pipeline.mutex:
                        if sent in in_pipeline:
                            mapping[sent] = in_pipeline[sent]
                    if sent not in mapping:
                        mapping[sent] = sent_mapping.get(sent, {}).copy()

                    mapping[sent].setdefault('timestamp', timestamp)
                
                return (sent, mapping[sent])
            
            if timestamp is None: timestamp = time.time()
            if len(sents) != 1:
                mapping = {
                    s : _filter_mel(mel, gate, attn)
                    for s, mel, gate, attn in zip(sents, mels, gates, attn_weights)
                }
            else:
                mapping = {
                    sents[0] : {
                        'mel'   : mels[0, :-1],
                        'gate'  : gates[0, :-1],
                        'attn_weights' : attn_weights[0, :-1]
                    }
                }
            for k, v in mapping.items(): v['timestamp'] = timestamp
            
            if isinstance(sent_infos, list):
                return [_get_mel(s) for s in sent_infos]
            return _get_mel(sent_infos)
        
        def add_in_pipeline(pred):
            if isinstance(pred, list):
                return [add_in_pipeline(p) for p in pred]
            
            sent, infos = pred
            logging.debug('[START] {}'.format(sent))
            in_pipeline[sent] = infos
            return pred
        
        @timer
        def _infer(sent_infos):
            """
                Get a (list of) str or tuple (text, overwrite, last_timestamp) : the text to read (must be previously splitted if required)
                Returnsa (list of) tuple (sent, infos) where `infos` is a dict containing new information
                    `infos` contains :
                        If generated: {mel, attn_weights, gate, timestamp}
                        else: `sent_mapping[sent]` (the information in `sent_mapping`)
            """
            start_time  = time.time()
            
            text    = _filter_sent(sent_infos)
            if text is None: text = []
            elif not isinstance(text, list): text = [text]
            
            mels, gates, attn_weights = [], [], []
            if len(text) > 0:
                encoded = [self.encode_text(t, to_expand_acronyms = expand_acronyms) for t in text]
                lengths = np.array([len(enc) for enc in encoded])

                inputs  = pad_batch(encoded, 0, dtype = np.int32) if len(encoded) > 1 else encoded

                lengths = tf.cast(lengths, tf.int32)
                inputs  = tf.cast(inputs, tf.int32)

                _, mels, gates, attn_weights = self.infer(
                    text = inputs, text_length = lengths, ** kwargs
                )
                mels, gates, attn_weights = mels.numpy(), gates.numpy(), attn_weights.numpy()
            
            outputs = _map_sent_to_mel(
                sent_infos, text, mels, gates, attn_weights, timestamp = start_time
            )
            outputs = add_in_pipeline(outputs)
            return outputs
        
        @timer
        def _save(pred):
            """ Saves required information from the input tuple (sent, new_infos) """
            @timer
            def _maybe_save_mel():
                mel = new_infos.get('mel', None)
                if save_mel and mel is not None and not isinstance(mel, str):
                    if 'mel' in infos:
                        filename    = infos['mel']
                    else:
                        num_pred    = len(os.listdir(mel_dir))
                        filename    = os.path.join(mel_dir, 'mel_{}.npy'.format(num_pred))
                        
                        infos.update({'mel' : filename})
                    
                    np.save(filename, mel)
                    return True
                return False
            
            @timer
            def _maybe_save_plot():
                mel     = new_infos.get('mel', None)
                attn    = new_infos.get('attn_weights', None)
                if save_plot and attn is not None and not isinstance(attn, str):
                    if 'plot' in infos:
                        filename = infos['plot']
                    else:
                        num_pred    = len(os.listdir(plot_dir))
                        filename    = os.path.join(plot_dir, 'attn_{}.png'.format(num_pred))
                        
                        infos.update({'plot' : filename})
                    
                    to_plot = {'attention' : attn}
                    if mel is not None and not isinstance(mel, str): to_plot['spectrogram'] = mel
                    plot_spectrogram(
                        ** to_plot, filename = filename, show = False, 
                        title = "Spectrogram for :\n{}".format(text)
                    )
                    return True
                return False

            @timer
            def _maybe_save_audio():
                audio = new_infos.get('audio', None)
                if save_audio and audio is not None and not isinstance(audio, str):
                    if 'audio' in infos:
                        filename = infos['audio']
                    else:
                        num_pred    = len(os.listdir(audio_dir))
                        filename    = os.path.join(audio_dir, 'audio_{}.mp3'.format(num_pred))
                    
                    infos.update({'audio' : filename, 'duree' : new_infos['duree']})
                    
                    write_audio(audio = audio, filename = filename, rate = self.audio_rate)
                    return True
                return False
            
            text, new_infos = pred
            
            infos   = sent_mapping.get(text, {})
            
            if not in_pipeline.get(text, False):
                logging.info('[END DUPLICATE] {}'.format(text))
                return (text, infos)
            
            new_infos['timestamp'] = time.time()
            
            if save:
                infos['timestamp'] = new_infos['timestamp']
                _maybe_save_mel()
                _maybe_save_plot()
                _maybe_save_audio()
            else:
                infos.update(new_infos)
            
            logging.debug('[END] {}'.format(text))
            sent_mapping[text] = infos
            
            if save:
                dump_json(map_file, sent_mapping, indent = 4)
            
            in_pipeline.pop(text, None)

            return (text, infos)
        
        # get saving directory
        if save_mel is None:    save_mel = save
        if save_plot is None:   save_plot = save
        if save_audio is None:  save_audio = save
        save = save_mel or save_plot or save_audio
        
        if not save:
            mel_dir, plot_dir, audio_dir, map_file = None, None, None, None
        else:
            if directory is None: directory = self.pred_dir
            mel_dir     = os.path.join(directory, 'mels')
            plot_dir    = os.path.join(directory, 'plots')
            audio_dir   = os.path.join(directory, 'sentence_audio')
            map_file    = os.path.join(directory, 'map_sentences.json')
            
            if save_mel:    os.makedirs(mel_dir, exist_ok = True)
            if save_plot:   os.makedirs(plot_dir, exist_ok = True)
            if save_audio:  os.makedirs(audio_dir, exist_ok = True)

        if required_keys is None: required_keys = []
        elif not isinstance(required_keys, (list, tuple)): required_keys = [required_keys]
        # load previous generated (if any)
        sent_mapping    = load_json(map_file, default = {}) if map_file else {}
        sent_mapping    = ThreadedDict(** sent_mapping)
        
        in_pipeline     = ThreadedDict()
        
        if post_processing is None:
            post_processing = {
                'consumer' : lambda item: item, 'max_workers' : -1, 'name' : 'identity'
            }
        
        return sent_mapping, [
            {
                'consumer'      : _infer,
                'batch_size'    : batch_size,
                'max_workers'   : min(max_workers, 1),
                'name'  : 'inference'
            },
            post_processing,
            {
                'consumer'      : _save,
                'max_workers'   : min(max_workers, 1),
                'name'  : 'saving'
            }
        ]
    
    def get_streaming_pipeline(self,
                               buffer   = None,
                               max_text_length = -1,

                               required_keys    = 'mel',
                               
                               save         = True,
                               save_parts   = None,
                               directory    = None,
                               
                               post_group   = None,
                               
                               pipeline_workers = 0,
                               max_workers  = 0,
                               
                               ** kwargs
                              ):
        """
            Creates the complete `inference pipeline` with processing (splitting and re-grouping sentences)
            The issue with the `Producer` is that it does not handle single-input to multi-output which is the case if the text is too long and should be splitted in multiple sub-sentences.
            In this case, this function splits the text, adds "manually" each sentence to the `inference pipeline` (cf `self.get_pipeline`) then waits that all individual sentences have been generated to save then return the final result. 
            It therefore co-exists 2 distinct pipelines : the `inference` pipeline and the `splitting / grouping` pipeline that adds individual sentences to the 1st pipeline. 
            
            Arguments :
                - buffer    : the `buffer` argument for the 2 pipelines
                - max_text_length   : maximum length for the text splitting
                
                - required_keys : same argument as `self.get_pipeline`
                
                - save  : whether to save the global result or not
                - save_parts    : whether to save individual parts or not (given as `save` to `self.get_pipeline`)
                - directory     : where to save the results
                
                - post_group    : function applied after `_group` (re-grouping information from individual sentences of the text) and `_save` that saves the global result to `map.json`
                
                - pipeline_workers  : given as `max_workers` to `self.get_pipeline`
                - max_workers   : max_workers for this second processing pipeline
                
                - kwargs    : forwarded to `self.get_pipeline`
            
            Returns : (sentence_splitter, grouper, pipeline, text_mapping, sent_mapping)
                - sentence_splitter : the `Consumer` that splits sentences, the input of the 2nd (processing) pipeline
                - grouper   : the `_group` (or `post_group`) `Consumer`
                - pipeline  : the `Pipeline` object for the 1st `inference pipeline`
                - text_mapping  : `ThreadedDict` that maps text to information
                - sent_mapping  : `ThreadedDict` that maps sentences to information
            
            Note that `sentence_splitter` is a `Consumer`, meaning that you have to "manually" add data to it to start the pipeline
        """
        @timer
        def _sentence_splitter(text, overwrite = False, p = -1, blocking = True, ** kwargs):
            infos       = text_mapping.get(text, {}).copy()
            splitted    = infos.get('splitted', None)
            if overwrite or splitted is None:
                splitted    = [
                    self.clean_text(s, ** kwargs) for s in split_text(text, max_text_length)
                ]
                splitted    = [s for s in splitted if s and any(c.isalnum() for c in s)]
                
                timestamps  = {
                    s : sent_mapping.get(s, {}).get('timestamp', 0) for s in set(splitted)
                }
                for sent, t in timestamps.items():
                    pipeline((sent, overwrite, t), priority = p)
            else:
                timestamps = {sent : t for sent, t in zip(splitted, infos.get('timestamps', []))}
            
            if not blocking: return None
            return text, splitted, timestamps, overwrite

        @timer
        def _group(text_infos):
            def _is_recent_enough(sent):
                if not overwrite and sent in sent_mapping: return True
                
                last_timestamp  = timestamps.get(sent, 0)
                current_timestamp   = sent_mapping.get(sent, {}).get('timestamp', 0)
                logging.debug('Notified (elapsed time {}) for {}'.format(
                    current_timestamp - last_timestamp, sent
                ))
                return last_timestamp < current_timestamp
            
            if text_infos is None: return None
            text, splitted, timestamps, overwrite = text_infos
            
            infos = text_mapping.get(text, {}).copy()
            
            if splitted == infos.get('splitted', None) and all(infos.get(k + 's', None) is not None for k in required_keys) and all(_is_recent_enough(s) for s in splitted):
                logging.debug('Not updating {}'.format(text))
                return (text, infos, False)
            
            infos['splitted'] = splitted
            for i, sent in enumerate(splitted):
                logging.debug('Waiting for {}'.format(sent))
                sent_mapping.wait_for(sent, cond = lambda: _is_recent_enough(sent))
                logging.debug('Finish waiting for {}'.format(sent))
                for k, v in sent_mapping[sent].items():
                    if i == 0: infos[k + 's'] = []
                    infos[k + 's'].append(v)

            return (text, infos, True)
        
        def _save_groupped(output):
            def _is_json_data(v):
                if isinstance(v, list): return all(_is_json_data(vi) for vi in v)
                if isinstance(v, (np.ndarray, tf.Tensor)): return False
                return True
            
            if output is None: return None
            text, infos, updated = output

            if updated:
                logging.debug('Updating {} !'.format(text))
                if map_file:
                    infos   = {
                        k : v for k, v in infos.items() if _is_json_data(v)
                    }
                text_mapping[text] = {** text_mapping.get(text, {}), ** infos}
                
                if map_file:
                    dump_json(map_file, text_mapping, indent = 4)
            else:
                logging.debug('Notifying {} !'.format(text))
                text_mapping.notify_all(text)
            
            return output
        
        if max_text_length <= 0: max_text_length = self.max_input_length
        if required_keys is None: required_keys = []
        if not isinstance(required_keys, (list, tuple)): required_keys = [required_keys]
        map_file    = None
        if save:
            if directory is None: directory = self.pred_dir
            map_file = os.path.join(directory, 'map.json')
        
        if save_parts is None: save_parts = save
        # load previous generated (if any)
        text_mapping    = load_json(map_file, default = {}) if map_file else {}
        text_mapping    = ThreadedDict(** text_mapping)
        
        sent_mapping, pipeline  = self.get_pipeline(
            save        = save_parts,
            directory   = directory,
            max_workers = pipeline_workers,
            required_keys   = required_keys,
            ** kwargs
        )
        pipeline    = Pipeline(pipeline, buffer = buffer)
        pipeline.start()
        
        sentence_splitter   = Consumer(
            _sentence_splitter, max_workers = max_workers, buffer = buffer
        )
        pipeline.add_listener(lambda: sentence_splitter.stop, on = 'stop')
        sentence_splitter.add_listener(lambda: pipeline.stop, on = 'stop')
        sentence_splitter.start()
        
        grouper = sentence_splitter.add_consumer(
            _group, start = True, link_stop = True, max_workers = max_workers
        )
        if post_group is not None:
            grouper = grouper.add_consumer(
                post_group, start = True, link_stop = True, max_workers = max_workers
            )
        grouper.add_consumer(
            _save_groupped, start = True, link_stop = True, max_workers = min(max_workers, 1)
        )
        
        return sentence_splitter, grouper, pipeline, text_mapping, sent_mapping
    
    @timer
    def predict(self, sentences, overwrite = False, ** kwargs):
        producer, _, _, results, _  = self.get_streaming_pipeline(** kwargs)
        
        for sent in sentences: producer(sent, overwrite = overwrite)
        
        producer.join(recursive = True)

        return [(p, results[p]) for p in sentences]

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
