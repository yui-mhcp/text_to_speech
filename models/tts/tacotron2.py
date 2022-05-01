
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
from utils.text import default_english_encoder, split_text

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
    
    def _init_folders(self):
        super()._init_folders()
        os.makedirs(self.mel_dir,  exist_ok = True)
        os.makedirs(self.plot_dir, exist_ok = True)
    
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
    def pred_map_file(self):
        return os.path.join(self.pred_dir, 'map_file.json')
    
    @property
    def mel_dir(self):
        return os.path.join(self.pred_dir, 'mels')
    
    @property
    def plot_dir(self):
        return os.path.join(self.pred_dir, 'plots')
    
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
                text = tf.expand_dims(self.encode_text(text), axis = 0)
                text_length = tf.cast([len(text[0])], tf.int32)
            elif len(tf.shape(text)) == 1:
                text = tf.expand_dims(text, axis = 0)
        
        if text_length is None and tf.shape(text)[0] == 1:
            text_length = tf.cast([tf.shape(text)[1]], tf.int32)
        
        assert text_length is not None
        
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

        
    @timer
    def predict(self,
                sentences,
                max_text_length = -1,
                
                directory   = None,
                batch_size  = 16,
                
                save    = True,
                overwrite = False,
                
                expand_acronyms = True,
                
                tqdm    = lambda x: x,
                
                ** kwargs
               ):
        """
            Perform Tacotron-2 inference on all phrases
            Arguments :
                - sentences   : str or list of str, the sentences to infer. 
                - max_text_length   : maximum length of sentence to infer. Split sentences in order to have parts of at most 'max_text_length'. 
                - directory : where to save data (if 'save' is not False). 
                - batch_size    : batch_size for inference. 
                - save  : whether to save result or not. 
                - tqdm  : progress bar
                - kwargs    : kwargs passed to self.infer() method
            Return : list of tuple (phrase, infos)
                infos is a dict containing : 
                    - splitted  : list of sentence sub-parts. 
                    - mels      : list of filenames (if save) or list of mels
                    - plots     : list of plot filenames (if save) or empty list
        """
        if max_text_length <= 0: max_text_length = self.max_input_length
        
        time_logger.start_timer('initialization')

        # get saving directory
        if not save:
            mel_dir, plot_dir, map_file = None, None, None
        elif directory is None:
            mel_dir, plot_dir, map_file = self.mel_dir, self.plot_dir, self.pred_map_file
        else:
            mel_dir     = os.path.join(directory, 'mels')
            plot_dir    = os.path.join(directory, 'plots')
            map_file    = os.path.join(directory, 'map.json')
            
            os.makedirs(mel_dir, exist_ok = True)
            os.makedirs(plot_dir, exist_ok = True)
        
        # load previous generated (if any)
        infos_pred = {}
        if save and os.path.exists(map_file):
            infos_pred = load_json(map_file)
        
        if not isinstance(sentences, (list, tuple)): sentences = [sentences]
        # get unique sentences to read
        sentences_to_read = list(set([
            p for p in sentences if overwrite or infos_pred.get(p, {}).get('mels', None) is None
        ]))
        # split them according to the 'max_text_length' argument
        splitted = split_text(sentences_to_read, max_text_length)
        # save them in a flattened list and keep track of index for each part
        # it allows to remember which part was part of which sentence (by its index)
        flattened, index, index_part = [], [], []
        for i, p in enumerate(splitted):
            flattened   += p
            index       += [i] * len(p)
            index_part  += list(range(len(p)))
        # encoded part-sentences 
        encoded = [self.encode_text(p, to_expand_acronyms = expand_acronyms) for p in flattened]
        
        idx_to_keep = [i for i, enc in enumerate(encoded) if len(enc) > 0]
        flattened   = [flattened[idx] for idx in idx_to_keep]
        index       = [index[idx] for idx in idx_to_keep]
        index_part  = [index_part[idx] for idx in idx_to_keep]
        encoded     = [encoded[idx] for idx in idx_to_keep]
        
        time_logger.stop_timer('initialization')
        
        # for each batch
        num = 0
        for start_idx in tqdm(range(0, len(encoded), batch_size)):
            time_logger.start_timer('processing')
            
            batch   = encoded[start_idx : start_idx + batch_size]
            lengths = np.array([len(part) for part in batch])

            padded_inputs = pad_batch(batch, 0, dtype = np.int32)
            
            lengths     = tf.cast(lengths, tf.int32)
            padded_inputs   = tf.cast(padded_inputs, tf.int32)
            
            time_logger.stop_timer('processing')

            _, mels, gates, attn_weights = self.infer(
                text = padded_inputs, text_length = lengths, ** kwargs
            )
            
            mels, gates, attn_weights = mels.numpy(), gates.numpy(), attn_weights.numpy()
            
            for mel, gate, attn in zip(mels, gates, attn_weights):
                time_logger.start_timer('post processing')
                
                stop_gate   = np.where(gate > 0.5)[0]
                mel_length  = stop_gate[0] if len(stop_gate) > 0 else len(gate)

                mel         = mel[ : mel_length, :]
                attn        = attn[ : mel_length, :]
                
                text = sentences_to_read[index[num]]
                
                if index_part[num] == 0:
                    infos_pred.setdefault(text, {})
                    infos_pred[text]['splitted'] = []
                    infos_pred[text].setdefault('mels', [])
                    infos_pred[text].setdefault('plots', [])

                infos_pred[text]['splitted'].append(flattened[num])
                
                time_logger.stop_timer('post processing')
                
                if save:
                    time_logger.start_timer('saving')

                    num_pred = len(os.listdir(mel_dir))
                    if len(infos_pred[text]['mels']) > index_part[num]:
                        mel_filename = infos_pred[text]['mels'][index_part[num]]
                        plot_filename = infos_pred[text]['plots'][index_part[num]]
                    else:
                        mel_filename = os.path.join(
                            mel_dir, 'mel_{}.npy'.format(num_pred)
                        )
                        plot_filename = os.path.join(
                            plot_dir, 'plot_{}.png'.format(num_pred)
                        )
                        
                        infos_pred[text]['mels'].append(mel_filename)
                        infos_pred[text]['plots'].append(plot_filename)

                    np.save(mel_filename, mel)
                    plot_spectrogram(
                        spectrogram = mel, attention = attn, 
                        filename = plot_filename, show = False, 
                        title = "Spectrogramme pour :\n{}".format(text)
                    )
                    
                    time_logger.stop_timer('saving')
                else:
                    if index_part[num] == 0:
                        infos_pred[text].update({
                            'mels' : [], 'plots' : []
                        })
                    
                    infos_pred[text]['mels'].append(mel)
                
                num += 1
        
        if save:
            time_logger.start_timer('json saving')
            dump_json(map_file, infos_pred, indent = 4)
            time_logger.stop_timer('json saving')

        return [(p, infos_pred.get(p, {})) for p in sentences if p in infos_pred]

    def get_config(self, * args, ** kwargs):
        config = super().get_config(* args, ** kwargs)
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
