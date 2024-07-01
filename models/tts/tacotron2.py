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
import pandas as pd

from loggers import timer, time_logger
from custom_architectures import get_architecture
from models.interfaces.base_text_model import BaseTextModel
from models.interfaces.base_audio_model import BaseAudioModel
from utils.keras_utils import TensorSpec, ops
from utils import time_to_string, load_json, dump_json, convert_to_str, should_predict, plot_spectrogram, pad_batch
from utils.audio import write_audio, load_audio, display_audio, play_audio
from utils.text import default_english_encoder, split_text

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
        if not tts_model:
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
    
    def call(self, inputs, training = False, ** kwargs):
        pred = self.tts_model(inputs, training = training, ** kwargs)
        return pred if len(pred) != 2 else pred[0]
    
    @timer(name = 'inference')
    def infer(self,
              text,
              * args,
              
              attn_mask_win_len = None,

              max_length    = -1,
              early_stopping    = True,
              
              return_state  = False,
              initial_state = None,
              
              max_trial = 5,
              min_fpt_ratio = -1,
              max_fpt_ratio = -1,
              
              ** kwargs
             ):
        if max_length <= 0: max_length = self.max_output_length
        
        if not isinstance(text, (list, tuple)):
            if isinstance(text, str):
                text    = ops.expand_dims(self.encode_text(text), axis = 0)
            elif len(ops.shape(text)) == 1:
                text    = ops.expand_dims(text, axis = 0)
        
        seq_len = text.shape[1] if not isinstance(text, (list, tuple)) else text[0].shape[1]
        
        trial = 0
        while trial < max_trial:
            outputs = self.tts_model.infer(
                text,
                attn_mask_win_len   = attn_mask_win_len,
            
                max_length  = max_length,
                early_stopping  = early_stopping,
                initial_state   = initial_state,
                return_state    = return_state,
                ** kwargs
            )
            
            ratio = ops.max(outputs.lengths) / seq_len
            if (min_fpt_ratio == -1 or ratio > min_fpt_ratio) and (max_fpt_ratio == -1 or ratio < max_fpt_ratio):
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

        gate = ops.zeros((ops.shape(mel)[0] - 1,), dtype = 'float32')
        gate = ops.concatenate([gate, ops.ones((1,), dtype = 'float32')], axis = 0)
        
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
    
    def _process_data(self, inputs, outputs):
        mel_input, mel_length   = inputs[-2:]
        mel_output, gate        = outputs
        
        if self.pad_to_multiple and self.max_train_frames > 0:
            to_pad  = ops.shape(gate)[1] % self.max_train_frames
            padding = self.max_train_frames - to_pad + 1
            
            if padding > 0:
                mel_input   = ops.pad(
                    mel_input, [(0, 0), (0, padding), (0, 0)], constant_values = self.pad_mel_value
                )
                mel_output  = ops.pad(
                    mel_output, [(0, 0), (0, padding), (0, 0)], constant_values = self.pad_mel_value
                )
                gate        = ops.pad(gate, [(0, 0), (0, padding)], constant_values = 1.)
        
        return inputs[:-2] + (mel_input, mel_length), (mel_output, gate)
    
    def get_dataset_config(self, * args, ** kwargs):
        kwargs.update({
            'pad_kwargs'    : {
                'padding_values'    : (
                    (self.blank_token_idx, self.pad_mel_value, 0), (self.pad_mel_value, 1.)
                )
            }
        })
        
        return super().get_dataset_config(* args, ** kwargs)
    
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
        infer = self.infer(* inputs[:-2], max_trial = 1)

        _, pred_mel, _, pred_attn = [p.numpy() for p in pred]
        infer_mel, infer_gate, infer_attn = [i.numpy() for i in infer]
        
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

    @timer(name = 'text encoding')
    def split_and_encode(self, cleaned_text, max_text_length):
        if isinstance(cleaned_text, list): splitted = cleaned_text
        elif max_text_length == -1:        splitted = [cleaned_text]
        elif max_text_length == -2:        splitted = split_sentence(cleaned_text)
        else:                              splitted = split_text(cleaned_text, max_text_length)

        splitted    = [s for s in splitted if any(c.isalnum() for c in s)]
        encoded     = [self.encode_text(text, cleaned = True) for text in splitted]
        
        splitted    = [splitted[i] for i in range(len(splitted)) if len(encoded[i]) > 0]
        encoded     = [enc for enc in encoded if len(enc) > 0]
        return splitted, encoded

    @timer
    def predict(self,
                texts,
                max_text_length = -1,
                max_length  = 10.,
                batch_size  = 1,
                use_cleaned_as_key  = True,
                
                min_fpt_ratio = 2.,
                max_fpt_ratio = 10.,
                
                vocoder = None,
                display = None,
                play    = False,
                show_plot   = False,
                silence_time    = 0.15,
                vocoder_config  = {},
                
                save    = None,
                save_mel    = None,
                save_plot   = False,
                save_audio  = None,
                directory   = None,
                overwrite   = False,
                timestamp   = -1,
                required_key    = None,
                
                filename    = 'audio_{}.mp3',
                
                cleaners    = {},
                expand_acronyms = True,
                
                post_processing      = None,
                part_post_processing = None,
                
                tqdm    = lambda x: x,
                
                ** kwargs
               ):
        """
            Perform Tacotron-2 inference on all sentences
            
            Arguments :
                - texts     : (list of) string, the sentence(s) to read.
                - max_text_length   : the maximum length of a sentence
                    - -1    : no split
                    - -2    : sentence-based split (basic punctuation-based split)
                    - > 0   : splits the sentence such that each part is shorted than `max_length`
                
                - max_length    : maximal length of a predicted spectrogram (float means a length relative to the encoded text length)
                - batch_size    : the number of sentences to infer in parallel
                    /!\ if `vocoder` is provided, the vocoder batch_size is forced to 1
                
                - vocoder   : a function that takes a `mel-spectrogram` as input and returns an audio 
                - display   : whether to display the audio (only relevant in notebooks)
                - play      : whether to play the audio
                    /!\ if `display is True`, it will use the `autoplay` of the `Audio` widget
                - silence_time  : the time of silence to add between each part of a splitted sentence (only relevant if `max_text_length != -1`)
                
                - save      : whether to save the results
                - save_mel  : whether to save the produced mel-spectrogram
                - save_plot : whether to save plots of the mel-spectrogram + attention (+ audio)
                - save_audio    : whether to save the produced audio (only relevant if `vocoder` is provided)
                - show_plot : whether to show the produced plots
                
                - directory : the directory in which the produced mapping file + files will be saved
                - overwrite : whether to overwrite if a sentence has already been predicted
                - timestamp : if provided and `overwrite = True`, only overwrites if `timestamp` is smaller than the timestamp of the prediction (i.e. the time at which it has been generated)
                - required_key  : this key is expected to be in the resulting output (see Notes)
                
                - filename  : the audio filename format (must contain a '{}' if multiple sentences)
                
                - expand_acronyms   : kwarg for the `self.clean_text` method
                
                - post_processing   : callback function applied after each sentence prediction, with signature `fn(infos : Dict, text : List[str] = splitted)`
                - part_post_processing  : callback function applied after each part of a splitted sentence with signature : `fn(text = splitted, mel = mel, attention = attn, audio = audio)`
                
                - tqdm  : progress bar
                
                - kwargs    : forwarded to `self.infer` and `vocoder`
            Returns :
                - results   : a list of tuple (sentence, infos)
                    - sentences : the sentence read
                    - infos     : a `dict` containing the information about the sentence
                        - text      : the original text
                        - parts     : the splitted version of the sentence
                        - timestamp : the time at which the sentence has been read by the model
                        - mel       : the global mel-spectrograms or its filename (if `save_mel`)
                        - plot  (if `save_plot`)    : the plot filenames
                        - audio (if `vocoder`)  : the global audio or its filename (if `save_audio`)
            
            Note : in the returned `infos`, the `mels`, `audios` andàudio  may contain either a string (the filename), either the raw mel / audio, depending on the `save_{mel / audio}` value. 
            Note that only the filenames are saved in the mapping file : if `save_mel = False` and `save_audio = True`, the mapping file will only contain the `audios` and `audio` keys (not `mels`) while, in the result, `mels` will be a list of np.ndarray
            
            If the sentence was already generated and is not overwritten, it is possible that some keys will not be in the output, only the previously saved ones will be there. 
            However, if `required_key` was not saved, the sentence is overwritten, even if `overwrite = False`, in order to have the expected key in the output result
        """
        ####################
        #  Initialization  #
        ####################
        
        with time_logger.timer('initialization'):
            now = time.time()

            if isinstance(texts, pd.DataFrame): texts = texts.to_dict('record')
            elif not isinstance(texts, (list, tuple, np.ndarray)) or ops.is_tensor(texts):
                texts = [texts]

            if not required_key:    required_key    = 'mel' if vocoder is None else 'audio'

            if save is None:        save = True if directory or vocoder is None else False
            if save_mel is None:    save_mel    = save
            if save_plot is None:   save_plot   = save
            if vocoder is None:     save_audio  = False
            elif save_audio is None:save_audio  = save
            if display is None:     display     = not save
            if len(texts) > 1 and play: play    = 2

            save = save_mel or save_plot or save_audio
            if vocoder is not None:
                if save:    save_audio = True
                elif not play:  display = True

            # get saving directory
            if directory is None: directory = self.pred_dir
            mel_dir, plot_dir, audio_dir = None, None, None

            map_file = os.path.join(directory, 'map.json')

            keys_to_skip = [
                'attention', 'mel' if not save_mel else None, 'audio' if not save_audio else None
            ]
            if save:
                if save_mel:
                    mel_dir = '/'.join((directory, 'mels'))
                    os.makedirs(mel_dir, exist_ok = True)

                if save_plot:
                    plot_dir = '/'.join((directory, 'plots'))
                    os.makedirs(plot_dir, exist_ok = True)

                if save_audio:
                    audio_dir = '/'.join((directory, 'audios'))
                    os.makedirs(audio_dir, exist_ok = True)
            # load previous generated (if any)
            predicted = load_json(map_file, default = {})

        ####################
        #  Pre-processing  #
        ####################

        with time_logger.timer('text cleaning'):
            results     = [None] * len(texts)
            duplicatas  = {}
            requested   = [(_get_text(txt), txt) for txt in texts]

            inputs, cleaned = [], {}
            for i, (text, data) in enumerate(requested):
                if isinstance(text, str):
                    if text not in cleaned:
                        cleaned[text] = self.clean_text(
                            text, to_expand_acronyms = expand_acronyms, ** cleaners
                        )

                    clean = cleaned[text]
                    key   = clean if use_cleaned_as_key else text
                    if not should_predict(predicted, key, overwrite = overwrite, timestamp = timestamp, required_keys = (required_key, )):
                        results[i] = (text, predicted[key])
                        continue

                    duplicatas.setdefault(key, []).append((text, i))
                    if len(duplicatas[key]) > 1:
                        continue
                else:
                    clean = [self.clean_text(
                        t, to_expand_acronyms = expand_acronyms, ** cleaners
                    ) for t in text]

                inputs.append((i, text, clean, data))
        
        ####################
        #  Inference loop  #
        ####################
        
        show_idx = post_process(
            results, 0, display, play, post_processing, rate = self.audio_rate, ** kwargs
        )
        
        if len(inputs) > 0:
            silence = np.zeros((int(self.audio_rate * silence_time), ))
            
            for (i, text, clean, data) in inputs:
                key   = clean if use_cleaned_as_key else text
                if not isinstance(key, str): key = ' '.join(key)
                
                splitted, encoded = self.split_and_encode(clean, max_text_length)
                
                synth_gen_time, vocoder_gen_time = 0., 0.
                
                all_outputs = {'mels' : [], 'attentions' : [], 'audios' : []}
                for s in range(0, len(encoded), batch_size):
                    batch = encoded[s : s + batch_size]
                    batch = ops.expand_dims(batch[0], axis = 0) if len(batch) <= 1 else ops.cast(
                        pad_batch(batch, pad_value = self.blank_token_idx), 'int32'
                    )

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
                        
                        audio = None
                        if vocoder is not None:
                            start_vocoder_time = time.time()
                            
                            audio = vocoder(mel, ** {** kwargs, ** vocoder_config})
                            if len(audio.shape) == 2: audio = audio[0]
                            all_outputs['audios'].append(audio)
                            
                            vocoder_gen_time += time.time() - start_vocoder_time
                        
                        all_outputs['mels'].append(mel)
                        all_outputs['attentions'].append(attn)
                        
                        if part_post_processing is not None:
                            part_post_processing(
                                text = splitted[s + idx], mel = mel, attention = attn, audio = audio
                            )
                
                mel, attention, audio = _combine_outputs(** all_outputs, _silence = silence)
                
                audio_time = (len(audio) / self.audio_rate) if vocoder is not None else -1
                infos = {
                    'text'  : text,
                    'cleaned'   : clean,
                    'parts' : splitted,
                    'timestamp' : now,
                    'mel'   : mel,
                    'attention' : attention,
                    'audio' : audio,
                    'time'  : audio_time
                }
                
                if vocoder is not None and len(mel) > 0:
                    logger.info('{} generated in {} ({} generated / sec) : {} synthesizer + {} vocoder'.format(
                        time_to_string(audio_time),
                        time_to_string(synth_gen_time + vocoder_gen_time),
                        time_to_string(audio_time / (synth_gen_time + vocoder_gen_time)),
                        time_to_string(synth_gen_time),
                        time_to_string(vocoder_gen_time)
                    ))
                
                if save:
                    if save_mel and len(mel) > 0:
                        with time_logger.timer('saving mel'):
                            infos['mel'] = _get_filename(
                                predicted, key, 'mel', mel_dir, 'mel_{}.npy'
                            )
                            np.save(infos['mel'], mel)
                    
                    if save_plot and len(mel) > 0:
                        with time_logger.timer('saving plot'):
                            infos['plot'] = _get_filename(
                                predicted, key, 'plot', plot_dir, 'plot_{}.png'
                            )
                            data_to_plot = {'mel' : mel}
                            if len(attention) > 1:
                                data_to_plot.update({
                                    'Attention part #{}'.format(i + 1) : attn
                                    for i, attn in enumerate(attention)
                                })
                            else:
                                data_to_plot['attention'] = attention[0]

                            if vocoder is not None:
                                data_to_plot['audio'] = {
                                    'x' : audio, 'rate' : self.audio_rate, 'plot_type' : 'audio'
                                }

                            plot_spectrogram(
                                ** data_to_plot, title = "Text :\n{}".format(text), filename = infos['plot'], show = show_plot
                            )
                    
                    if save_audio:
                        with time_logger.timer('saving audio'):
                            infos['audio'] = _get_filename(
                                predicted, key, 'audio', audio_dir, filename, is_silence = len(mel) == 0
                            )
                            write_audio(
                                audio = audio, filename = infos['audio'], rate = self.audio_rate
                            )
                    
                    if len(mel) > 0:
                        predicted[key] = {k : v for k, v in infos.items() if k not in keys_to_skip}

                        with time_logger.timer('saving json'):
                            dump_json(filename = map_file, data = predicted, indent = 4)
                
                if isinstance(key, str):
                    for original_txt, duplicate_idx in duplicatas[key]:
                        results[duplicate_idx] = (original_txt, {** infos, 'text' : original_txt})
                else:
                    results[i] = (text, infos)
                
                show_idx = post_process(
                    results, show_idx, display, play, post_processing, rate = self.audio_rate,
                    ** kwargs
                )
        
        return results

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

def _get_text(text):
    return convert_to_str(text['text'] if isinstance(text, (dict, pd.Series)) else text)

def _get_filename(predicted, text, key, directory, file_format, is_silence = False):
    filename = predicted.get(text, {}).get(key, file_format)
    if directory and not filename.startswith(directory):
        filename = os.path.join(directory, filename)
    if '{}' in filename:
        filename = filename.format(
            len(glob.glob(filename.replace('{}', '*'))) if not is_silence else 'silence'
        )
    
    return filename

def _combine_outputs(mels, attentions, audios, _silence):
    if not mels: return np.zeros((0, 0)), [], _silence
    audio = None
    if audios:
        audio = [audios[0]]
        for a in audios[1:]: audio.extend([_silence, a])
        audio = np.concatenate(audio, axis = 0)
    
    return np.concatenate(mels, axis = 0), attentions, audio

@timer(name = 'post-processing')
def post_process(results, idx, display, play, post_processing, rate, ** kwargs):
    while idx < len(results) and results[idx] is not None:
        text, infos = results[idx]
        
        if 'audio' in infos:
            if display:
                logger.info('Text : "{}"'.format(text))
                display_audio(infos['audio'], rate = rate, play = play == 1)
            if (play) and (not display or play == 2):
                play_audio(infos['audio'], rate = rate, ** kwargs)
        
        if post_processing is not None:
            post_processing(infos, text = text)
            
        idx += 1
    
    return idx
