import os
import time
import random
import numpy as np
import tensorflow as tf

from tqdm import tqdm

from models.base_model import BaseModel
from custom_architectures import get_architecture
from models.weights_converter import partial_transfer_learning, pt_convert_model_weights
from utils import load_json, dump_json, time_to_string, plot_spectrogram, pad_batch
from utils.audio import MelSTFT, load_audio, load_mel
from utils.text import TextEncoder, split_text, default_english_encoder, get_symbols

DEFAULT_MAX_MEL_LENGTH  = 1024
DEFAULT_MAX_TEXT_LENGTH = 150

DEFAULT_MEL_FN_CONFIG  = {
    'filter_length'    : 1024,
    'hop_length'       : 256, 
    'win_length'       : 1024,
    'n_mel_channels'   : 80, 
    'sampling_rate'    : 22050, 
    'mel_fmin'         : 0.0,
    'mel_fmax'         : 8000.0,
    'normalize_mode'   : None,
}

class Tacotron2(BaseModel):
    def __init__(self,
                 lang,
                 
                 max_input_length   = DEFAULT_MAX_TEXT_LENGTH,
                 max_output_length  = DEFAULT_MAX_MEL_LENGTH,
                 
                 text_encoder       = None,
                 
                 mel_fn_type        = 'TacotronSTFT',
                 mel_fn_config      = DEFAULT_MEL_FN_CONFIG,
                 
                 ** kwargs
                ):
        self.lang   = lang
        self.max_input_length   = max_input_length
        self.max_output_length  = max_output_length
        
        # Initialization of Text Encoder
        if text_encoder is None: text_encoder = {}
        if isinstance(text_encoder, dict):
            text_encoder['use_sos_and_eos'] = False
            if 'vocab' not in text_encoder:
                text_encoder['vocab'] = get_symbols(lang, arpabet = False)
                text_encoder['word_level'] = False
            else:
                text_encoder.setdefault('word_level', False)
            text_encoder.setdefault('cleaners', ['french_cleaners'] if lang == 'fr' else ['english_cleaners'])
            self.text_encoder = TextEncoder(** text_encoder)
        
        elif isinstance(text_encoder, str):
            self.text_encoder = TextEncoder.load_from_file(text_encoder)
        elif isinstance(text_encoder, TextEncoder):
            self.text_encoder = text_encoder
        else:
            raise ValueError("input encoder de type inconnu : {}".format(text_encoder))
        
        
        # Initialization of mel fn
        if isinstance(mel_fn_type, MelSTFT):
            self.mel_fn = mel_fn_type
        else:
            self.mel_fn    = MelSTFT.create(mel_fn_type, ** mel_fn_config)

        super().__init__(**kwargs)
                
        # Saving text encoder and mel fn (if needed)
        if not os.path.exists(self.text_encoder_file):
            self.text_encoder.save_to_file(self.text_encoder_file)
        if not os.path.exists(self.mel_fn_file):
            self.mel_fn.save_to_file(self.mel_fn_file)
        
        if hasattr(self.tts_model, '_build'): self.tts_model._build()
        if hasattr(self.tts_model, 'set_step'): self.tts_model.set_step(self.steps)
    
    def _init_folders(self):
        super()._init_folders()
        os.makedirs(self.mel_dir,  exist_ok = True)
        os.makedirs(self.plot_dir, exist_ok = True)
    
    def init_train_config(self,
                          max_input_length  = None,
                          max_output_length = None,
                          
                          max_train_frames = -1,
                          pad_to_multiple  = False,
                          augment_prct     = 0.25,
                          
                          trim_audio   = False,
                          reduce_noise = False,
                          trim_threshold   = 0.1,
                          max_silence  = 0.15,
                          trim_method  = 'window',
                          trim_mode    = 'start_end',
                          
                          trim_mel     = False,
                          trim_factor  = 0.6,
                          trim_mel_method  = 'max_start_end',
                          ** kwargs
                          ):
        if max_train_frames > 0:
            raise NotImplementedError("max_train_frames > 0 not working yet !")
        if pad_to_multiple and max_train_frames <= 0:
            raise ValueError("If pad_to_multiple is True, max_train_frames must be > 0 !")
        
        if max_input_length: self.max_input_length   = max_input_length
        if max_output_length: self.max_output_length  = max_output_length
        
        self.max_train_frames   = tf.cast(max_train_frames, tf.int32)
        self.pad_to_multiple    = pad_to_multiple
                
        self.trim_audio     = trim_audio
        self.trim_kwargs    = {
            'trim_silence'  : trim_audio,
            'reduce_noise'  : reduce_noise,
            'method'    : trim_method,
            'mode'      : trim_mode,
            'threshold' : tf.cast(trim_threshold, tf.float32),
            'max_silence'   : tf.cast(max_silence, tf.float32)
        }
        
        self.trim_mel       = trim_mel
        self.trim_factor    = tf.cast(trim_factor, tf.float32)
        self.trim_mel_method    = trim_mel_method if trim_mel else None
        
        super().init_train_config(** kwargs)
        
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
        return (
            tf.TensorSpec(shape = (None, None), dtype = tf.int32),
            tf.TensorSpec(shape = (None,), dtype = tf.int32),
            tf.TensorSpec(shape = (None, None, self.n_mel_channels), dtype = tf.float32),
            tf.TensorSpec(shape = (None,), dtype = tf.int32),
        )
    
    @property
    def output_signature(self):
        return (
            tf.TensorSpec(shape = (None, None, self.n_mel_channels),dtype = tf.float32),
            tf.TensorSpec(shape = (None, None), dtype = tf.float32)
        )
        
    @property
    def training_hparams(self):
        return super().training_hparams(
            max_input_length    = None,
            max_output_length   = None,
            
            max_train_frames = -1,
            pad_to_multiple  = False,
            
            trim_audio   = False,
            reduce_noise = False,
            trim_threshold   = 0.1,
            max_silence  = 0.15,
            trim_method  = 'window',
            trim_mode    = 'start_end',

            trim_mel     = False,
            trim_factor  = 0.6,
            trim_mel_method  = 'max_start_end'
        )
    
    @property
    def text_encoder_file(self):
        return os.path.join(self.save_dir, 'text_encoder.json')
    
    @property
    def mel_fn_file(self):
        return os.path.join(self.save_dir, 'mel_fn.json')
    
    @property
    def audio_rate(self):
        return self.mel_fn.sampling_rate
    
    @property
    def n_mel_channels(self):
        return self.mel_fn.n_mel_channels
    
    @property
    def vocab(self):
        return self.text_encoder.vocab

    @property
    def vocab_size(self):
        return self.text_encoder.vocab_size
                
    @property
    def blank_token_idx(self):
        return self.text_encoder.blank_token_idx

    @property
    def go_frame(self):
        return tf.zeros((1, self.n_mel_channels), dtype = tf.float32)
    
    def __str__(self):
        des = super().__str__()
        des += "Input language : {}\n".format(self.lang)
        des += "Input vocab (size = {}) : {}\n".format(self.vocab_size, self.vocab)
        des += "Audio rate : {}\n".format(self.audio_rate)
        des += "Mel channels : {}\n".format(self.n_mel_channels)
        return des
    
    def call(self, inputs, training = False, **kwargs):
        pred = self.tts_model(inputs, training = training, **kwargs)
        return pred if len(pred) != 2 else pred[0]
    
    def infer(self, text, text_length = None, * args, ** kwargs):
        if isinstance(text, str):
            text = tf.expand_dims(self.encode_text(text), axis = 0)
            text_length = [len(text[0])]
        assert text_length is not None
        
        pred = self.tts_model.infer(text, text_length, * args, ** kwargs)
        return pred if len(pred) != 2 else pred[0]
    
    def compile(self, loss = 'tacotronloss', metrics = [], **kwargs):
        super().compile(loss = loss, metrics = metrics, ** kwargs)
    
    def encode_text(self, phrase):
        if isinstance(phrase, tf.Tensor):
            phrase = phrase.numpy()
            if isinstance(phrase, (list, np.ndarray)):
                phrase = [p.decode('utf-8') for p in phrase]
            else:
                phrase = phrase.decode('utf-8')
        elif isinstance(phrase, bytes):
            phrase = phrase.decode('utf-8')
            
        return self.text_encoder.encode(phrase)
    
    def decode_text(self, encoded):
        if isinstance(encoded, tf.Tensor): encoded = encoded.numpy()
        return self.text_encoder.decode(encoded)
        
    def get_mel_input(self, data):
        mel = load_mel(
            data, self.mel_fn, trim_mode = self.trim_mel_method,
            min_factor = self.trim_factor, ** self.trim_kwargs
        )

        mel = tf.concat([self.go_frame, mel], axis = 0)
        
        gate = tf.zeros((tf.shape(mel)[0] - 1,), dtype = tf.float32)
        gate = tf.concat([gate, tf.ones((1,), dtype = tf.float32)], axis = 0)
        
        return mel, gate
    
    def encode_data(self, data):
        text = data['text']
        encoded_text = tf.py_function(self.encode_text, [text], Tout = tf.int32)
        encoded_text.set_shape([None])
        
        mel, gate = self.get_mel_input(data)
        
        return encoded_text, len(encoded_text), mel, len(mel), mel, gate
        
    def filter_data(self, text, text_length, mel_input, mel_length, mel_output, gate):
        if self.max_train_frames > 0: return True
        return tf.logical_and(
            text_length <= self.max_input_length, 
            mel_length <= self.max_output_length
        )
    
    def augment_mel(self, mel):
        return tf.cond(
            tf.random.uniform(()) < self.augment_prct,
            lambda: mel + tf.random.normal(tf.shape(mel)),
            lambda: mel
        )
        
    def augment_data(self, text, text_length, mel_input, mel_length, mel_output, gate):
        mel_input = self.augment_mel(mel_input)
        
        return text, text_length, mel_input, mel_length, mel_output, gate
    
    def preprocess_data(self, text, text_length, mel_input, mel_length, mel_output, gate):
        if self.pad_to_multiple and self.max_train_frames > 0:
            reste = tf.shape(gate)[1] % self.max_train_frames
            padding = self.max_train_frames - reste + 1
            
            if padding > 0:
                mel_input   = tf.pad(mel_input, [(0,0), (0,padding), (0,0)])
                mel_output  = tf.pad(mel_output, [(0,0), (0,padding), (0,0)])
                gate        = tf.pad(gate, [(0,0), (0, padding)], constant_values = 1.)
        
        mel_input   = mel_input[:, :-1]
        mel_output  = mel_output[:, 1:]
        gate        = gate[:, 1:]
        
        return (text, text_length, mel_input, mel_length), (mel_output, gate)
    
    def get_dataset_config(self, **kwargs):
        kwargs['pad_kwargs']    = {
            'padded_shapes'     : (
                (None,), (), (None, self.n_mel_channels), (),
                (None, self.n_mel_channels), (None,)
            ),
            'padding_values'    : (self.blank_token_idx, 0, 0., 0, 0., 1.)
        }
        kwargs['batch_before_map']  = True
        kwargs['padded_batch']      = True
        
        return super().get_dataset_config(**kwargs)
        
    def train_step(self, batch):
        inputs, target = batch

        loss_fn     = self.tts_model_loss
        optimizer   = self.tts_model_optimizer
        variables   = self.tts_model.trainable_variables
        
        with tf.GradientTape() as tape:
            pred = self(inputs, training = True)
            losses = loss_fn(target, pred)
            loss = losses[0]

        gradients = tape.gradient(loss, variables)
                
        optimizer.apply_gradients(zip(gradients, variables))
        
        return self.update_metrics(target, pred)
    
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

        
    def predict(self,
                sentences,
                max_text_length = -1,
                
                directory   = None,
                batch_size  = 16,
                
                save    = True,
                overwrite = False,
                
                tqdm    = lambda x: x,
                debug   = False,
                
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
        
        t_process, t_infer, t_save = time.time(), 0., 0.
        
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
            p for p in sentences if overwrite or p not in infos_pred
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
        encoded = [self.encode_text(p) for p in flattened]
        
        t_process = time.time() - t_process
        
        # for each batch
        num = 0
        for start_idx in tqdm(range(0, len(encoded), batch_size)):
            start_process = time.time()
            
            batch   = encoded[start_idx : start_idx + batch_size]
            lengths = np.array([len(part) for part in batch])
            
            padded_inputs = pad_batch(batch, 0, dtype = np.int32)
            
            lengths     = tf.cast(lengths, tf.int32)
            padded_inputs   = tf.cast(padded_inputs, tf.int32)
            
            start_infer = time.time()
            t_process += start_infer - start_process
            
            _, mels, gates, attn_weights = self.infer(
                text = padded_inputs, text_length = lengths, ** kwargs
            )
            
            t_infer += time.time() - start_infer
            
            mels, gates, attn_weights = mels.numpy(), gates.numpy(), attn_weights.numpy()
            
            for mel, gate, attn in zip(mels, gates, attn_weights):
                stop_gate = np.where(gate > 0.5)[0]
                mel_length = stop_gate[0] if len(stop_gate) > 0 else len(gate)
                mel = mel[ : mel_length, :]
                attn = attn[ : mel_length, :]
                
                text = sentences_to_read[index[num]]
                
                if index_part[num] == 0:
                    infos_pred.setdefault(text, {
                        'splitted' : [], 'mels' : [], 'plots' : []
                    })
                    infos_pred[text]['splitted'] = []

                infos_pred[text]['splitted'].append(flattened[num])
                
                start_save = time.time()
                
                if save:
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
                else:
                    if index_part[num] == 0:
                        infos_pred[text].update({
                            'mels' : [], 'plots' : []
                        })
                    
                    infos_pred[text]['mels'].append(mel)
                
                t_save += time.time() - start_save
                num += 1
        
        if save:
            start_save = time.time()
            dump_json(map_file, infos_pred, indent = 4)
            t_save += time.time() - start_save
        
        if debug:
            print("Total time : {}\n- Processing time : {}\n- Inference time : {}\n- Saving time : {}".format(
                time_to_string(t_process + t_infer + t_save),
                time_to_string(t_process),
                time_to_string(t_infer),
                time_to_string(t_save)
            ))
        
        return [(p, infos_pred[p]) for p in sentences]
                                    
    def get_config(self, *args, **kwargs):
        config = super().get_config(*args, **kwargs)
        config['lang']      = self.lang
        
        config['max_input_length']  = self.max_input_length
        config['max_output_length'] = self.max_output_length
        
        config['text_encoder']      = self.text_encoder_file
        config['mel_fn_type']       = self.mel_fn_file
        
        return config
    
    @classmethod
    def build_from_pretrained(cls, 
                              nom,
                              pretrained_name,
                              new_lang      = None,
                              new_encoder   = None,
                              ** kwargs
                             ):
        with tf.device('cpu') as device:        
            pretrained_model = Tacotron2(nom = pretrained_name)
        
        config = {
            'nom'   : nom,
            'lang'  : pretrained_model.lang if new_lang is None else new_lang,
            'text_encoder'  : pretrained_model.text_encoder if new_encoder is None else new_encoder
        }
        config = {** kwargs, ** config}
        
        instance = cls(** config)

        partial_transfer_learning(instance.tts_model, pretrained_model.tts_model)
                
        instance.save()
        
        return instance

    @classmethod
    def build_from_nvidia_pretrained(cls, 
                                     nom    = 'pretrained_tacotron2', 
                                     new_lang      = None,
                                     new_encoder   = None,
                                     ** kwargs
                                    ):            
        config = {
            'nom'   : nom,
            'lang'  : 'en' if new_lang is None else new_lang,
            'text_encoder'  : default_english_encoder() if new_encoder is None else new_encoder
        }
        config = {** kwargs, ** config}
        
        with tf.device('cpu') as device:
            instance = cls(** config)
        
        nvidia_model = get_architecture('nvidia_tacotron', to_gpu = False)
        
        pt_convert_model_weights(nvidia_model, instance.tts_model)
        
        instance.save()
        
        return instance
