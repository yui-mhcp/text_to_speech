import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm

from models.siamese.siamese_network import SiameseNetwork
from custom_architectures import get_architecture
from utils import load_json, dump_json, normalize_filename
from utils.audio import MelSTFT, load_audio, load_mel, AudioAnnotation
from utils.audio import random_pad, random_shift, random_noise
from utils.distance import KPropagation

_supported_input_types      = ('raw', 'audio', 'mel', 'spect', 'spectrogram')
_supported_encoder_types    = ('rnn', 'conv1d', 'conv2d', 'transformer', 'prenet_transformer')

MIN_AUDIO_TIME      = 0.1 # below 0.1sec the encoding is not really relevant

DEFAULT_MEL_FRAMES      = 80
DEFAULT_AUDIO_RATE      = 16000
DEFAULT_MAX_AUDIO_TIME  = 3

DEFAULT_MEL_FN_CONFIG  = {
    'filter_length'    : 1024,
    'hop_length'       : 256, 
    'win_length'       : 1024,
    'n_mel_channels'   : 80, 
    'mel_fmin'         : 0.0,
    'mel_fmax'         : 8000.0,
    'normalize_mode'   : None,
}

class AudioSiamese(SiameseNetwork):
    def __init__(self,
                 audio_rate     = DEFAULT_AUDIO_RATE,
                 input_type     = 'mel',
                 
                 encoder_type       = 'conv1d',
                 max_audio_time     = DEFAULT_MAX_AUDIO_TIME,
                 use_fixed_length_input = False,
                 
                 mel_fn_type        = 'TacotronSTFT',
                 mel_fn_config      = DEFAULT_MEL_FN_CONFIG,
                 
                 ** kwargs
                ):        
        encoder_type = encoder_type.lower()
        assert input_type in _supported_input_types
        assert encoder_type in _supported_encoder_types
        
        self.audio_rate = audio_rate
        self.input_type = input_type
        self.encoder_type   = encoder_type
        
        self.max_audio_time     = max_audio_time
        self.use_fixed_length_input = use_fixed_length_input
        
        self.mel_fn = None
        if self.use_mel_fn:
            if isinstance(mel_fn_type, MelSTFT):
                self.mel_fn = mel_fn_type
            else:
                mel_fn_config['sampling_rate'] = audio_rate
                self.mel_fn    = MelSTFT.create(mel_fn_type, ** mel_fn_config)
        
        super().__init__(**kwargs)
        
        if self.use_mel_fn and not os.path.exists(self.mel_fn_file):
            self.mel_fn.save_to_file(self.mel_fn_file)
    
    def _init_folders(self):
        super()._init_folders()
        os.makedirs(self.identification_dir, exist_ok = True)

    def build_encoder(self, 
                      depth          = 128,
                      flatten_type   = 'max',
                      embedding_dim  = 128, 
                      flatten_kwargs = {},
                      **kwargs
                     ):
        flatten_kwargs.setdefault('units', embedding_dim)
        audio_encoder_config = {
            'architecture_name'     : 'simple_cnn', 
            'input_shape'   : self.encoder_input_shape[1:],
            'output_shape'  : embedding_dim,
            'n_conv'    : 5,
            'filters'   : [
                depth, 
                depth * 2,
                depth * 3, 
                [depth * 3, depth * 4],
                [depth * 4, depth * 4],
            ],
            'strides'   : [4, 4, 2, 2, 1],
            'kernel_size'   : [
                32, 32, 3, 3, [3, 1]
            ],
            'residual'      : True,
            'drop_rate'     : 0.05,
            'conv_type'     : self.encoder_type,
            'flatten'       : True,
            'dense_as_final'    : True,
            'name'  : 'Encoder'
        }
        voicemap_encoder_config = {
            'architecture_name'     : 'simple_cnn', 
            'input_shape'   : self.encoder_input_shape[1:],
            'output_shape'  : embedding_dim,
            'n_conv'    : 4,
            'filters'   : [
                depth, 
                depth * 2,
                depth * 3, 
                depth * 4
            ],
            'strides'   : [4, 1, 1, 1],
            'kernel_size'   : [
                32, 3, 3, 3
            ],
            'pooling'       : 'max',
            'pool_size'     : [4, 2, 2, 2],
            'pool_strides'  : [4, 2, 2, 2],
            'residual'      : False,
            'activation'    : 'relu',
            'drop_rate'     : 0.05,
            'conv_type'     : 'conv1d',
            'flatten'       : True,
            'flatten_type'  : flatten_type,
            'flatten_kwargs'    : flatten_kwargs,
            'dense_as_final'    : True,
            'name'  : 'Encoder',
            ** kwargs
        }
        mel_encoder_config = {
            'architecture_name'     : 'simple_cnn', 
            'input_shape'   : self.encoder_input_shape[1:],
            'output_shape'  : embedding_dim,
            'n_conv'    : 4,
            'filters'   : [
                depth, 
                depth * 2, 
                [depth * 2, depth * 4],
                [depth * 4, depth * 4],
            ],
            'strides'   : [2, 2, 1, 1],
            'kernel_size'   : [
                7, 5, 3, [3, 1]
            ],
            'residual'      : False,
            'conv_type'     : self.encoder_type,
            'flatten'       : True,
            'flatten_type'  : flatten_type,
            'flatten_kwargs'    : flatten_kwargs,
            'dense_as_final'    : flatten_type not in ('lstm', 'gru'),
            'name'  : 'Encoder',
            ** kwargs
        }
        
        encoder_config = mel_encoder_config if self.use_mel_fn else voicemap_encoder_config
        
        return get_architecture(** encoder_config)

    @property
    def identification_dir(self):
        return os.path.join(self.folder, 'identification')
    
    @property
    def mel_fn_file(self):
        return os.path.join(self.save_dir, 'mel_fn.json')

    @property
    def encoder_input_shape(self):
        length = None if not self.use_fixed_length_input else self.max_input_length
        
        if not self.use_mel_fn:
            shape = (None, length, 1)
        elif not self.mel_as_image:
            shape = (None, length, self.n_mel_channels)
        else:
            shape = (None, length, self.n_mel_channels, 1)
        return shape
    
    @property
    def min_input_length(self):
        if self.use_mel_fn:
            return self.mel_fn.get_length(int(MIN_AUDIO_TIME * self.audio_rate))
        else:
            return int(MIN_AUDIO_TIME * self.audio_rate)
        
    @property
    def max_input_length(self):
        if self.use_mel_fn:
            return self.mel_fn.get_length(int(self.max_audio_time * self.audio_rate))
        else:
            return int(self.max_audio_time * self.audio_rate)
                
    @property
    def use_mel_fn(self):
        return self.input_type not in ('audio', 'raw')
    
    @property
    def mel_as_image(self):
        return self.encoder_type in ('conv2d', 'prenet_transformer') and self.use_mel_fn
        
    @property
    def n_mel_channels(self):
        return self.mel_fn.n_mel_channels if self.use_mel_fn else -1
                
    def __str__(self):
        des = super().__str__()
        des += "Audio rate : {}\n".format(self.audio_rate)
        des += "Input type : {}\n".format(self.input_type)
        if self.use_mel_fn:
            des += "N mel channels : {}\n".format(self.n_mel_channels)
        return des
    
    def get_audio_input(self, data):
        audio = load_audio(data, self.audio_rate)
        
        return tf.expand_dims(audio, 1)
    
    def get_mel_input(self, data):
        mel = load_mel(data, self.mel_fn)
        
        if self.mel_as_image:
            mel = tf.expand_dims(mel, axis = -1)
        
        return mel
    
    def get_input(self, data):
        if isinstance(data, pd.DataFrame):
            return [self.get_input(row) for idx, row in data.iterrows()]
        elif isinstance(data, list):
            return [self.get_input(data_i) for data_i in data]
        
        if self.use_mel_fn:
            input_data = self.get_mel_input(data)
        else:
            input_data = self.get_audio_input(data)
        
        if tf.shape(input_data)[0] > self.max_input_length:
            start = tf.random.uniform(
                (), minval = 0, 
                maxval = tf.shape(input_data)[0] - self.max_input_length,
                dtype = tf.int32
            )
            input_data = input_data[start : start + self.max_input_length]
        
        return input_data
    
    def augment_audio(self, audio):
        audio = random_shift(audio, min_length = self.max_input_length)
        audio = random_pad(audio, self.max_input_length)
        audio = tf.cond(
            tf.random.uniform(()) < self.augment_prct,
            lambda: random_noise(audio),
            lambda: audio
        )
        
        return audio
    
    def augment_mel(self, inp):
        maxval = self.max_input_length - tf.shape(inp)[0]
        if maxval > 0:
            padding_left = tf.random.uniform(
                (), minval = 0, 
                maxval = maxval,
                dtype = tf.int32
            )
            
            if maxval - padding_left > 0:
                padding_right = tf.random.uniform(
                    (), minval = 0, 
                    maxval = maxval - padding_left,
                    dtype = tf.int32
                )
            else:
                padding_right = 0
            
            if self.mel_as_image:
                padding = [(padding_left, padding_right), (0, 0), (0, 0)]
            else:
                padding = [(padding_left, padding_right), (0, 0)]
            
            inp = tf.pad(inp, padding)
        
        
        inp = tf.cond(
            tf.random.uniform(()) < self.augment_prct,
            lambda: inp + tf.random.uniform(
                tf.shape(inp), 
                minval = -1., maxval = 1.,
                dtype = inp.dtype),
            lambda: inp
        )
        
        return inp
    
    def augment_input(self, inp):
        if self.use_mel_fn:
            return self.augment_mel(inp)
        else:
            return self.augment_audio(inp)
    
    def concat(self, x_same, x_not_same):
        seq_1, seq_2 = tf.shape(x_same)[1], tf.shape(x_not_same)[1]
        
        if seq_1 != seq_2:
            padding = [(0,0), (0, tf.abs(seq_1 - seq_2)), (0,0)]
            if seq_1 > seq_2:
                x_not_same = tf.pad(x_not_same, padding)
            else:
                x_same = tf.pad(x_same, padding)
                
        return tf.concat([x_same, x_not_same], axis = 0)
        
    def get_dataset_config(self, **kwargs):
        kwargs['pad_kwargs']    = {}
        if self.use_fixed_length_input:
            input_shape = self.encoder_input_shape
            kwargs['pad_kwargs'] = {
                'padded_shapes' : (
                    ((input_shape[1:], input_shape[1:]), ()),
                    ((input_shape[1:], input_shape[1:]), ()),
                )
            }
        kwargs['padded_batch']  = True
        
        return super().get_dataset_config(** kwargs)
                        
    def embed_dataset(self, * args, ** kwargs):
        kwargs['rate'] = self.audio_rate
        return super().embed_dataset(* args, ** kwargs)
    
    def identify(self,
                 filenames,
                 alignments = None,
                 batch_size = 64,
                 step       = 2,
                 
                 directory  = None,
                 overwrite  = False,
                 
                 verbose    = True,
                 show   = False,
                 tqdm   = tqdm,
                 
                 ** kwargs
                ):
        filenames = normalize_filename(filenames, invalid_mode = 'keep')
        
        if not isinstance(filenames, (list, tuple)): filenames = [filenames]
        if alignments is not None and not isinstance(alignment, (list, tuple)):
            alignments = [alignments]
        if self.use_fixed_length_input: step = self.max_audio_time
        step = int(step * self.audio_rate)
        
        ####################
        #  Init directory  #
        ####################
        
        if not directory: directory = self.identification_dir
        audio_dir   = os.path.join(directory, 'audios')
        result_dir  = os.path.join(directory, 'results')
        map_file    = os.path.join(directory, 'map.json')

        os.makedirs(audio_dir, exist_ok = True)
        os.makedirs(result_dir, exist_ok = True)
        
        # Load already predicted data (if any)
        all_outputs = load_json(map_file, default = {})
        outputs     = []
        
        ########################################
        #  Perform identification on each file #
        ########################################
        
        for i, filename in enumerate(filenames):
            # Normalize filename and save raw audio (if necessary)
            if isinstance(filename, tf.Tensor) and filename.dtype == tf.float32:
                filename = filename.numpy()
            if isinstance(filename, np.ndarray) and filename.dtype == np.float32:
                audio_num = len(os.listdir(audio_dir))
                audio_filename = os.path.join(audio_dir, 'audio_{}.wav'.format(audio_num))

                write_audio(filename, audio_filename, rate = self.audio_rate)
            else:
                audio_filename = normalize_filename(filename)
            
            print("Processing file {}...".format(audio_filename))
            # Load already predicted result
            if audio_filename in all_outputs and not overwrite:
                result = AudioAnnotation.load_from_file(
                    all_outputs[audio_filename]
                )
                outputs.append(result)
                continue
            
            audio = audio_filename if isinstance(audio_filename, np.ndarray) else load_audio(audio_filename, self.audio_rate)
            
            if alignments is not None:
                alignment = alignments[i]
            else:
                alignment = [{
                    'start' : start / self.audio_rate,
                    'end'   : min(len(audio), start + step) / self.audio_rate, 
                    'time'  : min(step, len(audio) - start) / self.audio_rate
                } for start in range(0, len(audio), step)]
            
            # Embed data
            # Note : embedding `filename` instead of `audio_filename` allows to not
            # reload audio if raw audio was given
            parts = [
                audio[int(align['start'] * self.audio_rate) : int(align['end'] * self.audio_rate)]
                for align in alignment
            ]
            embedded_parts = self.embed(
                parts, 
                batch_size = batch_size, 
                tqdm       = tqdm
            )
            
            # Build similarity matrix
            similarity_matrix = self.pred_similarity_matrix(embedded_parts)

            # Predict ids based on the `KPropagation` clustering
            k_prop = KPropagation(
                embedded_parts, similarity_matrix = similarity_matrix, tqdm = tqdm, 
                ** kwargs
            )
            if show:
                k_prop.plot()
            
            for i, label in enumerate(k_prop.labels):
                alignment[i]['id'] = label
            
            # Make time alignment with predicted ids
            result_dir = all_outputs.get(
                audio_filename, 
                os.path.join(result_dir, '{}_result'.format(os.path.basename(filename)[:-4]))
            )
            result = AudioAnnotation(
                directory   = result_dir,
                filename    = audio_filename, 
                rate        = self.audio_rate,
                infos       = alignment,
                ** kwargs
            )
            result.save()

            outputs.append(result)
            all_outputs[audio_filename] = result_dir

            dump_json(map_file, all_outputs, indent = 4)

        return outputs
    
    def get_config(self, *args, ** kwargs):
        config = super().get_config(*args, **kwargs)
        config['audio_rate']        = self.audio_rate
        config['input_type']        = self.input_type
        config['max_audio_time']    = self.max_audio_time
        config['encoder_type']      = self.encoder_type
        config['use_fixed_length_input']    = self.use_fixed_length_input
        
        if self.use_mel_fn:
            config['mel_fn_type']       = self.mel_fn_file
            
        return config
