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
import glob
import time
import inspect
import logging
import warnings
import collections
import numpy as np

from .. import ops, timer
from .runtime import Runtime
from .tensorrt_runtime import TensorRTRuntime
from ..gpu import get_gpu_memory_infos

logging.getLogger('tensorrt_llm').setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

_default_kv_cache_free_gpu_memory_fraction  = 0.25

_default_enc_dec_config = {
    'max_input_len' : 32,
    'max_output_len'    : 512,
    'max_batch_size'    : 16,
    'max_beam_width'    : 5,
    'cross_kv_cache_fraction'   : 0.5
}

class TensorRTLLMRuntime(Runtime):
    def __init__(self, path, *, multimodal_engine = None, ** kwargs):
        super().__init__(path, ** kwargs)

        _subdirs = os.listdir(self.path)
        self.is_enc_dec = 'encoder' in _subdirs
        self.is_multimodal  = 'vision' in _subdirs
        self.infer_signature    = set(inspect.signature(self.engine.generate).parameters.keys())
        
        self.multimodal_engine  = multimodal_engine
        if self.is_multimodal and multimodal_engine is None:
            self.multimodal_engine = TensorRTRuntime(os.path.join(self.path, 'vision', 'model.engine'))
        
        self.sos_token  = -1
        self.eos_token  = -1
        self.pad_token  = -1
        
        self._eos_mask  = None
    
    @property
    def max_input_length(self):
        return self.engine.max_input_len
    
    def set_tokens(self, sos_token = None, eos_token = None, pad_token = None):
        if sos_token not in (-1, None): self.sos_token = sos_token
        if eos_token not in (-1, None): self.eos_token = eos_token
        if pad_token not in (-1, None): self.pad_token = pad_token

    @timer(name = 'TRT-LLM inference')
    def __call__(self,
                 inputs,
                 *,
                 
                 tokens = None,
                 streaming  = False,
                 num_beams  = None,
                 encoder_output_lengths = None,
                 
                 tokenizer  = None,
                 stop_condition = None,
                 allowed_tokens = None,

                 ** kwargs
                ):
        if num_beams is None: num_beams = getattr(self.engine, 'max_beam_width', 1)
        
        kwargs.update(self.prepare_inputs(
            inputs, tokens = tokens, encoder_output_lengths = encoder_output_lengths, ** kwargs
        ))

        if 'kwargs' not in self.infer_signature:
            kwargs = {k : v for k, v in kwargs.items() if k in self.infer_signature}
        else:
            kwargs = {k : v for k, v in kwargs.items() if not k.endswith(('_format', '_prompt'))}

        kwargs.update({
            'end_id'    : self.eos_token,
            'pad_id'    : self.pad_token,
            'num_beams' : num_beams,
            'logits_processors' : self.prepare_logits_processor(
                tokenizer, stop_condition, allowed_tokens
            ),
            'return_dict'   : streaming,
            'num_return_sequences'  : 1,
            'include_input_in_output'   : False,
            'output_sequence_lengths'   : False
        })
        
        if logger.isEnabledFor(logging.DEBUG):
            def _get_shape(k, x):
                if hasattr(x, 'shape'):
                    return '<{} shape={} dtype={}>'.format(
                        x.__class__.__name__, tuple(x.shape), getattr(x.dtype, 'name', x.dtype)
                    )
                elif not isinstance(x, list):
                    return x
                elif k == 'batch_input_ids':
                    return [len(xi) for xi in x]
                else:
                    return [_get_shape(k, xi) for xi in x]
            
            logger.debug('Calling `TRT-LLM` inference with {}'.format({
                k : _get_shape(k, v) for k, v in kwargs.items()
            }))
        
        t0 = time.time()
        result = self.engine.generate(streaming = streaming, ** kwargs)
        
        if streaming:
            result = InferenceStream(self.engine, result)
        elif logger.isEnabledFor(logging.INFO):
            t1 = time.time()
            n  = sum(sum(len(beam) for beam in beams) for beams in result)
            logger.info('[TRT-LLM] {} tokens generated in {} ({:.3f} tokens/sec)'.format(
                n, _time_to_string(t1 - t0), n / (t1 - t0)
            ))
            
        return result
    
    def prepare_inputs(self, inputs, tokens = None, encoder_output_lengths = None, ** kwargs):
        if self.is_enc_dec:
            inputs = self._prepare_encoder_features(inputs, dtype = self.engine.dtype)
            if inputs[0].dtype.is_floating_point:
                kwargs['encoder_input_features'] = inputs
            else:
                kwargs['encoder_input_ids'] = inputs
            
            if encoder_output_lengths:
                if not isinstance(encoder_output_lengths, list):
                    encoder_output_lengths = [encoder_output_lengths] * len(inputs)
                kwargs['encoder_output_lengths'] = encoder_output_lengths
            
            inputs = tokens
        
        if self.is_multimodal and 'prompt_table' not in kwargs and kwargs.get(self.multimodal_engine.argnames[0], None) is not None:
            features = self.encode_multimodal_data(** kwargs)
            if isinstance(features, list):
                if len(features) == 1:
                    features = features[0].unsqueeze(0)
                else:
                    import torch
                    features = torch.concat(features, dim = 1) if len(features) > 1 else features[0]
            elif len(features.shape) == 2:
                features = features.unsqueeze(0)
            else:
                features = features.view(1, -1, features.shape[-1])
            
            kwargs['prompt_table'] = features
        
        if not isinstance(inputs, list): inputs = inputs.tolist()
        if not isinstance(inputs[0], list) and not hasattr(inputs[0], 'shape'):
            inputs = [inputs]
        
        kwargs['batch_input_ids'] = inputs
        
        return kwargs

    def encode_multimodal_data(self, ** kwargs):
        multimodal_inputs = [
            kwargs[k] for k in self.multimodal_engine.argnames
        ]
        if isinstance(multimodal_inputs[0], list):
            return [self.multimodal_engine(* inp) for inp in zip(* multimodal_inputs)]
        else:
            return self.multimodal_engine(* multimodal_inputs)

    def prepare_logits_processor(self, tokenizer, stop_condition = None, allowed_tokens = None):
        if stop_condition is None and allowed_tokens is None:
            return None
        
        elif stop_condition:
            if self._eos_mask is None:
                self._eos_mask   = np.ones((1, 1, tokenizer.vocab_size), dtype = bool)
                self._eos_mask[0, 0, tokenizer.eos_token_idx] = False
        
            if isinstance(stop_condition, str):
                _regex = stop_condition
                stop_condition = lambda text: re.search(_regex, text)
        
        return LogitsProcessor(
            tokenizer,
            allowed_tokens  = allowed_tokens,
            stop_condition  = stop_condition,
            eos_mask    = self._eos_mask
        )
        
    @staticmethod
    def _prepare_encoder_features(tensor, dtype = None):
        if not isinstance(tensor, list):
            tensor = ops.convert_to_torch_tensor(tensor, dtype = dtype, device = 'cuda')

            # batched rank is equal to 3 for encoder features (like whisper)
            batched_rank = 2 + int(tensor.dtype.is_floating_point)
            if len(tensor.shape) == batched_rank:
                tensor = list(tensor)
            else:
                tensor = [tensor]
            
        elif isinstance(tensor[0], int):
            tensor = [torch.from_numpy(np.array(tensor, dtype = 'int32')).cuda()]
        else:
            tensor = [
                ops.convert_to_torch_tensor(t, dtype = dtype, device = 'cuda') for t in tensor
            ]
        
        return tensor

    @staticmethod
    def load_engine(path,
                    *,
                    
                    use_cpp = True,
                    kv_cache_free_gpu_memory    = None,
                    kv_cache_enable_block_reuse = True,
                    kv_cache_free_gpu_memory_fraction = None,
                    
                    ** kwargs
                   ):
        if not use_cpp: raise NotImplementedError()
        
        from .custom_model_runner_cpp import CustomModelRunnerCpp

        subdirs = os.listdir(path)
        if 'encoder' in subdirs:
            kwargs['is_enc_dec'] = True
            kv_cache_enable_block_reuse = False
            for k, v in _default_enc_dec_config.items():
                if k not in kwargs: kwargs[k] = v
        elif 'vision' in subdirs:
            assert 'llm' in subdirs, 'Multimodal models require a `llm` sub-directory'
            path = os.path.join(path, 'llm')

        if use_cpp:
            if kv_cache_free_gpu_memory:
                if kv_cache_free_gpu_memory < 128:
                    kv_cache_free_gpu_memory = kv_cache_free_gpu_memory * 1024 ** 3
                elif kv_cache_free_gpu_memory < 128 * 1024:
                    kv_cache_free_gpu_memory = kv_cache_free_gpu_memory * 1024 ** 2
                kv_cache_free_gpu_memory_fraction = _get_kv_cache_fraction(
                    path, kv_cache_free_gpu_memory
                )
            elif not kv_cache_free_gpu_memory_fraction:
                kv_cache_free_gpu_memory_fraction = _default_kv_cache_free_gpu_memory_fraction
            
            kwargs.update({
                'kv_cache_enable_block_reuse'   : kv_cache_enable_block_reuse,
                'kv_cache_free_gpu_memory_fraction' : kv_cache_free_gpu_memory_fraction
            })

        allowed_kwargs = inspect.signature(CustomModelRunnerCpp.from_dir).parameters
        kwargs     = {
            k : v for k, v in kwargs.items() if k in allowed_kwargs
        }
        
        return CustomModelRunnerCpp.from_dir(engine_dir = path, ** kwargs)

class InferenceStream:
    def __init__(self, engine, stream):
        self.engine = engine
        self.stream = stream
        
        self._aborted   = False
        self._last_item = None
    
    @property
    def tokens(self):
        return self._last_item['output_ids']
    
    @property
    def request_id(self):
        return self._last_item['request_ids']
    
    def __iter__(self):
        for item in self.stream:
            self._last_item = item
            yield item['output_ids']
            if self.is_aborted(): break
    
    def abort(self):
        self._aborted = True
        if self._last_item is None: next(iter(self))
        self.engine.abort(self.request_id)
        
    def is_aborted(self):
        return self._aborted
    
class LogitsProcessor:
    def __init__(self,
                 tokenizer,
                 allowed_tokens = None,
                 stop_condition = None,
                 *,
                 
                 eos_mask   = None
                ):
        self.tokenizer  = tokenizer
        self.allowed_tokens = allowed_tokens
        self.stop_condition = stop_condition
        
        self._eos_mask  = eos_mask
        
        self._step  = 0
        self._text  = ''
        
        if self.allowed_tokens is not None:
            self.call = self._guided_process_logits
        elif self.stop_condition is not None:
            assert eos_mask is not None
            self.call = self._stopper_process_logits

    def __call__(self, req_id, logits, ids, stream_ptr, clint_id):
        logits = self.call(logits, ids, stream_ptr)
        
        self._step += 1
        
        return logits
    
    def _stopper_process_logits(self, logits, ids, stream_ptr):
        if self._step == 0: return logits
        
        self._text += self.tokenizer.decode_ids(ids[0][-1])
        
        if self.stop_condition(self._text):
            import torch
            with torch.cuda.stream(torch.cuda.ExternalStream(stream_ptr)):
                logits[self._eos_mask] = float('-inf')

            return logits

    def _guided_process_logits(self, logits, ids, stream_ptr):
        infos = self._requests.get(req_id, {})
        if infos and infos['step'] < infos['max_length']:
            import torch

            with torch.cuda.stream(torch.cuda.ExternalStream(stream_ptr)):
                logits[infos['mask'][:, :, infos['step'], :]] = self.value
                infos['step'] += 1

        return logits

        
def _get_kv_cache_fraction(path, kv_cache_memory):
    free = get_gpu_memory_infos()['free'] - _get_engine_size(path)
    return min(kv_cache_memory / free, 0.75)

def _get_engine_size(path):
    if 'encoder' in os.listdir(path): path = os.path.join(path, '**')
    return sum(os.path.getsize(f) for f in glob.glob(os.path.join(path, '*.engine')))

def _time_to_string(seconds):
    """ Returns a string representation of a time (given in seconds) """
    if seconds < 0.001: return '{} \u03BCs'.format(int(seconds * 1000000))
    if seconds < 0.01:  return '{:.3f} ms'.format(seconds * 1000)
    if seconds < 1.:    return '{} ms'.format(int(seconds * 1000))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = ((seconds % 3600) % 60)
    
    return '{}{}{}'.format(
        '' if h == 0 else '{}h '.format(h),
        '' if m == 0 else '{}min '.format(m),
        '{:.3f} sec'.format(s) if m + h == 0 else '{}sec'.format(int(s))
    )
