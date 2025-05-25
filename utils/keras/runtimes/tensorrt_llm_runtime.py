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
import collections
import numpy as np

from threading import Lock

from .. import timer
from .runtime import Runtime
from ..gpu import get_gpu_memory_infos

TRTLLMInferenceOutput = collections.namedtuple(
    "TRTLLMInferenceOutput", [
        "tokens", "lengths", "offset"
    ]
)

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
    def __init__(self, path, ** kwargs):
        super().__init__(path, ** kwargs)

        self.is_enc_dec = 'encoder' in os.listdir(self.path)
        self.infer_signature    = inspect.signature(self.engine.generate).parameters.keys()
        
        self.sos_token  = -1
        self.eos_token  = -1
        self.pad_token  = -1
        
        self._token_masking = self.engine.logits_processor_map['token_masking']
        self._inference_stopper = self.engine.logits_processor_map['inference_stopper']
        
        self._mutex = Lock()
        self._request_index = 0
        self._max_input_length  = self.engine.max_input_len
    
    @property
    def max_input_length(self):
        return self._max_input_length
    
    def set_tokens(self, sos_token = None, eos_token = None, pad_token = None):
        if sos_token not in (-1, None): self.sos_token = sos_token
        if eos_token not in (-1, None): self.eos_token = eos_token
        if pad_token not in (-1, None): self.pad_token = pad_token

    def start_request(self,
                      inputs,
                      *,
                      
                      streaming,
                      batch_size,
                      
                      tokenizer = None,
                      allowed_tokens = None,
                      stop_condition = None,
                     ):
        with self._mutex:
            self._request_index += batch_size
            trt_llm_request_id = self._request_index
            
            if allowed_tokens is not None:
                self._token_masking.init_request(trt_llm_request_id, allowed_tokens)
                inputs['max_new_tokens'] = self._token_masking.get_max_length(trt_llm_request_id),
                inputs.setdefault('logits_processor_names', []).append('token_masking')
            
            if stop_condition is not None:
                self._inference_stopper.tokenizer = tokenizer
                self._inference_stopper.init_request(trt_llm_request_id, stop_condition)
                inputs.setdefault('logits_processor_names', []).append('inference_stopper')

            if streaming:
                output = self.engine.generate(streaming = True, ** inputs)
        
        if not streaming:
            output = self.engine.generate(streaming = False, ** inputs)
        
        return trt_llm_request_id, output
    
    def finalize_request(self, req_id):
        for logits_processor in (self._token_masking, self._inference_stopper):
            logits_processor.finalize_request(req_id)
        
    @timer(name = 'TRT-LLM inference')
    def __call__(self,
                 inputs,
                 *,
                 
                 tokens = None,
                 num_beams  = None,
                 max_input_len  = None,
                 encoder_output_lengths = None,
                 
                 tokenizer  = None,
                 stop_condition = None,
                 allowed_tokens = None,
                 
                 request_id = None,
                 decode_fn  = None,
                 stream_callback    = None,
                 add_none_at_eos    = False,

                 ** kwargs
                ):
        import torch
        
        if max_input_len: self.engine.max_input_len = max_input_len
        else:             self.engine.max_input_len = self.max_input_length
        
        inputs = self.prepare_tensor(
            inputs, self.is_enc_dec, pad_token = self.pad_token, dtype = self.engine.dtype
        )
        if tokens is not None:
            tokens = self.prepare_tensor(tokens, pad_token = self.pad_token)
        
        if 'kwargs' not in self.infer_signature:
            kwargs = {k : v for k, v in kwargs.items() if k in self.infer_signature}
        else:
            kwargs = {k : v for k, v in kwargs.items() if 'prompt' not in k and 'format' not in k}
        
        if num_beams is None: num_beams = getattr(self.engine, 'max_beam_width', 1)
        
        kwargs.update({
            'end_id'    : self.eos_token,
            'pad_id'    : self.pad_token,
            'num_beams' : num_beams,
            'return_dict'   : True,
            'num_return_sequences'  : 1,
            'output_sequence_lengths'   : True
        })
        if self.is_enc_dec:
            kwargs['batch_input_ids'] = tokens
            if inputs[0].dtype.is_floating_point:
                kwargs['encoder_input_features'] = inputs
            else:
                kwargs['encoder_input_ids'] = inputs
            
            if encoder_output_lengths:
                if not isinstance(encoder_output_lengths, list):
                    encoder_output_lengths = [encoder_output_lengths] * len(inputs)
                kwargs['encoder_output_lengths'] = encoder_output_lengths
        else:
            kwargs['batch_input_ids'] = inputs
        
        inp_lengths = [len(tok) for tok in kwargs['batch_input_ids']]
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Calling `TRT-LLM` generate with {}'.format({
                k : v if not isinstance(v, list) or not hasattr(v[0], 'shape') else [
                    '<Tensor shape={} dtype={}>'.format(vi.shape, vi.dtype) for vi in v
                ] for k, v in kwargs.items()
            }))
        
        
        t0 = time.time()
        with torch.no_grad():
            trt_llm_request_id, output = self.start_request(
                kwargs,
                streaming   = stream_callback is not None,
                batch_size  = len(inputs),
                
                tokenizer   = tokenizer,
                allowed_tokens  = allowed_tokens,
                stop_condition  = stop_condition
            )

            if stream_callback is not None:
                if not callable(stream_callback): stream_callback = stream_callback.put
                
                stream = output
                for i, output in enumerate(stream):
                    if i == 0 and logger.isEnabledFor(logging.INFO):
                        logger.info('[TRT-LLM] Time to first token : {}'.format(
                            _time_to_string(time.time() - t0)
                        ))
                    
                    out_i = TRTLLMInferenceOutput(
                        tokens  = output['output_ids'],
                        lengths = output['sequence_lengths'],
                        offset  = inp_lengths
                    )
                    if decode_fn is not None:
                        out_i = decode_fn(out_i)[0][0]
                    
                    if request_id is not None:
                        out_i = (request_id, out_i)
                    
                    if stream_callback(out_i) is False:
                        self.engine.session.cancel_request(trt_llm_request_id)
                        if hasattr(stream_callback, 'send_status'):
                            stream_callback.send_status(request_id, 'stopped')
                        break

                if add_none_at_eos:
                    final = None if request_id is None else (request_id, None)
                    stream_callback(final)

            self.finalize_request(trt_llm_request_id)
        
        if logger.isEnabledFor(logging.INFO):
            n = output['sequence_lengths'].sum().cpu().numpy() - sum(inp_lengths)
            t1 = time.time()
            logger.info('[TRT-LLM] {} tokens generated in {} ({:.3f} tokens/sec)'.format(
                n, _time_to_string(t1 - t0), n / (t1 - t0)
            ))
        
        return TRTLLMInferenceOutput(
            tokens  = output['output_ids'],
            lengths = output['sequence_lengths'],
            offset  = inp_lengths
        )
    
    @staticmethod
    def prepare_tensor(tensor, is_encoder_input = False, pad_token = -1, dtype = None):
        import torch
        
        if not is_encoder_input:        dtype = torch.int32
        elif isinstance(dtype, str):    dtype = getattr(torch, dtype)
        else:    dtype = getattr(dtype, 'name', dtype)
        
        if not isinstance(tensor, list):
            if not torch.is_tensor(tensor):
                tensor = torch.from_numpy(np.array(tensor)).to(dtype = dtype, device = 'cuda')

            # batched rank is equal to 3 for encoder features (like whisper)
            batched_rank = 2 + int(is_encoder_input and tensor.dtype.is_floating_point)
            if len(tensor.shape) == batched_rank:
                if len(tensor) == 1:
                    tensor = [tensor[0]]
                elif not is_encoder_input:
                    tensor = [
                        inp[:length] for length in (tensor != pad_token).count_nonzero(1)
                    ]
            else:
                tensor = [tensor]
            
        elif isinstance(tensor[0], int):
            tensor = [torch.from_numpy(np.array(tensor, dtype = 'int32')).cuda()]
        else:
            tensor = [
                torch.from_numpy(np.asarray(t)).to(dtype = dtype, device = 'cuda') if not torch.is_tensor(t) else t
                for t in tensor
            ]
        
        if is_encoder_input and dtype and any(t.dtype != dtype for t in tensor):
            tensor = [t.to(dtype = getattr(torch, dtype)) for t in tensor]
        
        return tensor

    @staticmethod
    def load_engine(path,
                    *,
                    
                    use_cpp = True,
                    kv_cache_free_gpu_memory    = None,
                    kv_cache_free_gpu_memory_fraction = None,
                    
                    ** kwargs
                   ):
        from tensorrt_llm.runtime import ModelRunnerCpp, ModelRunner

        if 'encoder' in os.listdir(path):
            kwargs['is_enc_dec'] = True
            for k, v in _default_enc_dec_config.items():
                if k not in kwargs: kwargs[k] = v

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
            kwargs['kv_cache_free_gpu_memory_fraction'] = kv_cache_free_gpu_memory_fraction

        runner_cls = ModelRunnerCpp if use_cpp else ModelRunner
        kwargs     = {
            k : v for k, v in kwargs.items()
            if k in inspect.signature(runner_cls.from_dir).parameters
        }
        kwargs['logits_processor_map'] = _get_logits_processor_map()
        
        engine = runner_cls.from_dir(engine_dir = path, ** kwargs)
        engine.logits_processor_map = kwargs['logits_processor_map']
        
        kwargs['logits_processor_map']['token_masking'].vocab_size = engine.vocab_size
        
        return engine

def _get_kv_cache_fraction(path, kv_cache_memory):
    free = get_gpu_memory_infos()['free'] - _get_engine_size(path)
    return kv_cache_memory / free

def _get_engine_size(path):
    if 'encoder' in os.listdir(path): path = os.path.join(path, '**')
    return sum(os.path.getsize(f) for f in glob.glob(os.path.join(path, '*.engine')))

def _get_logits_processor_map():
    import tensorrt_llm

    class TokenMasking(tensorrt_llm.runtime.generation.LogitsProcessor):
        def __init__(self, value = float('-inf')):
            super().__init__()
            self.value  = value

            self._requests  = {}

        def __call__(self, req_id, logits, ids, stream_ptr, clint_id):
            infos = self._requests.get(req_id, {})
            if infos and infos['step'] < infos['max_length']:
                import torch

                with torch.cuda.stream(torch.cuda.ExternalStream(stream_ptr)):
                    logits[infos['mask'][:, :, infos['step'], :]] = self.value
                    infos['step'] += 1

            return logits

        def get_max_length(self, req_id):
            return self._requests[req_id]['max_length']
        
        def init_request(self, req_id, tokens):
            tokens = np.array(tokens)
            mask   = np.ones((1, 1, tokens.shape[-1], self.vocab_size), dtype = bool)

            for step in range(tokens.shape[-1]):
                mask[:, :, step, tokens[:, step]] = False

            self._requests[req_id] = {
                'stop'  : 0,
                'mask'  : mask,
                'max_length'    : tokens.shape[-1]
            }
        
        def finalize_request(self, req_id):
            self._requests.pop(req_id, None)

    class InferenceStopper(tensorrt_llm.runtime.generation.LogitsProcessor):
        def __init__(self):
            self._tokenizer = None
            self._requests  = {}
            
            self.mask   = None
        
        @property
        def tokenizer(self):
            return self._tokenizer
        
        @tokenizer.setter
        def tokenizer(self, value):
            if value is not self._tokenizer:
                self._tokenizer = value
                self.mask   = np.ones((1, 1, value.vocab_size), dtype = bool)
                self.mask[0, 0, self._tokenizer.blank_token_idx] = False
        
        def __call__(self, req_id, logits, ids, stream_ptr, client_id):
            if req_id not in self._requests: return logits
            
            self._requests[req_id]['text'] += self.tokenizer.decode_ids(ids[0][-1])
            
            if self._requests[req_id]['stop_condition'](self._requests[req_id]['text']):
                import torch
                with torch.cuda.stream(torch.cuda.ExternalStream(stream_ptr)):
                    logits[self.mask] = float('-inf')

            return logits

        def init_request(self, req_id, condition):
            if isinstance(condition, str):
                _pattern = condition
                condition = lambda text: re.search(_pattern, text) is not None
            
            self._requests[req_id] = {
                'text'  : '',
                'stop_condition'    : condition
            }
        
        def finalize_request(self, req_id):
            self._requests.pop(req_id, None)

    return {'token_masking' : TokenMasking(), 'inference_stopper' : InferenceStopper()}

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

class _FakeLock:
    def __enter__(self): pass
    def __exit__(self, * args): pass
