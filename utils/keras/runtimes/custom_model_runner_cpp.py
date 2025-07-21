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

import copy
import torch
import numpy as np

from tensorrt_llm.layers import MropeParams
from tensorrt_llm.runtime import ModelRunnerCpp
from tensorrt_llm.bindings import executor as trtllm
from typing import Dict, List, Optional, Union, Callable

from loggers import Timer, timer

_sampling_params = {k for k in vars(trtllm.SamplingConfig).keys() if not k.startswith('_')}
_rename_params   = {"num_beams": "beam_width", "random_seed": "seed"}

class CustomModelRunnerCpp(ModelRunnerCpp):
    """
        This class re-implements the `generate` method of `tensorrt_llm.ModelRunnerCpp` to :
        1) Replace `torch.Tensor` inputs to `np.ndarray`, as the `Tensor` are never used directly
           --> It avoids copying arrays to GPU to then re-convert them to list on CPU
        2) Remove the automatic padding to `max_seq_len` (`pad_output`) to avoid unnecessary operations
        3) Add the `request_id` in the stream output to facilitate abort
        4) Add `add_input_tokens` to control whether or not to add the input ids to the output tokens
           --> Most of the time they are not used, it is therefore a waste of compute to add them
    """
    def abort(self, request_id):
        if not isinstance(request_id, (list, tuple)): request_id = [request_id]
        for req_id in request_id: self.session.cancel_request(req_id)
    
    def generate(self,
                 batch_input_ids    : List[Union[torch.Tensor, np.ndarray, List[int]]],
                 *,
                 
                 position_ids       : List[Union[torch.Tensor, np.ndarray, List[int]]]    = None,
                 encoder_input_ids  : List[Union[torch.Tensor, np.ndarray, List[int]]]    = None,
                 encoder_input_features : List[Union[torch.Tensor, np.ndarray]]  = None,
                 encoder_output_lengths : List[int] = None,
                 cross_attention_masks  : List[torch.Tensor]  = None,
                 
                 lora_uids  : Optional[list] = None,
                 mrope_params   : Optional[MropeParams] = None,
                 sampling_config    : Optional[trtllm.SamplingConfig] = None,
                 lookahead_config   : Optional[List[int]]   = None,
                 kv_cache_retention_config  : Optional[trtllm.KvCacheRetentionConfig]   = None,
                 
                 end_id : Optional[int] = None,
                 pad_id : Optional[int] = None,
                 max_new_tokens : int   = 1,
                 bad_words_list : Optional[List[List[int]]] = None,
                 stop_words_list    : Optional[List[List[int]]] = None,

                 stopping_criteria  = None,
                 logits_processors  : Optional[List[Callable]]  = None,
                 logits_processor_names : Optional[List[str]] = None,
                 
                 streaming  : bool = False,
                 return_dict    : bool = False,
                 output_log_probs   : bool = False,
                 output_cum_log_probs   : bool = False,
                 output_sequence_lengths    : bool = False,
                 output_generation_logits   : bool = False,
                 return_all_generated_tokens    : bool = False,
                 include_input_in_output    : bool  = False,
                 
                 prompt_table   : Optional[str] = None,
                 prompt_tasks   : Optional[str] = None,
                 input_token_extra_ids  : List[List[int]] = None,
                 language_adapter_uids  : Optional[List[int]] = None,
                 mm_embedding_offloading    : bool = False,
                 ** kwargs
                ) -> Union[List[List[List[int]]], dict]:
        """
            Generates sequences of token ids.
            The generation-controlling parameters are set in the sampling_config; it will be set to a default one if not passed.
            You can override any sampling_config's attributes by passing corresponding parameters.

            Args:
                batch_input_ids (List[torch.Tensor]):
                    A list of input id tensors. Each tensor is of shape (sequence_length, ).
                position_ids (List[torch.Tensor]):
                    A list of position id tensors. Each tensor is of shape (sequence_length, ).
                encoder_input_ids (List[torch.Tensor]):
                    A list of encoder input id tensors for encoder-decoder models (optional). Each tensor is of shape (sequence_length, ).
                encoder_input_features: (List[torch.Tensor]):
                    A list of encoder input feature tensors for multimodal encoder-decoder models (optional). Each tensor is of shape (sequence_length, feature_dim).
                encoder_output_lengths: (List[int]):
                    A list of encoder output lengths (optional) if encoder output has different length from encoder input (due to convolution down-sampling, etc.)
                sampling_config (SamplingConfig):
                    The sampling configuration to be used as base parametrization for the generation call.
                    The passed **kwargs matching the sampling_config's attributes will override them.
                    If the sampling_config is not provided, a default will be used.
                prompt_table (str or torch.Tensor):
                    The file path of prompt table (.npy format, exported by nemo_prompt_convert.py) or the prompt table itself.
                prompt_tasks (str):
                    The prompt tuning task ids for the input batch, in format of comma-separated list (e.g., 0,3,1,0).
                input_token_extra_ids (List[List[int]]):
                    Input token extra ids for using p-tuning and KV Cache reuse together
                lora_uids (list):
                    The uids of LoRA weights for the input batch. Use -1 to disable the LoRA module.
                streaming (bool):
                    Whether or not to use streaming mode for generation.
                stopping_criteria (StoppingCriteria):
                    Custom stopping criteria.
                logits_processor_names (List[str]):
                    Custom logits processor names.
                return_all_generated_tokens (bool):
                    Whether the full output is returned at each streaming step
                kwargs (Dict[str, Any]:
                    Ad hoc parametrization of sampling_config.
                    The passed **kwargs matching the sampling_config's attributes will override them.
            Returns:
                torch.Tensor or dict:
                    If return_dict=False, the method returns generated output_ids.
                    If return_dict=True, the method returns a dict of output_ids,
                    sequence_lengths (if sampling_config.output_sequence_lengths=True),
                    context_logits and generation_logits (if self.gather_context_logits=True and
                    self.gather_generation_logits=True, respectively).
        """
        # TODO: Check if these can be supported now and support them
        if stopping_criteria is not None:
            raise RuntimeError("Stopping criteria is not supported in C++ session.")

        if not self.use_kv_cache and max_new_tokens > 1:
            raise RuntimeError('Disabled KV cache is intended for context phase only now.')

        # If we are in a multi-gpu scenario, only rank 0 continues
        if not self.session.can_enqueue_requests():
            raise RuntimeError('Session cannot enqueue new requests')

        batch_size = len(batch_input_ids)
        # Convert input ids to plain lists
        batch_input_ids_list = [
            inp.tolist() if not isinstance(inp, list) else inp
            for inp in batch_input_ids
        ]
        position_ids_list = [
            pos.tolist() if not isinstance(pos, list) else pos
            for pos in position_ids
        ] if position_ids is not None else [None] * batch_size

        encoder_input_ids_list = [
            inp.tolist() if not isinstance(inp, list) else inp
            for inp in encoder_input_ids
        ] if encoder_input_ids is not None else [None] * batch_size
        
        if encoder_input_features is not None:
            for i in range(len(encoder_input_features)):
                if encoder_input_features[i] is not None:
                    encoder_input_features[i] = encoder_input_features[i].contiguous()
        else:
            encoder_input_features = [None] * batch_size
        
        if cross_attention_masks:
            for i in range(batch_size):
                if cross_attention_masks[i] is not None:
                    cross_attention_masks[i] = cross_attention_masks[i].contiguous()
        else:
            cross_attention_masks = [None] * batch_size
        
        if not encoder_output_lengths:
            encoder_output_lengths = [None] * batch_size

        
        sampling_config_list = self._prepare_sampling_config(
            sampling_config, batch_size, ** kwargs
        )
        self._check_inputs(
            batch_input_ids_list,
            encoder_input_ids_list if encoder_input_ids is not None else None,
            sampling_config_list[0],
            max_new_tokens
        )

        prompt_tuning_config_list = self._prepare_ptuning_executor(
            batch_input_ids_list,
            prompt_table,
            prompt_tasks,
            input_token_extra_ids,
            mm_embedding_offloading=mm_embedding_offloading)
        mrope_config_list = self._prepare_mrope_executor(batch_input_ids_list, mrope_params)
        lora_config_list = self._prepare_lora_configs(lora_uids, batch_size)

        stop_words_list = self._prepare_words_list(stop_words_list, batch_size)
        bad_words_list  = self._prepare_words_list(bad_words_list, batch_size)
        logits_processor_names = self._prepare_names_list(logits_processor_names, batch_size)
        
        if logits_processors is None or not isinstance(logits_processors, list):
            logits_processors = [logits_processors] * batch_size
        
        request_lookahead_config = None
        if lookahead_config is not None:
            [w, n, g] = lookahead_config
            request_lookahead_config = trtllm.LookaheadDecodingConfig(w, n, g)
        skip_cross_attn_blocks = kwargs.get('skip_cross_attn_blocks', None)
        
        # Draft-Target-Model speculative decoding
        is_draft_target_model = False
        external_draft_tokens_config_list   = [None] * batch_size
        if kwargs.get('draft_tokens_list', None) is not None:
            is_draft_target_model = True
            if kwargs.get('draft_logits_list', None) is not None:
                # Use logits to accept
                external_draft_tokens_config_list = [
                    ExternalDraftTokensConfig(draft_tokens, draft_logits)
                    for draft_tokens, draft_logits in zip(
                        kwargs["draft_tokens_list"], kwargs["draft_logits_list"])
                ]
            else:
                # Use tokens to accept
                external_draft_tokens_config_list = [
                    ExternalDraftTokensConfig(draft_tokens)
                    for draft_tokens in kwargs["draft_tokens_list"]
                ]

        if language_adapter_uids is None:
            language_adapter_uids = [None] * batch_size
        
        output_config = trtllm.OutputConfig(
            return_log_probs    = output_log_probs,
            return_context_logits   = self.gather_context_logits,
            return_generation_logits    = self.gather_generation_logits or output_generation_logits,
        )

        requests = [
            trtllm.Request(
                input_token_ids     = batch_input_ids_list[i],
                
                encoder_input_token_ids = encoder_input_ids_list[i],
                encoder_input_features  = encoder_input_features[i],
                encoder_output_length   = encoder_output_lengths[i],
                
                position_ids    = position_ids_list[i],
                cross_attention_mask    = cross_attention_masks[i],
                
                end_id  = end_id,
                pad_id  = pad_id,
                max_tokens  = max_new_tokens,
                
                stop_words  = stop_words_list[i],
                bad_words   = bad_words_list[i],
                
                lora_config = lora_config_list[i],
                mrope_config    = mrope_config_list[i],
                output_config   = output_config,
                sampling_config = sampling_config_list[i],
                lookahead_config    = request_lookahead_config,
                prompt_tuning_config    = prompt_tuning_config_list[i],
                external_draft_tokens_config    = external_draft_tokens_config_list[i],
                kv_cache_retention_config   = kv_cache_retention_config,
                
                skip_cross_attn_blocks  = skip_cross_attn_blocks,
                language_adapter_uid    = language_adapter_uids[i],

                return_all_generated_tokens = return_all_generated_tokens,

                streaming   = streaming,
                logits_post_processor   = logits_processors[i],
                logits_post_processor_name  = logits_processor_names[i]
            ) for i in range(batch_size)
        ]

        request_ids = self.session.enqueue_requests(requests)

        input_lengths   = [len(inp) for inp in batch_input_ids_list]
        num_sequences   = self._get_num_sequences(sampling_config_list[0])

        if include_input_in_output:
            outputs = {req_id : {
                'input_ids'     : batch_input_ids_list[b],
                'input_length'  : input_lengths[b],
                'output_ids'    : [batch_input_ids_list[b].copy() for _ in range(num_sequences)]
            } for b, req_id in enumerate(request_ids)}
        else:
            outputs = {
                req_id : {
                    'input_ids'     : batch_input_ids_list[b],
                    'input_length'  : input_lengths[b],
                    'output_ids'    : [[] for _ in range(num_sequences)]
                }
                for b, req_id in enumerate(request_ids)
            }

        config = {
            'outputs'   : outputs,
            'request_ids'   : request_ids,
            'input_lengths' : input_lengths,
            'batch_input_ids_list' : batch_input_ids_list,
            
            'end_id'        : end_id,
            'streaming'     : streaming,
            'batch_size'    : batch_size,
            'beam_width'    : getattr(
                sampling_config_list[0], 'num_beams', getattr(sampling_config_list[0], 'beam_width')
            ),
            'num_sequences' : num_sequences,
            'sampling_config'   : sampling_config_list[0],
            
            'return_dict' : return_dict,
            'output_sequence_lengths' : output_sequence_lengths,
            'output_generation_logits'  : output_generation_logits,
            'output_log_probs'  : output_log_probs,
            'output_cum_log_probs'  : output_cum_log_probs,
            'include_input_in_output'   : include_input_in_output,
            'return_all_generated_tokens'   : return_all_generated_tokens,
            'is_draft_target_model' : is_draft_target_model
        }
        if not streaming:
            return self._initialize_and_fill_output(** config)
        else:
            return self._stream(** config)

    def _prepare_sampling_config(self, sampling_config, batch_size, ** kwargs):
        if sampling_config is None:
            # Convert from old API of SamplingConfig
            # Note: Due to a Python3.10 bug one cannot use inspect on it currently
            sampling_params = {
                k : v for k, v in kwargs.items() if k in _sampling_params
            }
        elif isinstance(sampling_config, list):
            assert len(sampling_config) == batch_size
            return sampling_config
        elif isinstance(sampling_config, trtllm.SamplingConfig):
            return [sampling_config] * batch_size
        else:
            sampling_params = copy.deepcopy(sampling_config)
        
        for k, v in _rename_params.items():
            if k in sampling_params:
                sampling_params[v] = sampling_params.pop(k)
            
        if "top_p" in sampling_params and sampling_params["top_p"] == 0.0:
            sampling_params["top_p"] = None

        # TODO: improve usage of SamplingConfig. For example,
        # construct SamplingConfig for each request, rather than one for the whole batch.
        # Here we use beam width array for each request for Variable-Beam-Width-Search.
        use_variable_beam_width = (
            'beam_width_array' in sampling_params
            and sampling_params['beam_width_array'] is not None
            and len(sampling_params['beam_width_array']) == batch_size
        )
        
        if not use_variable_beam_width:
            sampling_config = trtllm.SamplingConfig(** sampling_params)
            return [sampling_config] * batch_size
        
        # Just placeholder for non-Variable-Beam-Width-Search
        sp_copy = copy.deepcopy(sampling_params)
        sampling_config_list = []
        for beam_width in sampling_params['beam_width_array']:
            sp_copy["beam_width_array"] = beam_width
            sp_copy["beam_width"] = max(beam_width)
            sampling_config_list.append(SamplingConfig(** sp_copy))
        
        return sampling_config_list

    def _prepare_ptuning_executor(self, batch_input_ids_list, prompt_table,
                                  prompt_tasks, input_token_extra_ids,
                                  mm_embedding_offloading):
        if input_token_extra_ids:
            assert len(batch_input_ids_list) == len(input_token_extra_ids), \
                f"Batch size of input_token_extra_ids ({len(input_token_extra_ids)}) must be the same as input batch size ({len(batch_input_ids_list)})"
        prompt_tuning_configs = len(batch_input_ids_list) * [None]
        if prompt_table is not None:
            if mm_embedding_offloading:
                # CUDA Stream Overlapping Requirements:
                # 1. Both memory copy stream and kernel execution stream must be non-default streams
                # 2. For host<->device transfers (H2D/D2H), host memory MUST be page-locked (pinned)
                prompt_table_data = self._prepare_embedding_table(
                    prompt_table).pin_memory()
            else:
                prompt_table_data = self._prepare_embedding_table(
                    prompt_table).cuda()
            if prompt_tasks is not None:
                task_indices = [int(t) for t in prompt_tasks.split(',')]
                assert len(task_indices) == len(batch_input_ids_list), \
                    f"Number of supplied tasks ({len(task_indices)}) must match input batch size ({len(batch_input_ids_list)})"
                prompt_tuning_configs = [
                    trtllm.PromptTuningConfig(
                        embedding_table=prompt_table_data[task_indices[i]],
                        input_token_extra_ids=input_token_extra_ids[i]
                        if input_token_extra_ids else None)
                    for i in range(len(batch_input_ids_list))
                ]
            else:
                prompt_tuning_configs = [
                    trtllm.PromptTuningConfig(
                        embedding_table=prompt_table_data[0],
                        input_token_extra_ids=input_token_extra_ids[i]
                        if input_token_extra_ids else None)
                    for i in range(len(batch_input_ids_list))
                ]
        return prompt_tuning_configs

    def _initialize_and_fill_output(self, request_ids, ** kwargs):
        multi_responses = self.session.await_responses(request_ids)
        responses = [
            response for responses in multi_responses for response in responses
        ]

        return self._fill_output(
            request_ids = request_ids, responses = responses, ** kwargs
        )

    def _stream(self, request_ids, ** kwargs):
        finished_request_ids = set()
        while len(finished_request_ids) != len(request_ids):
            multi_responses = self.session.await_responses(request_ids)
            responses = [
                response for responses in multi_responses for response in responses
            ]
            for response in responses:
                if response.result.is_final:
                    finished_request_ids.add(response.request_id)

            yield self._fill_output(
                request_ids = request_ids, responses = responses, ** kwargs
            )

    def _fill_output(self,
                     *,
                     
                     outputs,
                     responses,
                     request_ids,
                     input_lengths,
                     batch_input_ids_list,
                     
                     end_id,
                     streaming,
                     batch_size,
                     beam_width,
                     num_sequences,
                     sampling_config,
                     
                     return_dict,
                     output_sequence_lengths,
                     output_generation_logits,
                     output_log_probs,
                     output_cum_log_probs,
                     return_all_generated_tokens,
         
                     output_type    : str   = 'list',
                     include_input_in_output : bool  = False,
                     is_draft_target_model   : bool  = False,
                     ** _
                    ):
        kwargs  = {
            'include_input_in_output'   : include_input_in_output,
            'return_all_generated_tokens'   : return_all_generated_tokens
        }

        is_beam_search = beam_width > 1
        for response in responses:
            result = response.result
            if response.has_error():
                raise RuntimeError(response.error_msg)
            elif is_beam_search:
                for beam, output_tokens in enumerate(result.output_token_ids):
                    _fill_output_ids(
                        outputs, output_tokens, response.request_id, beam, ** kwargs
                    )
            else:
                _fill_output_ids(
                    outputs, result.output_token_ids[0], response.request_id, result.sequence_index,
                    ** kwargs
                )

        output_ids = [
            outputs[req_id]['output_ids'] for req_id in request_ids
        ]
        
        sequence_lengths = None
        if output_sequence_lengths:
            sequence_lengths = [
                [len(token_ids) for token_ids in beams]
                for beams in output_ids
            ]

        # Pad by end_id tokens (batch, num_sequences, max_seq_len).
        if streaming:
            if not return_all_generated_tokens:
                output_ids = copy.deepcopy(output_ids)
            else:
                output_ids = [beams.copy() for beams in output_ids]
        elif not include_input_in_output:
            for b, req_id in enumerate(request_ids):
                l = outputs[req_id]['input_length']
                output_ids[b] = [beam[l :] for beam in output_ids[b]]
        
        if output_type != 'list':
            for beams in output_ids:
                for token_ids in beams:
                    token_ids.extend([end_id] * (self.max_seq_len - len(token_ids)))

            output_ids = _maybe_convert_output(output_ids, output_type)

        if return_dict:
            outputs = {'output_ids': output_ids, 'request_ids' : request_ids}

            if output_sequence_lengths:
                outputs['sequence_lengths'] = _maybe_convert_output(sequence_lengths, output_type)

            if self.gather_context_logits:
                context_logits = None
                max_input_len = max(input_lengths)
                for response in responses:
                    result = response.result
                    logits = result.context_logits
                    if logits is None:
                        continue
                    input_len, vocab_size = logits.shape
                    if context_logits is None:
                        context_logits = torch.zeros(
                            (batch_size, max_input_len, vocab_size),
                            dtype=logits.dtype,
                            device=cuda_device)
                    if result.sequence_index == 0:
                        batch_idx = request_ids.index(response.request_id)
                        context_logits[batch_idx, :input_len, :] = logits
                assert context_logits is not None
                outputs['context_logits'] = context_logits

            if self.gather_generation_logits or output_generation_logits:
                gen_logits = None
                if is_draft_target_model:
                    # Put the outputs in a list rather than a tensor since their
                    # length may vary among requests in a batch
                    gen_logits = [
                        a.result.generation_logits.cuda() for a in responses
                        if a.result.generation_logits is not None
                    ]
                else:
                    # The shape of generation logits
                    #   (num_sequences, seq_len, vocab_size) in non-streaming
                    #   (seq_len, num_sequences, vocab_size) in streaming
                    seq_dim = 0 if streaming else 1
                    max_out_len = max(
                        response.result.generation_logits.size(seq_dim)
                        for response in responses
                        if response.result.generation_logits is not None)
                    vocab_size = responses[0].result.generation_logits.size(-1)
                    if not streaming:
                        gen_shape = (num_sequences, max_out_len, vocab_size)
                    elif streaming and return_all_generated_tokens:
                        gen_shape = (max_out_len, num_sequences, vocab_size)
                    else:
                        # streaming and not return_all_generated_tokens
                        gen_shape = (1, num_sequences, vocab_size)
                    logits_dtype = responses[0].result.generation_logits.dtype
                    gen_logits = torch.zeros((batch_size, *gen_shape),
                                             dtype=logits_dtype,
                                             device=cuda_device)

                    for response in responses:
                        logits = response.result.generation_logits
                        if logits is None:
                            continue
                        seq_len = logits.size(seq_dim)

                        batch_idx = request_ids.index(response.request_id)
                        seq_idx = response.result.sequence_index
                        if streaming:
                            if is_beam_search:
                                # WAR: gen_logits contains all beams, clipping
                                # the first n beams as a postprocessing.
                                gen_logits[batch_idx, :seq_len,
                                           ...] = logits[:, :num_sequences, :]
                            else:
                                gen_logits[batch_idx, :seq_len, seq_idx,
                                           ...] = logits[:, 0, :]
                        else:
                            if is_beam_search:
                                gen_logits[batch_idx, :, :seq_len, ...] = logits
                            else:
                                gen_logits[batch_idx, seq_idx, :seq_len,
                                           ...] = logits[0]
                outputs['generation_logits'] = gen_logits

            if output_log_probs:
                max_log_probs_len = max(
                    len(lprobs) for response in responses
                    for lprobs in response.result.log_probs)
                log_probs = torch.zeros(
                    (batch_size, num_sequences, max_log_probs_len),
                    dtype=torch.float32)
                for response in responses:
                    batch_idx = request_ids.index(response.request_id)
                    if is_beam_search:
                        for beam_idx, lprobs in enumerate(
                                response.result.log_probs):
                            log_probs[batch_idx,
                                      beam_idx, :len(lprobs)] = torch.tensor(
                                          lprobs)
                    else:
                        seq_idx = response.result.sequence_index
                        lprobs = response.result.log_probs[0]
                        log_probs[batch_idx,
                                  seq_idx, :len(lprobs)] = torch.tensor(lprobs)
                assert isinstance(log_probs, torch.Tensor)
                outputs['log_probs'] = log_probs.to(cuda_device)

            if output_cum_log_probs:
                cum_log_probs = torch.zeros((batch_size, num_sequences),
                                            dtype=torch.float32)
                for response in responses:
                    if response.result.cum_log_probs is None:
                        continue
                    batch_idx = request_ids.index(response.request_id)
                    clprobs = torch.tensor(response.result.cum_log_probs)
                    if is_beam_search:
                        cum_log_probs[batch_idx, :] = clprobs
                    else:
                        seq_idx = response.result.sequence_index
                        cum_log_probs[batch_idx, seq_idx] = clprobs
                outputs['cum_log_probs'] = cum_log_probs.to(cuda_device)

            #outputs = self._prepare_outputs(outputs, input_lengths)
        else:
            outputs = output_ids

        return outputs

def _fill_output_ids(outputs,
                     result_token_ids,
                     request_id,
                     seq_idx,
                     include_input_in_output,
                     return_all_generated_tokens
                    ):
    # Return shape = (batch_size, num_sequences, seq_len)
    if not return_all_generated_tokens:
        outputs[request_id]['output_ids'][seq_idx].extend(result_token_ids)
    elif include_input_in_output:
        outputs[request_id]['output_ids'][seq_idx] = (
            outputs[request_id]['input_ids'] + result_token_ids
        )
    else:
        outputs[request_id]['output_ids'][seq_idx] = result_token_ids

def _maybe_convert_output(out, output_type, dtype = 'int32'):
    if output_type == 'list':
        return out
    elif output_type in ('np', 'numpy'):
        return np.array(out, dtype = dtype)
    elif output_type in ('torch', 'tensor'):
        return torch.tensor(
            out, dtype = getattr(torch, dtype), device = torch.device('cuda')
        )

