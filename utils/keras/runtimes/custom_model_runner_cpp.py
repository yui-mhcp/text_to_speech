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
from typing import List, Optional, Union, Callable

# `trtllm.SamplingConfig` is a pybind11 class : `inspect` cannot introspect its `__init__`,
# the accepted parameters are therefore derived from its attributes
_sampling_params = {k for k in vars(trtllm.SamplingConfig).keys() if not k.startswith('_')}
_rename_params   = {"num_beams": "beam_width", "random_seed": "seed"}
# the legacy names are not attributes of `SamplingConfig` : they have to be accepted *before*
# being renamed, otherwise they are silently dropped by the filtering
_accepted_params = _sampling_params | set(_rename_params)

class CustomModelRunnerCpp(ModelRunnerCpp):
    """
        This class re-implements the `generate` method of `tensorrt_llm.ModelRunnerCpp` to :
        1) Replace `torch.Tensor` inputs to `np.ndarray`, as the `Tensor` are never used directly
           --> It avoids copying arrays to GPU to then re-convert them to list on CPU
        2) Remove the automatic padding to `max_seq_len` (`pad_output`) to avoid unnecessary operations
        3) Add the `request_id` in the stream output to facilitate abort
        4) Add `include_input_in_output` to control whether or not to add the input ids to the
           output tokens
           --> Most of the time they are not used, it is therefore a waste of compute to add them

        Note : the requests are created with `exclude_input_from_output = True`, meaning that the
        responses only contain the *generated* tokens, in streaming as well as in non-streaming
        (the default `ModelRunnerCpp` gets the prompt back from the C++ executor in non-streaming,
        only to strip it afterwards). The prompt is therefore prepended in python, and only when
        `include_input_in_output` is set.
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

                 end_id : Optional[int] = None,
                 pad_id : Optional[int] = None,
                 max_new_tokens : int   = 1,
                 bad_words_list : Optional[List[List[int]]] = None,
                 stop_words_list    : Optional[List[List[int]]] = None,

                 stopping_criteria  = None,
                 logits_processors  : Optional[List[Callable]]  = None,
                 logits_processor_names : Optional[List[str]] = None,

                 lora_uids  : Optional[list] = None,
                 mrope_params   : Optional[MropeParams] = None,
                 sampling_config    : Optional[trtllm.SamplingConfig] = None,
                 lookahead_config   : Optional[List[int]]   = None,
                 kv_cache_retention_config  : Optional[trtllm.KvCacheRetentionConfig]   = None,

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
                prompt_table (str, np.ndarray or torch.Tensor):
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
                include_input_in_output (bool):
                    Whether to prepend the prompt tokens to the returned `output_ids`
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
            return []

        batch_size = len(batch_input_ids)
        # Convert input ids to plain lists
        batch_input_ids_list = [
            inp if isinstance(inp, list) else inp.tolist()
            for inp in batch_input_ids
        ]
        position_ids_list = [
            pos if isinstance(pos, list) else pos.tolist()
            for pos in position_ids
        ] if position_ids is not None else [None] * batch_size

        # kept to `None` (instead of a list of `None`) as `_check_inputs` uses it to determine
        # whether the model is an encoder-decoder one
        encoder_input_ids_list = [
            inp if isinstance(inp, list) else inp.tolist()
            for inp in encoder_input_ids
        ] if encoder_input_ids else None

        # a new list is built to avoid mutating the caller's one
        encoder_input_features = [
            feat.contiguous() if feat is not None else None for feat in encoder_input_features
        ] if encoder_input_features else [None] * batch_size

        cross_attention_masks = [
            mask.contiguous() if mask is not None else None for mask in cross_attention_masks
        ] if cross_attention_masks else [None] * batch_size

        if not encoder_output_lengths:
            encoder_output_lengths = [None] * batch_size


        sampling_config_list = self._prepare_sampling_config(
            sampling_config, batch_size, ** kwargs
        )
        # for Variable-Beam-Width-Search, the requests may have different beam widths : the
        # widest one is used for the checks / allocations, like the `ModelRunnerCpp` placeholder
        widest_sampling_config = max(sampling_config_list, key = _get_beam_width)

        self._check_inputs(
            batch_input_ids_list,
            encoder_input_ids_list,
            widest_sampling_config,
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
            # `LogitsProcessor` are stateful : a single instance must not be shared across requests
            logits_processors = [logits_processors] * batch_size
        else:
            assert len(logits_processors) == batch_size, (
                'There should be exactly 1 `logits_processor` per request, got {} for a batch of {}'.format(
                    len(logits_processors), batch_size
            ))

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
                    trtllm.ExternalDraftTokensConfig(draft_tokens, draft_logits)
                    for draft_tokens, draft_logits in zip(
                        kwargs["draft_tokens_list"], kwargs["draft_logits_list"])
                ]
            else:
                # Use tokens to accept
                external_draft_tokens_config_list = [
                    trtllm.ExternalDraftTokensConfig(draft_tokens)
                    for draft_tokens in kwargs["draft_tokens_list"]
                ]

        if language_adapter_uids is None:
            language_adapter_uids = [None] * batch_size

        output_config = trtllm.OutputConfig(
            # `cum_log_probs` are only computed when the log probs are returned
            return_log_probs    = output_log_probs or output_cum_log_probs,
            return_context_logits   = self.gather_context_logits,
            return_generation_logits    = self.gather_generation_logits or output_generation_logits,
            # the prompt is prepended in python (see the class documentation)
            exclude_input_from_output   = True
        )

        requests = [
            trtllm.Request(
                input_token_ids     = batch_input_ids_list[i],

                encoder_input_token_ids = encoder_input_ids_list[i] if encoder_input_ids_list else None,
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
        beam_width      = _get_beam_width(widest_sampling_config)
        num_sequences   = self._get_num_sequences(widest_sampling_config)

        outputs = {
            req_id : {
                'input_ids'     : batch_input_ids_list[b],
                'output_ids'    : [
                    batch_input_ids_list[b].copy() if include_input_in_output else []
                    for _ in range(num_sequences)
                ]
            }
            for b, req_id in enumerate(request_ids)
        }

        config = {
            'outputs'   : outputs,
            'request_ids'   : request_ids,
            'input_lengths' : input_lengths,

            'streaming'     : streaming,
            'batch_size'    : batch_size,
            'beam_width'    : beam_width,
            'num_sequences' : num_sequences,

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
            # `StreamingResult` (a plain iterable) additionally exposes `request_ids`, so that
            # callers (`InferenceStream.abort`) can cancel the requests *before* the first
            # generation step is consumed (the requests are already enqueued at this point)
            return StreamingResult(self._stream(** config), request_ids)

    def _prepare_sampling_config(self, sampling_config, batch_size, ** kwargs):
        if sampling_config is None:
            # Convert from old API of SamplingConfig
            # Note: Due to a Python3.10 bug one cannot use inspect on it currently
            sampling_params = {
                k : v for k, v in kwargs.items() if k in _accepted_params
            }
        elif isinstance(sampling_config, list):
            assert len(sampling_config) == batch_size
            return sampling_config
        elif isinstance(sampling_config, trtllm.SamplingConfig):
            return [sampling_config] * batch_size
        elif isinstance(sampling_config, dict):
            sampling_params = copy.deepcopy(sampling_config)
        else:
            # old API `SamplingConfig` instance : forwarded as-is, like `ModelRunnerCpp` does
            return [copy.deepcopy(sampling_config)] * batch_size

        for k, v in _rename_params.items():
            if k in sampling_params:
                sampling_params[v] = sampling_params.pop(k)

        if sampling_params.get("top_p", None) == 0.0:
            sampling_params["top_p"] = None

        # TODO: improve usage of SamplingConfig. For example,
        # construct SamplingConfig for each request, rather than one for the whole batch.
        # Here we use beam width array for each request for Variable-Beam-Width-Search.
        beam_width_array = sampling_params.get('beam_width_array', None)
        use_variable_beam_width = (
            beam_width_array is not None and len(beam_width_array) == batch_size
        )

        if not use_variable_beam_width:
            sampling_config = trtllm.SamplingConfig(** sampling_params)
            return [sampling_config] * batch_size

        sp_copy = copy.deepcopy(sampling_params)
        sampling_config_list = []
        for beam_width in beam_width_array:
            sp_copy["beam_width_array"] = beam_width
            sp_copy["beam_width"] = max(beam_width)
            sampling_config_list.append(trtllm.SamplingConfig(** sp_copy))

        return sampling_config_list

    def _prepare_ptuning_executor(self, batch_input_ids_list, prompt_table,
                                  prompt_tasks, input_token_extra_ids,
                                  mm_embedding_offloading):
        # `_prepare_embedding_table` only handles a file path or a `torch.Tensor`
        if isinstance(prompt_table, np.ndarray):
            prompt_table = torch.from_numpy(prompt_table)

        return super()._prepare_ptuning_executor(
            batch_input_ids_list,
            prompt_table,
            prompt_tasks,
            input_token_extra_ids,
            mm_embedding_offloading = mm_embedding_offloading
        )

    def _await_responses(self, request_ids, beam_width, num_sequences):
        """
            Yields the responses, batch per batch, until every request is complete

            The requests are awaited by id (unlike `ModelRunnerCpp`, which awaits *any* response)
            to remain safe when multiple threads call `generate` concurrently (in-flight batching).
            Only the still-pending ids are awaited, as re-awaiting a completed request would
            block on an empty queue.
        """
        # in beam search, a single (final) response contains every beam, while each sequence
        # sends its own final response when using `num_return_sequences`
        num_finals = 1 if beam_width > 1 else num_sequences
        remaining  = {req_id : num_finals for req_id in request_ids}

        while remaining:
            multi_responses = self.session.await_responses(list(remaining))
            responses = [
                response for responses in multi_responses for response in responses
            ]
            for response in responses:
                if response.result.is_final:
                    remaining[response.request_id] -= 1
                    if not remaining[response.request_id]:
                        remaining.pop(response.request_id)

            yield responses

    def _initialize_and_fill_output(self, request_ids, beam_width, num_sequences, ** kwargs):
        responses = [
            response
            for batch in self._await_responses(request_ids, beam_width, num_sequences)
            for response in batch
        ]

        return self._fill_output(
            request_ids = request_ids,
            responses   = responses,
            beam_width  = beam_width,
            num_sequences   = num_sequences,
            ** kwargs
        )

    def _stream(self, request_ids, beam_width, num_sequences, ** kwargs):
        for responses in self._await_responses(request_ids, beam_width, num_sequences):
            yield self._fill_output(
                request_ids = request_ids,
                responses   = responses,
                beam_width  = beam_width,
                num_sequences   = num_sequences,
                ** kwargs
            )

    def _fill_output(self,
                     *,

                     outputs,
                     responses,
                     request_ids,
                     input_lengths,

                     streaming,
                     batch_size,
                     beam_width,
                     num_sequences,

                     return_dict,
                     output_sequence_lengths,
                     output_generation_logits,
                     output_log_probs,
                     output_cum_log_probs,
                     return_all_generated_tokens,

                     include_input_in_output : bool  = False,
                     is_draft_target_model   : bool  = False,
                     ** _
                    ):
        cuda_device = torch.device('cuda')

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

        if streaming:
            # the `outputs` entries keep being mutated by the next steps : they are copied to
            # keep the yielded item valid. `copy.deepcopy` would be O(n²) over the generation
            if not return_all_generated_tokens:
                output_ids = [[beam[:] for beam in beams] for beams in output_ids]
            else:
                # the inner lists are re-assigned (not mutated) at each step
                output_ids = [beams.copy() for beams in output_ids]

        sequence_lengths = None
        if output_sequence_lengths:
            sequence_lengths = [
                [len(token_ids) for token_ids in beams]
                for beams in output_ids
            ]

        if return_dict:
            outputs = {'output_ids': output_ids, 'request_ids' : request_ids}

            if output_sequence_lengths:
                outputs['sequence_lengths'] = sequence_lengths

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

        else:
            outputs = output_ids

        return outputs

class StreamingResult:
    """
        Thin iterable wrapper around the `_stream` generator, additionally exposing the
        `request_ids` of the enqueued requests (available *before* the first step is
        consumed, unlike the per-step `request_ids` entry of the yielded items)
    """
    def __init__(self, generator, request_ids):
        self._generator = generator
        self.request_ids    = request_ids

    def __iter__(self):
        return iter(self._generator)

def _get_beam_width(sampling_config):
    """ Returns the beam width of either a `trtllm.SamplingConfig` or a legacy `SamplingConfig` """
    return getattr(sampling_config, 'num_beams', None) or sampling_config.beam_width

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
