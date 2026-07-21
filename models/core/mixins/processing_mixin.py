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

import inspect

from loggers import timer
from utils.keras import ops

class ModelProcessingMixin:
    """
        Input / output processing pipeline of a `BaseModel` (see the `BaseModel` docstring for the
        full ordering : `prepare` -> `filter` -> `process` -> `augment` -> `process_batch`).

        `_init_processing_functions` auto-wires the `{prefix}_data` functions from the `{prefix}_input`
        / `{prefix}_output` ones defined by the sub-class, so a sub-class usually only implements the
        per-side (`_input` / `_output`) hooks.
    """

    @timer(debug = True)
    def _init_processing_functions(self):
        # Resolve `get_output` -> `prepare_output` / `process_output` FIRST : the generic
        # `{prefix}_data` closures built below capture `{prefix}_output` at build time, so any later
        # (re)assignment of `prepare_output` / `process_output` would be silently ignored.
        if hasattr(self, 'get_output'):
            if hasattr(self, 'prepare_input') and not hasattr(self, 'prepare_output'):
                self.prepare_output = self.get_output
            elif hasattr(self, 'process_input') and not hasattr(self, 'process_output'):
                self.process_output = self.get_output
        else:
            self.get_output = self._get_output

        for prefix in ('prepare', 'augment', 'process', 'process_batch'):
            if not hasattr(self, f'{prefix}_data') and (
                hasattr(self, f'{prefix}_input') or hasattr(self, f'{prefix}_output')
            ):
                setattr(self, f'{prefix}_data', _build_generic_processing(
                    getattr(self, f'{prefix}_input', lambda inp, ** kwargs: inp),
                    getattr(self, f'{prefix}_output', lambda out, ** kwargs: out),
                    name = f'{prefix}_data'
                ))
        if hasattr(self, 'filter_input') or hasattr(self, 'filter_output'):
            self.filter_data = self._filter_data

    def get_input(self, data, ** kwargs):
        """ Sequentially calls `prepare_input` then `process_input` if defined """
        inputs = data
        if hasattr(self, 'prepare_input'): inputs = self.prepare_input(inputs, ** kwargs)
        if hasattr(self, 'process_input'): inputs = self.process_input(inputs, ** kwargs)
        return inputs

    def _get_output(self, data, ** kwargs):
        """ Sequentially calls `prepare_output` then `process_output` if defined """
        output = data
        if hasattr(self, 'prepare_output'): output = self.prepare_output(output, ** kwargs)
        if hasattr(self, 'process_output'): output = self.process_output(output, ** kwargs)
        return output

    def _filter_data(self, inputs, output):
        valid_inp = self.filter_input(inputs) if hasattr(self, 'filter_input') else True
        valid_out = self.filter_output(output) if hasattr(self, 'filter_output') else True
        return ops.logical_and(valid_inp, valid_out)

    def get_dataset_config(self, mode, ** kwargs):
        """ Prepares the arguments for `prepare_dataset` to build the processing pipeline """
        assert mode in ('train', 'valid', 'predict')

        suffix = 'input' if mode == 'predict' else 'data'
        for prefix in ('augment_raw', 'prepare', 'filter', 'process', 'augment', 'process_batch'):
            if 'augment' in prefix and mode != 'train': continue

            key = f'{prefix}_fn'
            if key in kwargs: continue

            if hasattr(self, f'{prefix}_{suffix}'):
                kwargs[key] = getattr(self, f'{prefix}_{suffix}')

        if mode == 'train':
            kwargs.setdefault('shuffle', True)
        elif mode == 'valid':
            kwargs.update({'shuffle' : False})
        elif mode == 'predict':
            kwargs.update({'shuffle' : False, 'cache' : False})

        # mode-specific overrides : `{mode}_<key>` takes precedence over `<key>` (e.g. `train_batch_size`)
        overrides = {
            k[len(mode) + 1:] : v for k, v in list(kwargs.items())
            if k.startswith(mode + '_') and v is not None
        }
        kwargs.update(overrides)
        return kwargs

    def prepare_dataset(self, dataset, mode, ** kwargs):
        from utils.datasets import prepare_dataset

        return prepare_dataset(dataset, ** self.get_dataset_config(mode, ** kwargs))


def _build_generic_processing(input_processing, output_processing, name):
    if name.startswith('prepare'):
        def inner(* data, ** kwargs):
            if len(data) == 1:
                if isinstance(data[0], tuple) and len(data[0]) == 2:
                    inputs, output = data[0][0], data[0][1]
                else:
                    inputs, output = data[0], data[0]
            elif len(data) == 2:
                inputs, output = data
            else:
                raise RuntimeError('input `data` should either be single element (`dict`) either 2-elements (`inputs` and `outputs`). Get {} elements instead :\n{}'.format(len(data), data))

            inputs = input_processing(inputs, ** kwargs)
            if has_input_kwarg: kwargs['inputs'] = inputs
            return inputs, output_processing(output, ** kwargs)
    else:
        def inner(inputs, output, ** kwargs):
            inputs = input_processing(inputs, ** kwargs)
            if has_input_kwarg: kwargs['inputs'] = inputs
            return inputs, output_processing(output, ** kwargs)

    has_input_kwarg = 'inputs' in inspect.signature(output_processing).parameters
    inner.__name__ = name
    return inner
