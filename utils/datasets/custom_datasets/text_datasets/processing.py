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

from functools import wraps

from loggers import timer
from .. import add_dataset

def text_dataset_wrapper(name, task, ** default_config):
    def wrapper(dataset_loader):
        @timer(name = '{} loading'.format(name if isinstance(name, str) else name[0]))
        @wraps(dataset_loader)
        def _load_and_process(directory,
                              * args,
                              
                              answer_mode   = 'longest',
                              skip_impossible   = False,
                              
                              ** kwargs
                             ):
            import pandas as pd
            
            dataset = dataset_loader(directory, * args, ** kwargs)
            
            if any('answers' in d for d in dataset):
                dataset = _select_answer(dataset, answer_mode, skip_impossible)
            
            dataset = pd.DataFrame(dataset)
            dataset['dataset_name'] = name
            
            return dataset
        
        add_dataset(name, processing_fn = _load_and_process, task = task, ** default_config)
        
        return _load_and_process
    return wrapper

def _select_answer(dataset, mode, skip_impossible = False):
    new_data = []
    for data in dataset:
        if not data.get('answers', None):
            if skip_impossible: continue
            new_data.append(data)
        elif mode == 'all':
            new_data.append(data)
        elif mode in ('longest', 'shortest'):
            indexes = sorted(range(len(data['answers'])), key = lambda i: len(data['answers'][i]))
            if mode == 'longest': indexes = list(reversed(indexes))
            data.update({
                k : v[indexes[0]] for k, v in data.items() if 'answer' in k
            })
            new_data.append(data)
        elif mode == 'one_per_line':
            for i in range(len(data['answers'])):
                data_i = data.copy()
                data_i.update({
                    k : v[i] for k, v in data_i.items() if 'answer' in k
                })
                new_data.append(data_i)
        else:
            raise ValueError('Unsupported `answer_mode` : {}'.format(mode))
            
    return new_data