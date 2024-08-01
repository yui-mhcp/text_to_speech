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
import logging
import numpy as np

from .. import Task
from utils import load_json
from .processing import text_dataset_wrapper

logger      = logging.getLogger(__name__)

@text_dataset_wrapper(
    name    = 'SQUAD', task = Task.QA,
    train   = {'directory' : '{}/SQUAD2.0', 'subset' : 'train'},
    valid   = {'directory' : '{}/SQUAD2.0', 'subset' : 'dev'}
)
def load_data(directory, *, subset, version = '2.0', clean_text = True, ** kwargs):
    filename = subset
    if version: filename += '-v{}'.format(version)
    filename = os.path.join(directory, '{}.json'.format(filename))
    
    if not os.path.exists(filename):
        raise ValueError('Filename {} does not exist !'.format(filename))
    
    metadata = load_json(filename)['data']
    
    dataset = []
    contexts = {}
    for data in metadata:
        for para in data['paragraphs']:
            if clean_text: para = _clean_paragraph(para)
            contexts.setdefault(para['context'], len(contexts))
            
            for qa in para['qas']:
                if qa.get('is_impossible', False):
                    answers = {'answers' : [], 'answer_start' : []}
                else:
                    answers = {
                        'answers' : [a['text'] for a in qa['answers']],
                        'answer_start'  : [a['answer_start'] for a in qa['answers']]
                    }
                
                dataset.append({
                    'title'     : data['title'],
                    'context'   : para['context'],
                    'context_id'    : contexts[para['context']],
                    'question'  : qa['question'],
                    ** answers
                })
    
    return dataset

def _clean_paragraph(para):
    from utils.text.cleaners import remove_control
    
    para['context'] = remove_control(para['context'])
    for qa in para['qas']:
        qa['question']  = remove_control(qa['question'])
        qa['answers']   = [
            _clean_answer(para['context'], a['text'], a['answer_start']) for a in qa['answers']
        ]
    return para
        
def _clean_answer(context, answer, answer_start):
    from utils.text.cleaners import remove_control
    
    answer = remove_control(answer).strip()
    if answer not in context:
        raise ValueError("Invalid answer {} !".format(answer))
    
    return {
        'text' : answer, 'answer_start' : answer_start, 'answer_end' : answer_start + len(answer)
    }
