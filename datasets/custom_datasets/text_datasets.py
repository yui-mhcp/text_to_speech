
# Copyright (C) 2022 yui-mhcp project's author. All rights reserved.
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
import logging
import numpy as np
import pandas as pd

from functools import wraps

from loggers import timer
from utils.file_utils import load_json
from datasets.custom_datasets import add_dataset
from datasets.dataset_utils import _maybe_load_embedding
from datasets.custom_datasets.preprocessing import parse_nq_annots

logger  = logging.getLogger(__name__)

_siamese_renaming = {'sentence1' : 'text_x', 'sentence2' : 'text_y'}
_spaces = (' ', '\n', '\t')

TRANS   = 'Translation'
QnA_EN  = 'Q&A (English)'
QnA_FR  = 'Q&A (French)'
TEXT_COMP   = 'Text comparison'

def text_dataset_wrapper(name, task, ** default_config):
    def wrapper(dataset_loader):
        @timer(name = '{} loading'.format(name))
        @wraps(dataset_loader)
        def _load_and_process(directory, * args, ** kwargs):
            dataset = dataset_loader(directory, * args, ** kwargs)
            
            dataset['dataset_name'] = name
            
            return dataset
        
        add_dataset(name, processing_fn = _load_and_process, task = task, ** default_config)
        
        return _load_and_process
    return wrapper

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

def _select_answer(candidates, keep_mode):
    assert keep_mode in ('all', 'concensus', 'shortest', 'longest')
    
    if keep_mode == 'all':
        return list(set(candidates))
    elif keep_mode == 'concensus':
        counts = {}
        for a in candidates:
            counts.setdefault(a, 0)
            counts[a] += 1
        return sorted(counts.items(), key = lambda a: a[1], reverse = True)[0][0]
    
    return sorted(candidates, key = lambda a: len(a), reverse = keep_mode == 'longest')[0]
    

@text_dataset_wrapper(
    name    = 'CoQA', task = QnA_EN,
    train   = {'directory' : '{}/CoQA', 'subset' : 'train'},
    valid   = {'directory' : '{}/CoQA', 'subset' : 'dev'}
)
def preprocess_coqa_annots(directory, subset, keep_mode = 'longest', ** kwargs):
    assert subset in ('train', 'dev'), 'Unknown subset : {}'.format(subset)
    
    filename = os.path.join(directory, 'coqa-{}-v1.0.json'.format(subset))
    
    data = load_json(filename)['data']
    
    dataset = []
    for instance in data:
        base_infos = {
            'id' : instance['id'], 'context_id' : instance['filename'], 'title' : instance['filename'], 'context' : instance['story']
        }
        questions = {
            q['turn_id'] : q['input_text'] for q in instance['questions']
        }
        for a in instance['answers']:
            dataset.append({
                ** base_infos,
                'question'  : questions[a['turn_id']],
                'answers'   : a['input_text'] if keep_mode != 'all' else [a['input_text']]
            })
            
    return pd.DataFrame(dataset)

@text_dataset_wrapper(
    name = 'europarl', task = TRANS, directory = '{}/Europarl', base_name = 'europarl-v7.fr-en'
)
def preprocess_europarl_annots(directory, base_name, input_lang = 'en', output_lang = 'fr'):
    input_filename = os.path.join(directory, '{}.{}'.format(base_name, input_lang))
    output_filename = os.path.join(directory, '{}.{}'.format(base_name, output_lang))
    
    with open(input_filename, 'r', encoding = 'utf-8') as input_file:
        inputs = input_file.read().split('\n')
        
    with open(output_filename, 'r', encoding = 'utf-8') as output_file:
        outputs = output_file.read().split('\n')

    datas = [[inp, out] for inp, out in zip(inputs, outputs)]
    return pd.DataFrame(data = datas, columns = [input_lang, output_lang])

@text_dataset_wrapper(
    name    = 'NQ', task = QnA_EN,
    train   = {'directory' : '{}/NaturalQuestions', 'subset' : 'train'},
    valid   = {'directory' : '{}/NaturalQuestions', 'subset' : 'dev'}
)
def preprocess_nq_annots(directory,
                         subset     = 'train',
                         file_no    = -1,
                         
                         use_long_answer    = False,
                         include_document   = False,
                         allow_la   = True,
                         keep_mode  = 'longest',
                         
                         tqdm       = lambda x: x,
                         
                         ** kwargs
                        ):
    def select_short_answer(row):
        candidates = row['short_answers']
        if isinstance(candidates, list): candidates = [a for a in candidates if a in row['context']]
        
        if not isinstance(candidates, list) or len(candidates) == 0:
            if not allow_la and row['long_answer'] not in ('yes', 'no'): return ''
            candidates = [row['long_answer']]
        
        return _select_answer(candidates, keep_mode)
    
    if file_no == -1: file_no = list(range(50))
    if isinstance(file_no, (list, tuple)):
        return pd.concat([preprocess_nq_annots(
            directory, subset = subset, file_no = no, use_long_answer = use_long_answer,
            include_document = include_document, keep_mode = keep_mode, tqdm = tqdm, ** kwargs
        ) for no in file_no], ignore_index = True)
    
    dataset = parse_nq_annots(directory, subset = subset, file_no = file_no, tqdm = tqdm, ** kwargs)
    dataset = pd.DataFrame(dataset)
    if len(dataset) == 0: return dataset
    
    dataset.dropna(inplace = True)
    dataset.reset_index(drop = True, inplace = True)
    
    if include_document:
        dataset['titles']   = dataset['paragraphs'].apply(lambda para: [p['title'] for p in para])
        dataset['paragraphs']   = dataset['paragraphs'].apply(lambda para: [p['text'] for p in para])
    else:
        dataset.pop('paragraphs')
    
    if use_long_answer:
        dataset['answers'] = dataset.apply(
            lambda row: row['long_answer'] if row['long_answer'] in row['context'] or row['long_answer'] in ('yes', 'no') else select_short_answer(row),
            axis = 'columns'
        )
    else:
        dataset['answers'] = dataset.apply(
            lambda row: select_short_answer(row),
            axis = 'columns'
        )
    
    dataset = dataset[dataset['answers'] != '']
    dataset.pop('long_answer')
    dataset.pop('short_answers')

    return dataset

@text_dataset_wrapper(
    name    = 'NewsQA', task = QnA_EN,
    train   = {'directory' : '{}/newsqa', 'subset' : 'train'},
    valid   = {'directory' : '{}/newsqa', 'subset' : 'dev'}
)
def preprocess_newsqa_annots(directory, subset, keep_mode = 'longest', ** kwargs):
    assert subset in ('train', 'dev', 'test'), "Unknown subset : {}".format(subset)
    
    filename = os.path.join(directory, 'combined-newsqa-data-v1.json')
    
    data = load_json(filename)['data']
    
    dataset = []
    for i, instance in enumerate(data):
        if instance['type'] != subset: continue
        if '--' in instance['text']:
            title   = instance['text'].split('--')[0].strip()
            context = '--'.join(instance['text'].split('--')[1:]).strip().replace('\n\n\n\n', '\n')
        else:
            title, context = '', instance['text']
            
        base_infos = {
            'id'            : instance['storyId'],
            'context_id'    : i,
            'title'         : title,
            'context'       : context
        }
        for q in instance['questions']:
            if 's' not in q['consensus']: continue
            
            answer = instance['text'][q['consensus']['s'] : q['consensus']['e']]
            dataset.append({
                ** base_infos,
                'question'  : q['q'],
                'answers'   : answer if keep_mode != 'all' else [answer]
            })
            
    return pd.DataFrame(dataset)

@text_dataset_wrapper(
    name    = 'PARADE', task = TEXT_COMP,
    train   = {'directory' : '{}/PARADE', 'subset' : 'train'},
    valid   = {'directory' : '{}/PARADE', 'subset' : 'validation'}
)
def preprocess_parade_annots(directory, subset = 'train', rename_siamese = True, ** kwargs):
    filename = os.path.join(directory, 'PARADE_{}.txt'.format(subset))
    
    dataset = pd.read_csv(filename, sep = '\t')
    dataset['Binary labels'] = dataset['Binary labels'].astype(np.bool)
    
    if rename_siamese:
        dataset = dataset.rename(
            columns = {'Binary labels' : 'same', 'Definition1' : 'text_x', 'Definition2' : 'text_y', 'Entity' : 'title'}
        )
    
    return dataset

@text_dataset_wrapper(
    name    = 'PAWS', task = TEXT_COMP,
    train   = {'directory' : '{}/PAWS', 'subset' : 'train'},
    valid   = {'directory' : '{}/PAWS', 'subset' : 'dev'}
)
def preprocess_paws_annots(directory, subset = 'train', rename_siamese = True, ** kwargs):
    filename = os.path.join(directory, '{}.tsv'.format(subset))
    
    dataset = pd.read_csv(filename, sep = '\t')
    dataset['label'] = dataset['label'].astype(np.bool)
    
    if rename_siamese:
        dataset = dataset.rename(
            columns = {'label' : 'same', 'sentence1' : 'text_x', 'sentence2' : 'text_y'}
        )
    
    return dataset

@text_dataset_wrapper(
    name    = 'QAngaroo', task = QnA_EN,
    train   = {'directory' : '{}/qangaroo_v1.1', 'subset' : 'train'},
    valid   = {'directory' : '{}/qangaroo_v1.1', 'subset' : 'dev'}
)
def preprocess_qangaroo_annots(directory, subset, mode = 'wiki', keep_mode = 'longest', ** kwargs):
    assert mode in ('wiki', 'med')
    assert subset in ('train', 'dev'), "Unknown subset : {}".format(subset)
    
    filename = os.path.join(directory, mode + 'hop', '{}.json'.format(subset))
    
    data = load_json(filename)

    dataset = []
    for i, instance in enumerate(data):
        dataset.append({
            'id' : instance['id'],
            'titles'    : [''] * len(instance['supports']),
            'context'   : '\n\n'.format(instance['supports']),
            'paragraphs'    : instance['supports'],
            'question'  : instance['query'].replace('_', ' '),
            'answers'   : instance['answer'] if keep_mode != 'all' else [instance['answer']]
        })
    
    return pd.DataFrame(dataset)

@text_dataset_wrapper(
    name    = 'QQP', task = TEXT_COMP, directory = '{}/QQP', subset = 'train'
)
def preprocess_qqp_annots(directory, subset = 'train', rename_siamese = True, ** kwargs):
    filename = os.path.join(directory, '{}.csv'.format(subset))
    
    dataset = pd.read_csv(filename, index_col = 0)
    dataset = dataset.dropna('index')
    dataset['is_duplicate'] = dataset['is_duplicate'].astype(np.bool)
    
    if rename_siamese:
        dataset = dataset.rename(
            columns = {'is_duplicate' : 'same', 'question1' : 'text_x', 'question2' : 'text_y'}
        )
    
    return dataset

@text_dataset_wrapper(
    name    = 'SNLI', task = TEXT_COMP,
    train   = {'directory' : '{}/snli_1.0', 'subset' : 'train'},
    valid   = {'directory' : '{}/snli_1.0', 'subset' : 'dev'}
)
def preprocess_snli_annots(directory, subset, version = '1.0', skip_parsed = True,
                           skip_sub_labels = True, skip_ukn_label = True, skip_id = True,
                           rename = _siamese_renaming, ** kwargs):
    if isinstance(subset, (list, tuple)):
        return pd.concat([preprocess_snli_annots(
            directory, sub, version = version, skip_parsed = skip_parsed,
            skip_sub_labels = skip_sub_labels, ** kwargs
        )] for sub in subset)
    
    filename = os.path.join(directory, 'snli_{}_{}.txt'.format(version, subset))
    dataset = pd.read_csv(filename, sep  = '\t')
    
    if skip_parsed or skip_id or skip_sub_label:
        for col in dataset.columns:
            if 'parse' in col and skip_parsed: dataset.pop(col)
            elif 'ID' in col and skip_id: dataset.pop(col)
            elif col != 'gold_label' and 'label' in col and skip_sub_labels: dataset.pop(col)
    
    if skip_ukn_label:
        dataset = dataset[dataset['gold_label'] != '-']
    
    dataset['same'] = dataset['gold_label'] == 'entailment'
    rename.setdefault('gold_label', 'label')
    dataset = dataset.rename(columns = rename)
    
    return dataset.dropna()

@text_dataset_wrapper(
    name    = 'STS', task = TEXT_COMP,
    train   = {'directory' : '{}/sts_benchmark', 'subset' : 'train'},
    valid   = {'directory' : '{}/sts_benchmark', 'subset' : 'dev'}
)
def process_sts_annots(directory, subset, rename = _siamese_renaming, ** kwargs):
    if isinstance(subset, (list, tuple)):
        return pd.concat([preprocess_snli_annots(
            directory, sub, ** kwargs
        )] for sub in subset)
    
    filename = os.path.join(directory, 'sts-{}.csv'.format(subset))

    with open(filename, 'r', encoding = 'utf-8') as file:
        lines = file.read().split('\n')

    dataset = pd.DataFrame(
        [l.split('\t')[:7] for l in lines if len(l) > 0],
        columns = ['category', 'type', 'date', 'id', 'score', 'sentence1', 'sentence2']
    )
    
    dataset = dataset.rename(columns = rename)
    
    return dataset

@text_dataset_wrapper(
    name    = 'SQUAD', task = QnA_EN,
    train   = {'directory' : '{}/SQUAD2.0', 'subset' : 'train'},
    valid   = {'directory' : '{}/SQUAD2.0', 'subset' : 'dev'}
)
def preprocess_SQUAD_annots(directory, subset, version = '2.0', skip_impossible = False,
                            clean_text = True, keep_mode = 'longest', ** kwargs):
    assert keep_mode in ('all', 'longest', 'shortest', 'one_per_line')
    
    filename = subset
    if version: filename += '-v{}'.format(version)
    filename = os.path.join(directory, '{}.json'.format(filename))
    metadata = load_json(filename)['data']
    
    dataset = []
    contexts = {}
    for data in metadata:
        for para in data['paragraphs']:
            if clean_text: para = _clean_paragraph(para)
            contexts.setdefault(para['context'], len(contexts))
            for qa in para['qas']:
                _base_infos = {
                    'title' : data['title'], 'context_id' : contexts[para['context']],
                    'context' : para['context'], 'question' : qa['question']
                }
                
                if  qa.get('is_impossible', False):
                    if not skip_impossible:
                        dataset.append({
                            ** _base_infos, 'answers' : '', 'answer_start' : -1
                        })
                    continue
                
                if keep_mode == 'one_per_line':
                    for a in qa['answers']:
                        dataset.append({
                            ** _base_infos, 'answers' : a['text'], 'answer_start' : a['answer_start']
                        })
                    continue
                
                candidates = [a['text'] for a in qa['answers']]
                dataset.append({
                    ** _base_infos, 'answers' : _select_answer(candidates, keep_mode = keep_mode)
                })
    
    dataset = pd.DataFrame(dataset)
    
    return dataset

@text_dataset_wrapper(
    name    = 'TriviaQA', task = QnA_EN,
    train   = {'directory' : '{}/TriviaQA', 'subset' : 'train'},
    valid   = {'directory' : '{}/TriviaQA', 'subset' : 'dev'}
)
def preprocess_triviaqa_annots(directory, unfiltered = False, wikipedia = True,
                               load_context = False, keep_doc_mode = 'one_per_line', subset = 'train',
                               tqdm = lambda x: x, ** kwargs):
    def get_contexts(contexts):
        result = []
        for c in contexts:
            if 'Filename' not in c: continue
            f = c['Filename']
            for char in ('?', ':', '*', '"'): f = f.replace(char, '_')
            f = os.path.join(directory, 'evidence', prefix, f)
            if not os.path.exists(f):
                logger.warning("File for context {} does not exist !".format(c))
            text = None
            if load_context:
                with open(f, 'r', encoding = 'utf-8') as file:
                    text = file.read()
            
            result.append({
                'context_id'    : c['Title'],
                'filename' : f,
                'title'    : c['Title'],
                'context'  : text
            })
        return result

    if unfiltered: wikipedia = False
    prefix = 'wikipedia' if wikipedia else 'web'
    
    filename = '{}-{}.json'.format(prefix, subset)
    if unfiltered:
        filename = os.path.join('triviaqa-unfiltered', 'unfiltered-' + filename)
    else:
        filename = os.path.join('qa', filename)
    
    filename = os.path.join(directory, filename)

    data = load_json(filename)['Data']
    
    metadata = []
    for qa in tqdm(data):
        contexts = get_contexts(qa.get('EntityPages', []))
        if len(contexts) == 0: continue
        
        if keep_doc_mode == 'one_per_line':
            for i, c in enumerate(contexts):
                metadata.append({
                    'id'       : qa['QuestionId'] + '_doc_{}'.format(i),
                    'question' : qa['Question'],
                    'answers'  : qa['Answer']['Value'],
                    ** c
                })
        else:
            c = contexts[0] if keep_doc_mode == 'first' else contexts[-1]
            metadata.append({
                'id'       : qa['QuestionId'],
                'question' : qa['Question'],
                'answers'  : qa['Answer']['Value'],
                ** c
            })
    
    return pd.DataFrame(metadata)


add_dataset(
    'French SQUAD', processing_fn = 'squad', task = QnA_FR,
    train   = {
        'directory' : '{}/French_SQUAD', 'subset' : 'SQuAD',
        'version' : '1.1-train_fr_ss999_awstart2_net'
    },
    valid   = {
        'directory' : '{}/French_SQUAD', 'subset' : 'SQuAD',
        'version' : '1.1-dev_fr_ss999_awstart2_net'
    }
)

add_dataset(
    'FQUAD', processing_fn = 'squad', task = QnA_FR,
    train   = {'directory' : '{}/FQUAD1.0', 'subset' : 'train', 'version' : ''},
    valid   = {'directory' : '{}/FQUAD1.0', 'subset' : 'valid', 'version' : ''}
)

add_dataset(
    'piaf', processing_fn = 'squad', task = QnA_FR,
    directory = '{}/piaf', subset = 'piaf', version = '1.1'
)
