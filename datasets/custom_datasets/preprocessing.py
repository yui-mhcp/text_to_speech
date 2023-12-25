
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
import json
import gzip
import logging
import pandas as pd

from tqdm import tqdm
from multiprocessing import cpu_count

from utils import load_json, dump_json
from utils.thread_utils import Consumer

logger  = logging.getLogger(__name__)

def _resample_dataset(dataset, new_rate, max_workers = cpu_count(), tqdm = None):
    from utils.audio import resample_file

    new_col = 'wavs_{}'.format(new_rate)
    
    processed = dataset[new_col].apply(lambda f: os.path.exists(f))
    
    to_process = dataset[~processed]
    
    logger.info("Resampling dataset to {}\n  {} files already processed\n  {} files to process".format(
        new_rate, processed.sum(), len(to_process)
    ))
    
    if len(to_process) > 0:
        callback = None
        if tqdm is not None:
            tqdm = tqdm(total = len(to_process), unit = 'file')
            callback    = lambda it: tqdm.update()

        pool = Consumer(resample_file, keep_result = False, max_workers = max_workers)
        pool.start()

        for idx, row in to_process.iterrows():
            pool(row['filename'], new_rate = new_rate, filename_out = row[new_col], callback = callback)

        pool.join()
    
    return dataset

def resample_commonvoice(directory, new_rate, file = 'validated.tsv', ** kwargs):
    new_col = 'wavs_{}'.format(new_rate)
    
    os.makedirs(os.path.join(directory, new_col), exist_ok = True)
    
    dataset = pd.read_csv(os.path.join(directory, file), sep = '\t')
    dataset['filename'] = dataset['path'].apply(lambda f: os.path.join(directory, 'clips', f))
    
    dataset[new_col] = dataset['filename'].apply(
        lambda f: f.replace('clips', new_col).replace('.mp3', '.wav')
    )
    
    _resample_dataset(dataset, new_rate, ** kwargs)
    
    return dataset

def resample_siwis(directory, new_rate, langue = 'fr', parts = [1, 2, 3, 5], ** kwargs):
    from datasets.custom_datasets.audio_datasets import preprocess_SIWIS_annots
    new_col = 'wavs_{}'.format(new_rate)
        
    dataset = preprocess_SIWIS_annots(directory, langue = langue, parts = parts)
    
    for part in parts:
        new_wav_dir = os.path.join(
            directory, langue, new_col, 'part{}'.format(part)
        )
        os.makedirs(new_wav_dir, exist_ok = True)

    dataset[new_col] = dataset['filename'].apply(
        lambda f: f.replace('wavs', new_col)
    )
    
    _resample_dataset(dataset, new_rate, ** kwargs)
    
    return dataset

def resample_kaggle(directory, new_rate, ** kwargs):
    from datasets.custom_datasets.audio_datasets import preprocess_kaggle_single_speaker_annots
    new_col = 'wavs_{}'.format(new_rate)
    
    dataset = preprocess_kaggle_single_speaker_annots(directory, ** kwargs)
    
    for spk_id in dataset['id'].unique():
        os.makedirs(os.path.join(directory, '{}_{}'.format(spk_id, new_rate)), exist_ok = True)

    dataset[new_col] = dataset['filename'].apply(
        lambda f: f.replace(os.path.basename(f).split('.')[0], new_col, 1)
    )
    
    _resample_dataset(dataset, new_rate, ** kwargs)
    
    return dataset

def resample_voxforge(directory, new_rate, langue = 'fr', max_workers = cpu_count()):
    def rename_file(f):
        ext = f.split('.')[-1]
        if ext == 'flac':
            f = f.replace('flac', 'wav')
        return f.replace('{}wav'.format(os.path.sep), '{}{}'.format(os.path.sep, new_col))
    
    from datasets.custom_datasets.audio_datasets import preprocess_VoxForge_annots

    dataset = preprocess_VoxForge_annots(directory, langue = langue)

    new_col = 'wavs_{}'.format(new_rate)
    
    directory = os.path.join(directory, langue)
    for sub_dir in os.listdir(directory):
        os.makedirs(os.path.join(directory, sub_dir, new_col), exist_ok = True)
    
    dataset[new_col] = dataset['filename'].apply(rename_file)
    
    dataset = _resample_dataset(dataset, new_rate)
        
    return dataset

def resample_mls(path, new_rate, langue = 'fr', subset = ['train', 'test', 'dev'], 
                 ** kwargs):
    from datasets.custom_datasets.audio_datasets import preprocess_mls_annots
    
    if not isinstance(subset, (tuple, list)): subset = [subset]
    new_col = 'wavs_{}'.format(new_rate)
    
    for sub in subset:
        dataset = preprocess_mls_annots(path, langue = langue, subset = sub)

        audio_dir = os.path.join(path, langue, sub, 'audio')
        for spk_dir in os.listdir(audio_dir):
            for book_dir in os.listdir(os.path.join(audio_dir, spk_dir)):
                os.makedirs(
                    os.path.join(path, langue, sub, new_col, spk_dir, book_dir),
                    exist_ok = True
                )

        dataset[new_col] = dataset['filename'].apply(
            lambda f: f.replace('.opus', '.wav').replace('audio', new_col)
        )

        dataset = _resample_dataset(dataset, new_rate, ** kwargs)
        
    return dataset

def resample_librispeech(directory, new_rate,
                         subset = ['train-clean-100', 'train-clean-360', 'test-clean'],
                         ** kwargs):
    from datasets.custom_datasets.audio_datasets import preprocess_LibriSpeech_annots

    if not isinstance(subset, (tuple, list)): subset = [subset]
    new_col = 'wavs_{}'.format(new_rate)
    
    for sub in subset:
        dataset = preprocess_LibriSpeech_annots(directory, subsets = sub)

        subset_dir = os.path.join(directory, sub)
        for spk_dir in os.listdir(subset_dir):
            for book_dir in os.listdir(os.path.join(subset_dir, spk_dir)):
                os.makedirs(
                    os.path.join(directory, new_col + '_' + sub, spk_dir, book_dir),
                    exist_ok = True
                )

        dataset[new_col] = dataset['filename'].apply(
            lambda f: f.replace('.flac', '.wav').replace(sub, new_col + '_' + sub)
        )

        dataset = _resample_dataset(dataset, new_rate, ** kwargs)
        
    return dataset
def parse_nq_annots(path,
                    subset = 'train',
                    file_no = 0,
                    
                    clean_ref = True,
                    
                    overwrite = False,
                    tqdm = lambda x: x,
                    ** kwargs
                   ):
    from utils.text.document_parser.html_parser import _wiki_cleaner, parse_html

    def file_generator(file):
        for l in file:
            yield l.decode('utf-8')
    
    def process_html(html_text, ** kwargs):
        parsed = parse_html(html_text, remove_pattern = _wiki_cleaner if clean_ref else None, ** kwargs)
        return parsed
    
    def get_answer_html(encoded_html, answer):
        if answer['start_token'] == -1 or answer['end_token'] == -1: return ''
        return encoded_html[answer['start_byte'] : answer['end_byte']].decode('utf-8')
    
    def get_paragraph_idx(context, paragraphs):
        idx = [i for i, p in enumerate(paragraphs) if context['title'] == p['title']]
        return idx[0] if len(idx) != 0 else -1
    
    def get_valid(encoded_html, containings, answer):
        if len(containings) == 0: return None
        if len(containings) == 1: return containings[0]
        p_idx = encoded_html[: answer['end_byte']].decode('utf-8').lower().count('<p>') - 1
        valid = None
        if p_idx >= 0:
            for p in containings:
                if p_idx in range(p['start_p_idx'], p['end_p_idx']):
                    valid = p
                    break
        
        return containings[0] if valid is None else valid
    
    def extract_answer(document_html, encoded, paragraphs, answer, candidates = None):
        answer_html = get_answer_html(encoded, answer)
        if not answer_html: return None, None, None
        answer_text = process_html(answer_html)
        if not answer_text: return None, None, None
        answer_text = answer_text[0]['text']
        
        if paragraphs is None: paragraphs = process_html(document_html)

        containing = [p for p in paragraphs if answer_text in p['text']]
        valid = get_valid(encoded, containing, answer)
        
        if valid is None and candidates is not None and answer.get('candidate_index', -1) != -1:
            candidate = process_html(get_answer_html(encoded, candidates[answer['candidate_index']]), is_sub_document = True)
            if candidate:
                containing = [p for p in paragraphs if candidate[0]['text'] in p['text']]
                valid = containing[0] if len(containing) > 0 else None
        
        return paragraphs, valid, answer_text
    
    filename = os.path.join('v1.0', subset, 'nq-{}-{:02d}.jsonl.gz'.format(subset, file_no))
    filename = os.path.join(path, filename)
    
    simplified_filename = filename.replace('.jsonl.gz', '-parsed.json')
    
    if os.path.exists(simplified_filename) and not overwrite: return load_json(simplified_filename)
    if not os.path.exists(filename): return []
    
    print("Processing file {}...".format(filename))
    dataset = []
    with gzip.open(filename) as file:
        for i, line in enumerate(tqdm(file_generator(file))):
            try:
                parsed = json.loads(line)
            except Exception as e:
                print("Error at index {} ({}) with line :\n{}".format(i, e, line))
                continue
            
            paragraphs = None
            encoded    = parsed['document_html'].encode('utf-8')
            
            infos = {'question' : parsed['question_text'], 'long_answer' : '', 'short_answers' : []}
            for annot in parsed['annotations']:
                if len(infos['long_answer']) == 0:
                    la = annot['long_answer']

                    paragraphs, p, la_text = extract_answer(
                        parsed['document_html'], encoded, paragraphs, la, candidates = parsed['long_answer_candidates']
                    )
                    if annot['yes_no_answer'] != 'NONE':
                        la_text = annot['yes_no_answer'].lower()
                    
                    if la_text:
                        infos.update({'long_answer' : la_text, 'paragraphs' : paragraphs})
                        if p is not None:
                            infos.update({'title' : p['title'], 'context' : p['text']})
                
                for sa in annot['short_answers']:
                    paragraphs, p, sa_text = extract_answer(parsed['document_html'], encoded, paragraphs, sa)
                    if sa_text:
                        if not infos.get('title', '') and p is not None:
                            infos.update({'title' : p['title'], 'context' : p['text'], 'paragraphs' : paragraphs})
                        infos.setdefault('short_answers', []).append(sa_text)

            if infos.get('context', ''):
                infos['valid_idx'] = get_paragraph_idx({'title' : infos['title'], 'text' : infos['context']}, infos['paragraphs'])
                if infos['valid_idx'] == -1:
                    raise ValueError("Valid idx == -1 !")
                dataset.append(infos)
        
    dump_json(simplified_filename, dataset)
    
    return dataset
