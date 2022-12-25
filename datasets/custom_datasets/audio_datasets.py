
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
import librosa
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from unidecode import unidecode
from multiprocessing import cpu_count

from loggers import timer
from utils import load_json, load_embedding
from datasets.dataset_utils import prepare_dataset

logger = logging.getLogger(__name__)

def audio_dataset_wrapper(name, task, ** default_config):
    def wrapper(dataset_loader):
        @timer(name = '{} loading'.format(name))
        def _load_and_process(directory, * args, add_audio_time = False, ** kwargs):
            dataset = dataset_loader(directory, * args, ** kwargs)
            
            dataset = _add_default_rate(dataset)

            if 'time' not in dataset.columns:
                if add_audio_time: dataset = _add_audio_time(dataset)
                else: dataset['time'] = -1.
            
            dataset = _maybe_load_embedding(directory, dataset, ** kwargs)
            dataset['dataset_name'] = name
            if 'id' not in dataset.columns: dataset['id'] = name
            
            return dataset
        
        from datasets.custom_datasets import add_dataset
        
        fn = _load_and_process
        fn.__name__ = dataset_loader.__name__
        fn.__doc__  = dataset_loader.__doc__
        
        add_dataset(name, processing_fn = fn, task = task, ** default_config)
        
        return fn
    return wrapper

def _maybe_load_embedding(directory, dataset, ** kwargs):
    if 'embedding' in dataset.columns: return dataset
    if 'embedding_name' in kwargs or 'embedding_dim' in kwargs:
        return load_embedding(directory, dataset = dataset, ** kwargs)
    return dataset

def _add_audio_time(dataset):
    dataset['time'] = dataset['filename'].apply(lambda f: librosa.get_duration(filename = f))
    return dataset

def _add_default_rate(dataset):
    if len(dataset) == 0: raise ValueError("Dataset is empty !")
        
    default_rate = librosa.get_samplerate(dataset.at[0, 'filename'])
    dataset['wavs_{}'.format(default_rate)] = dataset['filename']
    
    return dataset

@audio_dataset_wrapper(name = 'siwis', task = ['tts', 'stt'], directory = '{}/SIWIS')
def preprocess_SIWIS_annots(directory, lang = 'fr', parts = [1, 2, 3, 5], ** kwargs):
    base_dir = os.path.join(directory, lang)
    
    text_audio_pairs = []
    for part in parts:
        txt_dir = os.path.join(base_dir, 'text', 'part{}'.format(part))
        
        for filename in os.listdir(txt_dir):
            wav_filename = os.path.join(base_dir, 'wavs', 'part{}'.format(part), filename[:-3] + 'wav')
            txt_filename = os.path.join(txt_dir, filename)

            with open(txt_filename, 'r', encoding = 'utf-8') as file:
                text = file.read()
            
            text_audio_pairs.append({'text' : text, 'filename' : wav_filename})
    
    dataset = pd.DataFrame(text_audio_pairs)
    
    processed_names = [f for f in os.listdir(base_dir) if 'mels_' in f or 'wavs_' in f]
    for col_name in processed_names:
        dataset[col_name] = dataset['filename'].apply(
            lambda f: f.replace('wavs', col_name)
        )
        if 'mels_' in col_name:
            dataset[col_name] = dataset[col_name].apply(lambda f: f + '.npy')
    
    return _maybe_load_embedding(base_dir, dataset, ** kwargs)

@audio_dataset_wrapper(name = 'VoxForge', task = ['tts', 'stt'], directory = '{}/VoxForge')
def preprocess_VoxForge_annots(directory, lang = 'fr', ** kwargs):
    def process_speaker(main_dir, name):
        speaker_dir = os.path.join(main_dir, name)
        filename = os.path.join(speaker_dir, 'etc', 'prompts-original')
        
        with open(filename, 'r', encoding = 'utf-8') as file:
            lines = file.read().split('\n')
        
        speaker_infos = []
        sub_dirs = os.listdir(speaker_dir)
        original_audio_dir = 'wav' if 'wav' in sub_dirs else 'flac'
        ext = original_audio_dir
        for l in lines:
            if l == '': continue
            l = l.split(' ')
            # Add original informations
            audio_name, text = unidecode(l[0]), ' '.join(l[1:])
            infos = {
                'id' : name,
                'filename' : os.path.join(speaker_dir, original_audio_dir, audio_name) + '.' + ext,
                'text' : text
            }
            
            # Add additional preprocessed files (like wavs_22050 if processed)
            for sub_dir in sub_dirs:
                if sub_dir in ('etc', original_audio_dir) or not os.path.isdir(os.path.join(speaker_dir, sub_dir)): continue
                infos[sub_dir] = os.path.join(speaker_dir, sub_dir, audio_name) + '.wav'
            
            speaker_infos.append(infos)
        
        return speaker_infos
    
    directory = os.path.join(directory, lang)

    data = []
    for name in os.listdir(directory):
        data += process_speaker(directory, name)

    return pd.DataFrame(data)

@audio_dataset_wrapper(name = 'common voice', task = ['tts', 'stt'], directory = '{}/CommonVoice')
def preprocess_CommonVoice_annots(directory, file = 'validated.tsv', 
                                  dropna = False, sexe = None, age = None, 
                                  accent = None, pop_down = True, pop_votes = True, 
                                  ** kwargs):
    def filter_col(dataset, col, values):
        if values is None: return dataset
        if not isinstance(values, (list, tuple)): values = [values]
        return dataset[dataset[col].apply(lambda v: any(val in str(v).lower() for val in values))]
    
    new_columns = {
        'client_id' : 'id', 'path' : 'filename', 'sentence' : 'text', 'gender' : 'sex'
    }
    
    dataset = pd.read_csv(os.path.join(directory, file), sep = '\t')
    if 'index' in dataset: dataset.pop('index')
    
    dataset['path'] = dataset['path'].apply(lambda f: os.path.join(directory, 'clips', f))
    
    for sub_dir in os.listdir(directory):
        if not sub_dir.startswith('wavs_'): continue
        dataset[sub_dir] = dataset['path'].apply(
            lambda f: f.replace('clips', sub_dir).replace('.mp3', '.wav')
        )

    if dropna: dataset.dropna(inplace = True)

    if pop_down:
        dataset = dataset[dataset['down_votes'] == 0]
        
    if pop_votes:
        dataset.pop('up_votes')
        dataset.pop('down_votes')

    dataset['gender'] = dataset['gender'].apply(
        lambda s: s[0].upper() if isinstance(s, str) else s
    )
    
    dataset = dataset.rename(columns = new_columns)
    
    if sexe:    dataset = filter_col(dataset, 'sex', sexe)
    if age:     dataset = filter_col(dataset, 'age', age)
    if accent:  dataset = filter_col(dataset, 'accents', accent)
    
    return dataset.reset_index()

@audio_dataset_wrapper(
    name    = 'mls', task = ['tts', 'stt'],
    train   = {'directory' : '{}/MLS', 'subset' : 'train'},
    valid   = {'directory' : '{}/MLS', 'subset' : 'test'}
)
def preprocess_mls_annots(directory, lang = 'fr', subset = 'train', ** kwargs):
    directory = os.path.join(directory, lang, subset)
    
    transcripts = os.path.join(directory, 'transcripts.txt')
    segments = os.path.join(directory, 'segments.txt')
    
    with open(segments, 'r', encoding = 'utf-8') as file:
        lines = file.read().split('\n')
    
    durees = {}
    for line in lines:
        if len(line) == 0: continue
        audio_name, url, start, end = line.split('\t')
        durees[audio_name] = float(end) - float(start)
    
    with open(transcripts, 'r', encoding = 'utf-8') as file:
        lines = file.read().split('\n')
    
    infos = []
    for line in lines:
        if len(line) == 0: continue
        audio_name, text = line.split('\t')
        spk_id, book_id, utt_nb = audio_name.split('_')
        
        audio_filename = os.path.join(
            directory, 'audio', spk_id, book_id, audio_name + '.opus'
        )
        
        resampled = {
            dir_name : os.path.join(directory, dir_name, spk_id, book_id, audio_name + '.wav')
            for dir_name in os.listdir(directory) if dir_name.startswith('wavs_')
        }
        
        infos.append({
            'id'    : spk_id,
            'text'  : text,
            'filename'  : audio_filename,
            'book_id'   : book_id,
            'time'      : durees.pop(audio_name, -1),
            ** resampled
        })
        
    dataset = pd.DataFrame(infos)
    
    return _maybe_load_embedding(base_dir, dataset, ** kwargs)

@audio_dataset_wrapper(
    name    = 'LibriSpeech', task = ['tts', 'stt'],
    train   = {'directory' : '{}/LibriSpeech', 'subsets' : ['train-clean-100', 'train-clean-360']},
    valid   = {'directory' : '{}/LibriSpeech', 'subsets' : 'test-clean'}
)
def preprocess_LibriSpeech_annots(directory, subsets = None, ** kwargs):
    def process_speaker(line):
        line = [p for p in line.split() if p != '|']
        
        speaker_id, subset = line[0], line[2]
        if subsets is not None and subset not in subsets: return []
        speaker_infos = {
            'id'    : speaker_id,
            'total_time'    : float(line[3]),
            'name'  : ' '.join(line[4:]),
            'sex'   : line[1]
        }
        
        speaker_dir = os.path.join(directory, subset, speaker_id)
        
        infos = []
        for subdir in os.listdir(speaker_dir):
            trans_filename = os.path.join(speaker_dir, subdir, '{}-{}.trans.txt'.format(speaker_id, subdir))
            
            with open(trans_filename, 'r', encoding = 'utf-8') as file:
                trans_lines = file.read().split('\n')
            
            for l in trans_lines:
                if len(l) == 0: continue
                l = l.split()
                filename = os.path.join(speaker_dir, subdir, l[0]) + '.flac'
                text = ' '.join(l[1:]).capitalize()
                
                trans_infos = {
                    'filename'  : filename,
                    'text'      : text,
                    ** speaker_infos
                }
                
                infos.append(trans_infos)
        
        return infos
    
    if not isinstance(subsets, (list, tuple)): subsets = [subsets]
    subsets = [s for s in subsets if os.path.exists(os.path.join(directory, s))]
    
    infos_filename = os.path.join(directory, 'SPEAKERS.txt')
    
    with open(infos_filename, 'r', encoding = 'utf-8') as file:
        infos = file.read()

    data = []
    for l in infos.split('\n'):
        if len(l) == 0 or l[0] == ';': continue

        data += process_speaker(l)

    dataset = pd.DataFrame(data)
    
    for sub_dir in os.listdir(directory):
        if not sub_dir.startswith('wavs_'): continue
        
        resampled_col = '_'.join(sub_dir.split('_')[:-1])
        subset = sub_dir.split('_')[-1]
        if subset in subsets:
            if resampled_col not in dataset.columns:
                dataset[resampled_col] = dataset['filename']
            dataset[resampled_col] = dataset[resampled_col].apply(
                lambda f: f.replace(subset, sub_dir).replace('.flac', '.wav')
            )
    
    
    return dataset

@audio_dataset_wrapper(name = 'identification', task = ['tts', 'stt'])
def preprocess_identification_annots(directory, by_part = False, ** kwargs):
    if 'parts' not in os.listdir(directory):
        return pd.concat([preprocess_identification_annots(
            os.path.join(directory, sub_dir), by_part = by_part, ** kwargs
        ) for sub_dir in os.listdir(directory)], ignore_index = True)
    
    from datasets.custom_datasets import get_dataset_dir
    
    sub_dir_name = 'parts' if by_part else 'alignments'
    
    directory           = os.path.join(directory, sub_dir_name)
    metadata_filename   = os.path.join(directory, 'map.json')

    data = load_json(metadata_filename)

    dataset = pd.DataFrame(data)
    if not os.path.exists(dataset.loc[0, 'filename']):
        dataset['filename'] = dataset['filename'].apply(
            lambda f: f.replace('D:/datasets', get_dataset_dir())
        )
    if 'indexes' in dataset.columns: dataset.pop('indexes')
    
    return _maybe_load_embedding(directory, dataset, ** kwargs)

def make_siwis_mel(directory, stft_fn, target_rate, langue = 'fr', parts = [1, 2, 3, 5]):
    from utils.audio import load_mel
    
    def map_mel_fn(data):
        mel = load_mel(data, stft_fn)
        return data['filename'], mel
    
    mel_dir_name = stft_fn.dir_name
    
    dataset = preprocess_SIWIS_annots(directory, langue = langue, parts = parts)
    
    dataset[mel_dir_name] = dataset['filename'].apply(lambda f: f.replace('wavs', mel_dir_name)[:-4] + '.npy')
    
    transformed = dataset[mel_dir_name].apply(lambda f: os.path.exists(f))
    
    logger.info("# elements : {}\n  Already processed : {}\n  To transform : {}".format(
        len(dataset), transformed.sum(), len(dataset) - transformed.sum()
    ))
    
    tf_dataset = prepare_dataset(dataset[~transformed][['filename']], batch_size = 0, 
                                 map_fn = map_mel_fn, cache = False)
    
    for p in parts:
        dir_name = os.path.join(directory, langue, mel_dir_name, 'part{}'.format(p))
        os.makedirs(dir_name, exist_ok = True)
    
    for filename, mel in tqdm(tf_dataset):
        filename = filename.numpy().decode('utf-8')
        mel_filename = filename.replace('wavs', mel_dir_name)[:-4] + '.npy'
        
        np.save(mel_filename, mel)
    
    return dataset

def make_CV_mel(directory, stft_fn, target_rate, file = 'validated.tsv'):
    from utils.audio import load_mel

    def map_mel_fn(data):
        mel = load_mel(data, stft_fn)
        return data['path'], mel
    
    mel_dir_name = stft_fn.dir_name
    os.makedirs(os.path.join(directory, mel_dir_name), exist_ok = True)
    
    dataset = pd.read_csv(os.path.join(directory, file), sep = '\t')
    dataset['path'] = dataset['path'].apply(lambda f: os.path.join(directory, 'clips', f))
    dataset[mel_dir_name] = dataset['path'].apply(lambda f: f.replace('clips', mel_dir_name)[:-4] + '.npy')
    
    transformed = dataset[mel_dir_name].apply(lambda f: os.path.exists(f))
    
    logger.info("# elements : {}\n  Already processed : {}\n  To transform : {}".format(
        len(dataset), transformed.sum(), len(dataset) - transformed.sum()
    ))
    
    tf_dataset = prepare_dataset(dataset[~transformed][['path']], batch_size = 0, 
                                 map_fn = map_mel_fn, cache = False)
        
    for filename, mel in tqdm(tf_dataset):
        filename = filename.numpy().decode('utf-8')
        
        mel_filename = filename.replace('clips', mel_dir_name)[:-4] + '.npy'
        
        np.save(mel_filename, mel)
    
    dataset.to_csv(os.path.join(directory, file), sep = '\t', index = False)
    
    return dataset

