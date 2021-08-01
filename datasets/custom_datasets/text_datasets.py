import os
import pandas as pd

from utils import load_json

_siamese_renaming = {'sentence1' : 'text_x', 'sentence2' : 'text_y'}

def preprocess_europarl_annots(directory, base_name, input_lang, output_lang):
    input_filename = os.path.join(directory, '{}.{}'.format(base_name, input_lang))
    output_filename = os.path.join(directory, '{}.{}'.format(base_name, output_lang))
    
    with open(input_filename, 'r', encoding = 'utf-8') as input_file:
        inputs = input_file.read().split('\n')
        
    with open(output_filename, 'r', encoding = 'utf-8') as output_file:
        outputs = output_file.read().split('\n')

    datas = [[inp, out] for inp, out in zip(inputs, outputs)]
    return pd.DataFrame(data = datas, columns = [input_lang, output_lang])
    
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

def preprocess_SQUAD_annots(directory, subset, version = '2.0', skip_impossible = False, ** kwargs):
    filename = os.path.join(directory, '{}-v{}.json'.format(subset, version))
    
    metadata = load_json(filename)['data']
    
    dataset = []
    for data in metadata:
        for para in data['paragraphs']:
            for qa in para['qas']:
                dataset.append({
                    'title' : data['title'],
                    'context' : para['context'],
                    ** qa
                })
    
    dataset = pd.DataFrame(dataset)
    
    if skip_impossible:
        dataset = dataset[~dataset['is_impossible']]
        dataset.pop('plausible_answers')
    
    return dataset

_custom_text_datasets = {
    'europarl'  : {
        'directory' : '{}/Europarl',
        'base_name' : 'europarl-v7.fr-en',
        'input_lang'    : 'en',
        'output_lang'   : 'fr'
    },
    'snli'  : {
        'train' : {'directory' : '{}/snli_1.0', 'subset' : 'train'},
        'valid' : {'directory' : '{}/snli_1.0', 'subset' : 'dev'}
    },
    'sts'   : {
        'train' : {'directory' : '{}/sts_benchmark', 'subset' : 'train'},
        'valid' : {'directory' : '{}/sts_benchmark', 'subset' : 'dev'}
    },
    'squad' : {
        'train' : {'directory' : '{}/SQUAD2.0', 'subset' : 'train'},
        'valid' : {'directory' : '{}/SQUAD2.0', 'subset' : 'dev'}
    }
}

_text_dataset_processing  = {
    'europarl'      : preprocess_europarl_annots,
    'snli'          : preprocess_snli_annots,
    'sts'           : process_sts_annots,
    'squad'         : preprocess_SQUAD_annots
}