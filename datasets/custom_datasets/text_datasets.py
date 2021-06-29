import os
import pandas as pd

def preprocess_europarl_annots(directory, base_name, input_lang, output_lang):
    input_filename = os.path.join(directory, '{}.{}'.format(base_name, input_lang))
    output_filename = os.path.join(directory, '{}.{}'.format(base_name, output_lang))
    
    with open(input_filename, 'r', encoding = 'utf-8') as input_file:
        inputs = input_file.read().split('\n')
        
    with open(output_filename, 'r', encoding = 'utf-8') as output_file:
        outputs = output_file.read().split('\n')

    datas = [[inp, out] for inp, out in zip(inputs, outputs)]
    return pd.DataFrame(data = datas, columns = [input_lang, output_lang])
    
_custom_text_datasets = {
    'europarl'  : {
        'directory' : '{}/Europarl',
        'base_name' : 'europarl-v7.fr-en',
        'input_lang'    : 'en',
        'output_lang'   : 'fr'
    }
}

_text_dataset_processing  = {
    'europarl'      : preprocess_europarl_annots,
}