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

import numpy as np
import pandas as pd

from functools import wraps
from multiprocessing import cpu_count

from loggers import timer
from .. import add_dataset, maybe_load_embedding

_synchronization_cols   = ('label', 'boxes', 'box_infos', 'segmentation')

def image_dataset_wrapper(name, task, ** default_config):
    def wrapper(dataset_loader):
        """
            Wraps a function that loads the image dataset, and apply some common post-processing
            
            The function must return a dict {filename : infos} where `infos` can contain :
                - (optional) label  : the label(s) in the image
                - (optional) boxes  : the bounding box(es) in the image (in mode [x, y, w, h])
                - (optional) box_infos  : the box(es)' information
                - (optional) segmentation   : the object's segmentation(s)
                
                - (optional) text   : the image caption(s)
        """
        @timer(name = '{} loading'.format(name if isinstance(name, str) else name[0]))
        @wraps(dataset_loader)
        def _load_and_process(directory,
                              * args,
                              
                              add_image_size = False,
                              
                              keep_empty    = True,
                              accepted_labels   = None,
                              labels_subtitutes = None,
                              
                              pad_boxes = False,
                              combine_box_lines = False,
                              min_box_per_image = -1,
                              max_box_per_image = -1,
                              one_line_per_box  = False,
                              
                              one_line_per_caption  = False,
                              
                              return_raw    = False,
                              
                              ** kwargs
                             ):
            assert not (one_line_per_box and one_line_per_caption)
            
            dataset = dataset_loader(directory, * args, ** kwargs)
            if return_raw: return dataset
            
            dataset = [
                {'filename' : file, ** row} for file, row in dataset.items()
            ]
            
            if (add_image_size or combine_box_lines) and 'width' not in dataset[0]:
                dataset = _add_image_size(dataset, tqdm = kwargs.get('tqdm', lambda x: x))
            
            if 'label' in dataset[0]:
                dataset = _replace_labels(dataset, labels_subtitutes)
                dataset = _filter_labels(dataset, accepted_labels, keep_empty)

                if any('segmentation' in row for row in dataset):
                    for row in dataset:
                        if row.setdefault('segmentation', []):
                            max_len     = max([len(poly) for poly in row[key]])
                            row[key]    = np.array(
                                [poly + [-1] * (max_len - len(poly)) for poly in row[key]],
                                dtype = np.int32
                            )

                if any('boxes' in row for row in dataset):
                    for row in dataset: row.setdefault('boxes', [])
                    
                    if combine_box_lines:
                        dataset = _combine_text_lines(dataset, ** kwargs)
                    
                    if min_box_per_image > 0 or max_box_per_image > 0:
                        if max_box_per_image <= 0: max_box_per_image = float('inf')
                        dataset = [
                            row for row in dataset if (
                                min_box_per_image <= len(row['boxes']) <= max_box_per_image
                            )
                        ]
                
                    if one_line_per_box:
                        dataset = _flatten_dataset(
                            dataset, keys = _synchronization_cols
                        )
                    
                    for row in dataset:
                        row.update({
                            'boxes'     : np.array(row['boxes'], dtype = np.int32),
                            'nb_box'    : len(row['boxes'])
                        })
                    
                    if pad_boxes and not one_line_per_box: dataset = _pad_boxes(dataset)
            
            if one_line_per_caption and 'text' in dataset[0]:
                dataset = _flatten_dataset(dataset, keys = 'text')
            
            dataset = pd.DataFrame(dataset)
            
            dataset = maybe_load_embedding(directory, dataset, ** kwargs)
            
            return dataset
        
        add_dataset(name, processing_fn = _load_and_process, task = task, ** default_config)
        
        return _load_and_process
    return wrapper

def _add_image_size(dataset, tqdm = None):
    from utils.image import get_image_size
    
    for row in tqdm(dataset):
        image_h, image_w = get_image_size(row['filename'])
        row.update({'height' : image_h, 'width' : image_w})
    
    return dataset

def _replace_labels(dataset, labels_subtitutes):
    if not labels_subtitutes: return dataset
    
    for row in dataset:
        if 'label' not in row: continue
        if not isinstance(row['label'], list):
            row['label'] = labels_subtitutes.get(row['label'], row['label'])
        else:
            row['label'] =  [
                labels_subtitutes.get(l, l) for l in row['label']
            ]
    return dataset

def _filter_labels(dataset, accepted_labels, keep_empty, synchronize = _synchronization_cols):
    if not accepted_labels:
        return dataset if keep_empty else [row for row in dataset if row.get('label', None)]
    
    if not isinstance(accepted_labels, (list, tuple)): accepted_labels = [accepted_labels]
    
    for row in dataset:
        if 'label' not in row: continue
        if not isinstance(row['label'], list):
            if row['label'] not in accepted_labels: row['label'] = None
        else:
            # also removes the box / segmentation associated to the removed label(s)
            for i in reversed(range(len(row['label']))):
                if row['label'][i] not in accepted_labels:
                    for k in synchronize:
                        if k not in row: continue
                        row[k].pop(i)
        
    return dataset if keep_empty else [row for row in dataset if row.get('label', None)]

def _flatten_dataset(dataset, keys):
    if not isinstance(keys, (list, tuple)): keys = [keys]
    if any(not isinstance(dataset[0].get(k, []), (list, tuple, np.ndarray)) for k in keys):
        return dataset
    
    keys    = [k for k in keys if k in dataset[0]]
    flat    = []
    for row in dataset:
        if all(k not in row for k in keys):
            flat.append(row)
            continue
        
        for i in range(len(row[keys[0]])):
            new_row = row.copy()
            for k in keys: new_row[k] = row[k][i]
            flat.append(new_row)
    
    return flat

def _pad_boxes(dataset):
    n = max(row['nb_box'] for row in dataset)
    for row in dataset:
        if row['nb_box'] != n:
            row['boxes'] = np.pad(
                row['boxes'], [(0, n - row['nb_box']), (0, 0)]
            )
            if 'label' in row: row['label'].extend([''] * (n - row['nb_box']))
    return dataset

def _combine_text_lines(dataset, keep_original = True, tqdm = lambda x: x, ** kwargs):
    from utils.image.bounding_box import NORMALIZE_WH, BoxFormat, convert_box_format, combine_boxes_horizontal
    
    new_data = [] if not keep_original else dataset
    for row in tqdm(dataset):
        if len(row['label']) == 1:
            if not keep_original: new_data.append(row)
            continue

        image_h, image_w = row['height'], row['width']
        boxes, indices, _ = combine_boxes_horizontal(
            row['boxes'], image_h = image_h, image_w = image_w, ** kwargs
        )

        comb_texts = [
            ' '.join([row['label'][idx] for idx in comb_index])
            for comb_index in indices
        ]
        
        if keep_original:
            boxes       = boxes[[i for i, idx in enumerate(indices) if len(idx) > 1]]
            comb_texts  = [comb_texts[i] for i, idx in enumerate(indices) if len(idx) > 1]
        
        if len(boxes) == 0: continue
        
        boxes   = convert_box_format(
            boxes,
            target  = BoxFormat.XYWH,
            source  = BoxFormat.CORNERS,
            image_h = image_h,
            image_w = image_w,
            normalize_mode = NORMALIZE_WH
        )
        new_data.append({** row, 'boxes' : boxes, 'label' : comb_texts})
    
    return new_data

