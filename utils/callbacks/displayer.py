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

import logging
import numpy as np

from .callback import Callback
from loggers import time_logger
from utils.plot_utils import plot

logger = logging.getLogger(__name__)

class Displayer(Callback):
    def __init__(self, data_key, max_display = -1, name = 'display', ** kwargs):
        super().__init__(name = name, ** kwargs)
        
        self.kwargs = kwargs
        self.data_key   = data_key if isinstance(data_key, (list, tuple)) else [data_key]
        self.max_display    = max_display if max_display is not True else -1

        self.index = 0
    
    def apply(self, infos, output):
        if self.max_display > 0 and self.index >= self.max_display: return
        
        self.index += 1
        with time_logger.timer(self.name):
            for k in self.data_key:
                if k in output:
                    return self.display(output[k], ** output)
                elif k in infos:
                    return self.display(infos[k], ** infos)

    def display(self, _data, ** kwargs):
        raise NotImplementedError()
    

class AudioDisplayer(Displayer):
    def __init__(self, data_key = ('audio', 'filename'), *, name = 'display audio', ** kwargs):
        super().__init__(data_key, name = name, ** kwargs)

    def display(self, _data, *, rate = None, ** kwargs):
        from utils.audio import display_audio
        
        display_audio(_data, rate = rate)


class ImageDisplayer(Displayer):
    def __init__(self, data_key = ('image', 'filename'), *, name = 'display image', ** kwargs):
        super().__init__(data_key, name = name, ** kwargs)

    def display(self, _data, ** kwargs):
        if isinstance(_data, str):
            from utils.image import load_image
            _data = load_image(_data, to_tensor = False, run_eagerly = True)
        
        plot(_data, plot_type = 'imshow')

class SpectrogramDisplayer(ImageDisplayer):
    def __init__(self, data_key = 'mel', *, name = 'display mel', ** kwargs):
        super().__init__(data_key, name = name, ** kwargs)

class BoxesDisplayer(ImageDisplayer):
    def __init__(self, * args, print_boxes = False, name = 'display boxes', ** kwargs):
        super().__init__(* args, name = name, ** kwargs)
        self.print_boxes    = print_boxes
    
    def display(self, _data, *, boxes, ** kwargs):
        from utils.image import load_image, show_boxes, sort_boxes
        
        if isinstance(_data, str): _data = load_image(_data, to_tensor = False, run_eagerly = True)
        
        boxes = sort_boxes(boxes, method = 'top', image = _data)
        _boxes = boxes if not isinstance(boxes, dict) else boxes['boxes']
        if self.print_boxes:
            logger.info("{} boxes found :\n{}".format(
                len(_boxes), '\n'.join(str(b) for b in _boxes)
            ))

        show_boxes(_data, boxes, ** self.kwargs)

class OCRDisplayer(Displayer):
    def __init__(self, data_key = ('image', 'filename'), *, name = 'display ocr', ** kwargs):
        super().__init__(data_key, name = name, ** kwargs)

    def display(self, _data, *, ocr, ** kwargs):
        from utils.image import load_image, draw_boxes
        
        if isinstance(_data, str): _data = load_image(_data, to_tensor = False, run_eagerly = True)
        elif isinstance(_data, np.ndarray): _data = _data.copy()
        
        boxes = np.concatenate([res['rows'] for res in ocr], axis = 0)
        plot(draw_boxes(_data, boxes, source = 'xyxy', show_text = False))
        for box_infos in ocr:
            logger.info('Text (score {}) : {}'.format(
                np.around(box_infos['scores'], decimals = 3), box_infos['text']
            ))
            plot(load_image(
                _data, boxes = box_infos['box'], source = box_infos['source']
            ))
