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

import logging
import numpy as np

from abc import abstractmethod
from threading import Lock

from loggers import Timer
from .callback import Callback
from ..plot_utils import plot

logger = logging.getLogger(__name__)

class Displayer(Callback):
    def __init__(self, data_key, max_display = -1, name = None, ** kwargs):
        if name is None: name = 'display {}'.format(self.__class__.__name__[:-9].lower())

        super().__init__(name = name, ** kwargs)

        self.kwargs = kwargs
        self.data_key   = data_key if isinstance(data_key, (list, tuple)) else [data_key]
        self.max_display    = max_display if max_display is not True else -1

        self.index = 0
        # `infer` (hence its callbacks) may run in multiple `Stream` workers : only the
        # counter is serialized ; the display backends themselves (matplotlib / IPython)
        # are not fully thread-safe, so displayers should be used with care when `max_workers > 1`
        self.mutex = Lock()

    def apply(self, infos, output, ** _):
        with self.mutex:
            if self.max_display > 0 and self.index >= self.max_display: return
            self.index += 1

        with Timer(self.name):
            for k in self.data_key:
                if k in output:
                    return self.display(output[k], ** output)
                elif k in infos:
                    return self.display(infos[k], ** infos)

    @abstractmethod
    def display(self, _data, ** kwargs):
        """ Display `_data` """
    

class AudioDisplayer(Displayer):
    def __init__(self,
                 data_key = ('audio', 'filename'),
                 *,

                 play   = False,
                 display    = True,
                 show_text  = True,
                 
                 ** kwargs
                ):
        assert play or display
        
        super().__init__(data_key, ** kwargs)

        self.play = play
        self._display   = display
        self.show_text  = show_text

    def display(self, _data, *, rate = None, ** kwargs):
        from utils.audio import display_audio, play_audio
        
        if self._display:
            if self.show_text and 'text' in kwargs:
                logger.info('Text : {}'.format(kwargs['text']))
            display_audio(_data, rate = rate, play = self.play == 1)
        
        if not self._display or self.play > 1:
            play_audio(_data, rate = rate)

class ImageDisplayer(Displayer):
    def __init__(self, data_key = ('image', 'filename'), ** kwargs):
        super().__init__(data_key, ** kwargs)

    def display(self, _data, ** kwargs):
        if isinstance(_data, str):
            if _data.endswith('.npy'):
                _data = np.load(_data)
            else:
                from utils.image import load_image
                _data = load_image(_data, to_tensor = False)
        
        plot(_data, plot_type = 'imshow')

class SpectrogramDisplayer(ImageDisplayer):
    def __init__(self, data_key = 'mel', ** kwargs):
        super().__init__(data_key, ** kwargs)

class BoxesDisplayer(Displayer):
    def __init__(self, data_key = ('image', 'filename'), print_boxes = False, ** kwargs):

        super().__init__(data_key, ** kwargs)
        
        self.print_boxes = print_boxes
    
    def display(self, _data, *, boxes, ** kwargs):
        from utils.image import load_image, show_boxes, sort_boxes
        
        if isinstance(_data, str): _data = load_image(_data, to_tensor = False)
        
        boxes = sort_boxes(boxes, method = 'top', image = _data)
        _boxes = boxes if not isinstance(boxes, dict) else boxes['boxes']
        if self.print_boxes:
            logger.info("{} boxes found :\n{}".format(
                len(_boxes), '\n'.join(str(b) for b in _boxes)
            ))

        show_boxes(_data, boxes, ** self.kwargs)

class OCRDisplayer(ImageDisplayer):
    def display(self, _data, *, ocr, ** kwargs):
        from utils.image import load_image, draw_boxes
        
        if isinstance(_data, str): _data = load_image(_data, to_tensor = False)
        elif isinstance(_data, np.ndarray): _data = _data.copy()
        
        boxes = np.concatenate([res['rows'] for res in ocr], axis = 0)
        plot(draw_boxes(_data, boxes, source = 'xyxy', show_text = False, labels = ['text']))
        for box_infos in ocr:
            logger.info('Text (score {}) : {}'.format(
                np.around(box_infos['scores'], decimals = 3), box_infos['text']
            ))
            plot(load_image(_data, boxes = box_infos['boxes'], source = box_infos['format']))