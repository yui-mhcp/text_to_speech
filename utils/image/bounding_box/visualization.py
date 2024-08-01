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

import cv2
import enum

from utils.keras_utils import ops
from utils.generic_utils import get_enum_item
from utils.plot_utils import plot, plot_multiple
from ..image_utils import normalize_color
from ..image_io import load_image
from .converter import BoxFormat, NORMALIZE_WH, box_converter_wrapper

class Shape(enum.IntEnum):
    CERCLE  = 0
    CIRCLE  = 0
    OVALE   = 1
    ELLIPSE = 1
    RECT    = 2
    RECTANGLE   = 2

@box_converter_wrapper(
    BoxFormat.XYXY, normalize = NORMALIZE_WH, force_np = True, force_dict = True
)
def draw_boxes(image,
               boxes,
               
               show_text    = True,
               
               shape    = Shape.RECTANGLE,
               color    = 'r',
               thickness    = 3,
               with_label   = True,
               labels   = None,
               
               vertical = True,
               ** kwargs
              ):
    if not isinstance(color, list): color = [color]
    if isinstance(shape, str):      shape = get_enum_item(shape, Shape)
    if isinstance(image, str):      image = load_image(image)
    image   = ops.convert_to_numpy(image)
    image_h, image_w = image.shape[:2]

    color = [
        ops.convert_to_numpy(normalize_color(c, dtype = ops.dtype_to_str(image.dtype))).tolist()
        for c in color
    ]
    label_color = {}
    for i, (x1, y1, x2, y2) in enumerate(boxes['boxes'].tolist()):
        if x2 <= x1 or y2 <= y1: continue
        
        c = color[i % len(color)]
        if with_label and boxes.get('labels', labels) is not None:
            label   = boxes['labels'][i] if 'labels' in boxes else (
                labels[i] if len(labels) > 1 else labels[0]
            )
            conf    = boxes['scores'][i] if 'scores' in boxes else None
            if labels and ops.is_int(label): label = labels[label]
            if label not in label_color: 
                label_color[label] = color[len(label_color) % len(color)]
            c = label_color[label]
            
            if show_text:
                text    = '{}{}'.format(label, '' if not conf else ' ({:.2f} %)'.format(conf))
                font_scale   = 1e-3 * image_h
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )
                # Draw the rectangle behind the text
                image = cv2.rectangle(
                    image, (x1, y1 - text_height - baseline - 13), (x1 + text_width, y1), c, -1
                )
                factor = 1. if isinstance(c[0], float) else 255
                if sum(c) / 3. < 0.4 * factor:
                    c_text = (255, 255, 255) if isinstance(c[0], int) else (1., 1., 1.)
                else:
                    c_text = (0, 0, 0)

                image   = cv2.putText(
                    image, text, (x1, y1 - 13), cv2.FONT_HERSHEY_SIMPLEX, font_scale, c_text, 2
                )
        
        if shape == Shape.RECTANGLE:
            image = cv2.rectangle(image, (x1, y1), (x2, y2), c, thickness)
        elif shape == Shape.CIRCLE:
            w, h = x2 - x1, y2 - y1
            image = cv2.circle(
                image, ((x1 + x2) // 2, (y1 + y2) // 2), min(w, h) // 2, c, thickness
            )
        elif shape == Shape.ELLIPSE:
            w, h    = x2 - x1, y2 - y1
            axes    = (w // 2, int(h / 1.5)) if vertical else (int(w / 1.5), h // 2)
            image   = cv2.ellipse(
                image,
                angle       = 0,
                startAngle  = 0,
                endAngle    = 360, 
                center      = ((x1 + x2) // 2, (y1 + y2) // 2),
                thickness   = thickness,
                axes    = axes,
                color   = c
            )
    
    return image

def show_boxes(image, boxes, source = BoxFormat.DEFAULT, dezoom_factor = 1., labels = None, ** kwargs):
    """
        Displays a (list of) `boxes` with `utils.plot_multiple`
        
        Arguments :
            - filename  : the image (raw or filename)
            - boxes     : the boxes coordinates
            - labels    : the labels for each box
            - dezoom_factor / box_mode  : forwarded to `convert_box_format`
            - kwargs    : forwarded to `plot_multiple`
    """
    from .processing import crop_box

    if isinstance(image, str): image = load_image(image, to_tensor = False, run_eagerly = True)
    image = ops.convert_to_numpy(image)
    
    _, images = crop_box(
        image, boxes, source = source, dezoom_factor = dezoom_factor
    )
    if images is None: return
    elif not isinstance(images, list): images = [images]
    
    if not isinstance(boxes, dict): boxes = {'boxes' : boxes}
    
    plot_data, counts = {}, {}
    for i, box_image in enumerate(images):
        if any(s == 0 for s in box_image.shape): continue
        
        label = 'Box'
        if boxes.get('labels', labels) is not None:
            label   = boxes['labels'][i] if 'labels' in boxes else (
                labels[i] if len(labels) > 1 else labels[0]
            )
            if labels and ops.is_int(label): label = labels[label]
        if label not in counts: counts[label] = 0
        counts[label] += 1
        
        title = '{} #{}'.format(label, counts[label])
        if 'scores' in boxes:
            title += ' ({:.2f} %)'.format(boxes['scores'][i])
        
        plot_data[title] = {'x' : box_image}
    
    plot_multiple(** plot_data, plot_type = 'imshow', ** kwargs)

