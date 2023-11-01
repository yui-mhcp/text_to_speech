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

import numpy as np
import tensorflow as tf

class BoundingBox:
    def __init__(self, x1, y1, x2 = None, y2 = None, w = None, h = None, 
                 classes = None, conf = 1., score = None, labels = None, angle = 0., ** kwargs):
        assert x2 is not None or w is not None
        assert y2 is not None or h is not None
        
        self.x1 = max(x1, 0.)
        self.y1 = max(y1, 0.)
        
        self.x2 = x2 if x2 is not None else x1 + w
        self.y2 = y2 if y2 is not None else y1 + h
        
        self.w  = max(self.x2 - self.x1, 0)
        self.h  = max(self.y2 - self.y1, 0)
        
        self.angle  = angle
        self.conf   = conf if score is None else score
        self.classes = classes
        
        self.c = np.argmax(classes) if isinstance(classes, (np.ndarray, tf.Tensor)) else classes
        if self.c is None: self.c = 0
        
        self.label  = labels[self.c] if labels is not None else self.c
    
    @property
    def p(self):
        if self.classes is None: return 1.
        return self.classes[self.c] if not isinstance(self.classes, (int, np.integer)) else 1.
    
    @property
    def area(self):
        return self.w * self.h
    
    @property
    def score(self):
        return self.p * self.conf
    
    @property
    def rectangle(self):
        return [self.x1, self.y1, self.x2, self.y2]
    
    @property
    def box(self):
        return [self.x1, self.y1, self.w, self.h, self.c]
    
    def __str__(self):
        return "{:.4f} {:.4f} {:.4f} {:.4f} {} {:.4f}".format(*self.box, self.score)
    
    def __repr__(self):
        return str(tuple(self.box))
    
    def get_config(self):
        return self.json()
    
    def json(self, labels = None):
        from utils.generic_utils import to_json
        
        infos = {
            'xmin' : self.x1,
            'ymin' : self.y1,
            'xmax' : self.x2,
            'ymax' : self.y2,
            'label' : self.label
        }
        label = labels[self.c] if labels is not None else self.label
        if label is not None: infos['label'] = label
        return to_json(infos)
    
    def to_image(self, image_w, image_h):
        return [self.x1 * image_w, self.y1 * image_h, self.x2 * image_w, self.y2 * image_h]

