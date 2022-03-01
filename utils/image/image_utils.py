
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

import cv2
import numpy as np

from matplotlib import colors

BASE_COLORS = list(colors.BASE_COLORS.keys())

def rgb2gray(rgb):
    return np.dot(rgb[...:3], [0.2989, 0.5870, 0.1140])

def _normalize_color(color, image = None):
    if colors.is_color_like(color):
        color = colors.to_rgb(color)
    
    color = np.array(color)
    if np.max(color) > 1.: color = color / 255.
    if image is not None and np.max(image) > 1.: color = (color * 255).astype(image.dtype)
    
    return color
