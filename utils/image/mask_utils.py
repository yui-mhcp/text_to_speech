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
import numpy as np

from utils.keras_utils import ops
from utils.image.image_io import get_image_size
from utils.image.image_utils import normalize_color
from utils.wrapper_utils import dispatch_wrapper

_mask_transformations = {
    'keep'      : lambda image, mask, ** kwargs: image * mask,
    'remove'    : lambda image, mask, ** kwargs: image * (1. - mask)
}

def create_poly_mask(image, points, color = 1., dtype = np.float32):
    """
        Creates a mask based on a polygon via the `cv2.fillPoly` method
        
        Arguments :
            - image     : the original image (to get the mask's shape)
            - points    : a list of polygons (`np.ndarray` of shape `[n_points, 2]`)
            - color     : the color for the mask
    """
    shape   = get_image_size(image)
    
    color   = normalize_color(color, image = image)
    points  = [np.reshape(np.array(pts), [-1, 2]).astype(np.int32) for pts in points]
    
    mask    = np.zeros(shape, dtype = dtype)
    for pt in points: mask = cv2.fillPoly(mask, pts = [pt], color = color)
    return mask

def create_color_mask(image, color, threshold = 0.1, mask = None, per_channel = True):
    """
        Creates a mask where `image == color (+/- threshold)`
        
        Arguments :
            - image     : the raw image (3D `Tensor`)
            - color     : a color (either 3-value tuple / single value), the color to mask
            - threshold : the accepted variation around `color`
            - mask      : a mask to apply on the produced color mask
            - per_channel   : boolean, whether to apply the `threshold`'s tolerance per channel or not
                - if `True`, each channel must be in the range [c - threshold, c + threshold]
                - if `False`, the sum of difference must be in the range
        Return :
            if `mask` is None:
            - `Tensor` of boolean with same shape as `image` (with 3rd channel == 1)
            else:
            - `Tensor` of same shape and dtype as `mask`, the mask's values where `color_mask` is True
    """
    is_float    = ops.is_float(image)
    color   = normalize_color(color, dtype = 'float32' if is_float else 'uint8')
    if ops.is_tensor(image): color = ops.convert_to_tensor(color, image.dtype)
    
    if is_float:
        if not ops.is_float(threshold): threshold = threshold / 255.
    elif ops.is_float(threshold):       threshold = int(threshold * 255)

    if per_channel:
        color_mask  = ops.all(ops.abs(image - color) <= threshold, axis = -1, keepdims = True)
    else:
        color_mask  = ops.sum(ops.abs(image - color), axis = -1, keepdims = True) <= threshold
    
    if mask is not None:
        zero = 0
        if ops.is_tensor(image):
            mask    = ops.convert_to_tensor(mask)
            zero    = ops.convert_to_tensor(0, dtype = mask.dtype)
        color_mask  = ops.where(color_mask, mask, zero)
    
    return color_mask

def smooth_mask(mask, smooth_size = 0.5, divide_factor = 2., ** kwargs):
    """ Smooth a mask to not have a 0-1 mask but smooth boundaries """
    if ops.is_float(smooth_size): smooth_size = int(smooth_size * len(mask))
    
    filters     = ops.ones([smooth_size, smooth_size, 1, 1]) / (smooth_size ** 2 / divide_factor)
    expanded    = ops.expand_dims(ops.cast(mask, 'float32'), axis = 0)
    
    smoothed = ops.conv2d(expanded, filters, 1, padding = 'same')
    return ops.clip(ops.squeeze(smoothed, axis = 0), 0., 1.)

@dispatch_wrapper(_mask_transformations, 'method', default = 'keep')
def apply_mask(image, mask, method, on_background = False, smooth = False, ** kwargs):
    """
        Applies `mask` on `image` with the given `method`
        
        Note that some transformations will keep the image where `mask == 1` (such as `keep`), while others will modify the masked regions (such as blur / replace)
        E.g., `keep + on_background` will keep the background while `blur + on_background` will blur the background

        Arguments :
            - image     : the image to mask
            - mask      : the mask (either boolean / scalar values)
            - method    : the transformation method name or a callable, the transformation to apply
            - on_background : whether to apply transformation on the mask or on itsbackground
            - smooth    : whether to smooth the mask or not
        Return :
            - the masked image with same shape as `image`
    """
    image   = ops.convert_to_tensor(image)
    mask    = ops.convert_to_tensor(mask)
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError("The image and the mask have different shapes !\n  Image shape : {}\n  Mask shape : {}".format(image.shape, mask.shape))
    
    if not callable(method) and method not in _mask_transformations:
        raise ValueError("Unknown mask transformation !\n  Accepted : {}\n  Got : {}".format(
            tuple(_mask_transformations.keys()), method
        ))
    
    if not ops.is_float(mask):
        mask = ops.cast(mask, 'float32')
    if smooth:
        mask = smooth_mask(mask, ** kwargs)
    
    if on_background: mask = 1. - mask
    
    if callable(method):
        return method(image, mask, ** kwargs)
    return _mask_transformations[method](image, mask, ** kwargs)

@apply_mask.dispatch('blur')
def blur_mask(image, mask, filter_size = (21, 21), ** kwargs):
    """ Blurs `image` where `mask == 1` """
    blurred = cv2.blur(ops.convert_to_numpy(image), filter_size)
    
    return replace_mask(image, mask, blurred)

@apply_mask.dispatch('replace')
def replace_mask(image, mask, background, ** kwargs):
    """ Replaces `image` by `background` where `mask == 1` """
    if not hasattr(background, 'shape') or len(background.shape) == 1:
        color = normalize_color(background, image = image)
        
        background          = np.zeros_like(image)
        background[..., :]  = color

    if ops.is_tensor(image): background = ops.convert_to_tensor(background, image.dtype)
    return image * (1. - mask) + mask * background

