import cv2
import numpy as np
import tensorflow as tf

from utils.image.image_utils import _normalize_color

def create_color_mask(image, color, threshold = 0.1, mask = None):
    """
        Create a mask where `image == color (+/- threshold)`
        
        Arguments :
            - image     : the raw image
            - color     : a color (either 3-value tuple / single value), the color to mask
            - threshold : the accepted variation around `color`
            - mask      : a mask to apply on the produced color mask
        Return :
            - array of boolean with same shape as `image` (with 3rd channel == 1)
        
        Note that the threshold is the accepted variation around `color` for the 3 channels (not independentely). 
        i.e. a variation of 75 allows a mean variation of 25 per channels (and not 75 per channel).
    """
    color = _normalize_color(color, image = image)
    
    if threshold < 1. and np.max(color) > 1.: threshold = threshold * 255
    if threshold > 1. and np.max(color) <= 1.: threshold = threshold / 255.
    
    color_mask = np.abs(image - color) < threshold
    color_mask = np.sum(color_mask, axis = -1, keepdims = True) == 3
    
    if mask is not None: color_mask = color_mask * mask
    
    return color_mask

def smooth_mask(mask, smooth_size = 0.5, divide_factor = 2., ** kwargs):
    """ Smooth a mask to have not have a 0-1 mask but smooth boundaries """
    if isinstance(smooth_size, float): smooth_size = int(smooth_size * len(mask))
    
    filters = tf.ones([smooth_size, smooth_size, 1, 1]) / (smooth_size * smooth_size / divide_factor)
    expanded = tf.expand_dims(tf.cast(mask, tf.float32), axis = 0)
    
    smoothed = tf.nn.conv2d(expanded, filters, 1, 'SAME')
    return tf.clip_by_value(tf.squeeze(smoothed, axis = 0), 0., 1.)

def apply_mask(image, mask, on_background = False,
               smooth = False, transform = 'keep', **kwargs):
    """
        Apply a mask on the `image` with the given transformation
        
        Arguments :
            - image     : the image to mask
            - mask      : the mask (either boolean / scalar values)
            - on_background : whether to apply transformation on the mask or on itsbackground
            - smooth    : whether to smooth the mask or not
            - transform : the transformation's name or a callable, the transformation to apply
        Return :
            - the masked image with same shape as `image`
    """
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError("Le mask n'a pas la bonne dimension !\n  Image shape : {}\n  Mask shape : {}".format(image.shape, mask.shape))
    
    if transform not in _transformations and not callable(transform):
        raise ValueError("Mask transformation inconnue !\n  Reçu : {}\n  Acceptés : {}".format(transform, list(_transformations.keys())))
    
    if not isinstance(mask, tf.Tensor) or mask.dtype != tf.float32:
        mask = tf.cast(mask, tf.float32)
    
    if smooth: mask = smooth_mask(mask, ** kwargs)
    
    if on_background: mask = 1. - mask
    
    if callable(transform):
        return transform(image, mask, ** kwargs)
    return _transformations[transform](image, mask, **kwargs)

def blur_mask(image, mask, filter_size = (21, 21), ** kwargs):
    if hasattr(image, 'numpy'): image = image.numpy()
    blurred = cv2.blur(image, filter_size)
    
    return replace_mask(image, mask, blurred)

def replace_mask(image, mask, background, ** kwargs):
    if not isinstance(background, np.ndarray) or background.shape != image.shape:
        background_image = np.zeros_like(image)
        
        color = _normalize_color(background, image = image)
            
        background_image[...,:] = color
    else:
        background_image = background
    
    return image * (1. - mask) + mask * background_image

    
_transformations    = {
    'keep'      : lambda image, mask, ** kwargs: image * mask,
    'remove'    : lambda image, mask, ** kwargs: image * ~mask,
    'blur'      : blur_mask,
    'replace'   : replace_mask
}