
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

import os
import cv2
import glob
import math
import time
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image
from tensorflow.errors import InvalidArgumentError

from utils.generic_utils import time_to_string

def display_image(image):
    from IPython.display import Image, display
    display(Image(image))
    
def get_image_size(image):
    """
        Return image size [height, width] supporting different formats 
        Arguments :
            - image : str (filename) or 2, 3 or 4-D np.ndarray / Tensor (image)
        Return :
            - [height, width]   : image's size
    """
    if isinstance(image, (np.ndarray, tf.Tensor)):
        if len(image.shape) in (2, 3):
            return image.shape[0], image.shape[1]
        elif len(image.shape) == 4:
            return image.shape[1], image.shape[2]
        else:
            raise ValueError("Unknown image shape : {}\n".format(image.shape, image))
    elif isinstance(image, str):
        image = Image.open(image)
        return image.size
    else:
        raise ValueError("Unknown image type : {}\n{}".format(type(image), image))
    
def load_image(filename, target_shape = None, mode = None, channels = 3, bbox = None,
               dezoom_factor = 1., dtype = tf.float32, preserve_aspect_ratio = False):
    """
        Load an image to a tf.Tensor by supporting different formats
        
        Arguments :
            - filename  : either str (filename) or np.ndarray / tf.Tensor (image)
            - target_shape  : reshape the image to this shape (if provided)
            - mode      : 'rgb', 'gray' or None, convert the image to the appropriate output type
                If gray, the last dimension will be 1 and if 'rgb' will be 3. If 'None' the last dimension will be either 1 or 3 depending on the original image format
            - bbox      : [x, y, w, h] position to extract
            - dezoom_factor : the factor to multiply w and h
            - dtype     : tensorflow.dtype for the output image (automatically rescaled)
        Return :
            - image : 3-D tf.Tensor
        
        Note : if a filename is given, it loads the image with `tf.image.decode_image` which supports multiple types (see documentation for supportedformats)
    """
    assert mode in (None, 'rgb', 'gray')
    # Get filename / image from dict (if dict)
    if isinstance(filename, (dict, pd.Series)):
        filename = filename['image'] if 'image' in filename else filename['filename']

    # Convert filename to a tf.string Tensor (if necessary)
    if not isinstance(filename, tf.Tensor): filename = tf.convert_to_tensor(filename)
    
    if filename.dtype == tf.string:
        image = tf.io.read_file(filename)
        image = tf.image.decode_image(image, channels = channels, expand_animations = False)
    else:
        image = filename
    
    if bbox is not None:
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        if dezoom_factor != 1.:
            new_h = tf.cast(tf.math.round(tf.cast(h, tf.float32) * dezoom_factor), tf.int32)
            new_w = tf.cast(tf.math.round(tf.cast(w, tf.float32) * dezoom_factor), tf.int32)
            y = y - (new_h - h) / 2
            x = x - (new_w - w) / 2
            h, w = new_h, new_w
        
        x, y = tf.cast(tf.maximum(x, 0), tf.int32), tf.cast(tf.maximum(y, 0), tf.int32)
        w, h = tf.minimum(w, tf.shape(image)[1] - x), tf.minimum(h, tf.shape(image)[0] - y)
        
        if w <= 0 or h <= 0:
            tf.print("Error with bbox :", bbox, ":", [x, y, w, h], "-", tf.shape(image))
        else:
            image = tf.image.crop_to_bounding_box(image, y, x, h, w)
    
    if image.dtype != dtype:
        image = tf.image.convert_image_dtype(image, dtype)
    
    if mode == 'gray' and image.shape[2] == 3:
        image = tf.image.rgb_to_grayscale(image)
    elif mode == 'rgb' and image.shape[2] == 1:
        image = tf.image.grayscale_to_rgb(image)
    
    if target_shape is not None:
        image = tf.image.resize(
            image, target_shape[:2], preserve_aspect_ratio = preserve_aspect_ratio
        )
        if preserve_aspect_ratio:
            pad_h = tf.cast(target_shape[0] - tf.shape(image)[0], tf.int32)
            pad_w = tf.cast(target_shape[1] - tf.shape(image)[1], tf.int32)
            
            padding = [(pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)]
            image   = tf.pad(image, padding)
    
    return image

def save_image(filename, image, ** kwargs):
    """
        Save given `image` to the given `filename` 
        The function internally `load_image` with `kwargs` to convert the image to `uint8` (which is required by `cv2.imwrite`)
        It means that you can apply different transformation (such as resizing / convert to grayscale) before saving. 
        Furthermore, this function can also be used to copy image (as the input to `load_image` can be a filename).
        
        Arguments :
            - filename  : filename where to save the image
            - image     : the image to save (any type supported by `load_image`)
            - kwargs    : kwargs passed to `load_image` when converting to uint8
    """
    kwargs['dtype'] = tf.uint8
    image = load_image(image, ** kwargs).numpy()[:, :, ::-1]
    
    cv2.imwrite(filename, image)
    return filename

def stream_camera(cam_id = 0, max_time = 60, transform_fn = None, ** kwargs):
    """
        Open your camera and stream it by applying `transform_fn` on each frame
        
        Arguments :
            - cam_id    : camera ID (0 is default camera)
            - max_time  : the maximum streaming time (press 'q' to quit before)
            - transform_fn  : callable, function applied on each frame which returns the modified frame
            - kwargs    : kwargs passed to `transform_fn`
    """
    cap = cv2.VideoCapture(cam_id)

    n, start = 0, time.time()
    while time.time() - start < max_time:
        ret, frame = cap.read()
        if frame is not None:
            n += 1
            if transform_fn is not None:
                frame = transform_fn(frame, ** kwargs)

            cv2.imshow("Camera {}".format(cam_id), frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
                
    total_time = time.time() - start
    logging.info("Streaming processed {} frames in {} ({:.2f} fps)".format(
        n, time_to_string(total_time), n / total_time
    ))
    cap.release()
    cv2.destroyAllWindows()

def build_gif(directory,
              img_name      = '*.png',
              filename      = 'result.gif',
              n_repeat      = 5,
              keep_frames   = 1
             ):
    """
        Creates a gif from all images in a given directory with given pattern name
        
        Arguments :
            - directory : directory where images are stored
            - img_name  : pattern for images to include in the gif (default *.png == 'all png files')
            - filename  : output filename
            - n_repeat  : number of time to repeat each image (to have a slower animation)
            - keep_frames   : keep each `n` image (other are skipped)
        Return :
            - filename  : the .gif output file
    """
    try:
        import imageio
    except ImportError as e:
        logging.error('`imageio` is not installed : run `pip install imageio`')
        return None
    
    image_names = os.path.join(directory, img_name)
    
    with imageio.get_writer(filename, mode = 'I') as writer:
        filenames = sorted(glob.glob(image_names))

        for i, img_filename in enumerate(filenames):
            if i % keep_frames != 0 and i < len(filenames) - 1: continue
            
            image = imageio.imread(img_filename)
            for _ in range(n_repeat):
                writer.append_data(image)
                    
    return filename

def build_sprite(images, directory = None, filename = 'sprite.jpg', image_size = 128):
    if directory is not None: filename = os.path.join(directory, filename)
    n = math.ceil(math.sqrt(len(images)))
    
    sprite = np.zeros((n * image_size, n * image_size, 3), dtype = np.float32)
    
    for i, img in enumerate(images):
        img = load_image(img, target_shape = (image_size, image_size, 3), dtype = tf.float32)
        
        row, col = i // n, i % n
        
        sprite[
            row * image_size : (row + 1) * image_size,
            col * image_size : (col + 1) * image_size
        ] = img
    
    return save_image(filename = filename, image = sprite)
