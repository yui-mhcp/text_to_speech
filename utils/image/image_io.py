
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
import collections
import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image

from utils.image.image_utils import resize_image
from utils.generic_utils import time_to_string
from utils.thread_utils import Producer, Consumer
try:
    from loggers import timer
except ImportError:
    timer = lambda fn: fn

logger = logging.getLogger(__name__)

DELAY_MS = 10

def display_image(image):
    """
        Displays the image with `IPython.display.Image`
        You can also use `plot(image)` or `plot_multiple(...)` (to display multiple images)
    """
    from IPython.display import Image, display
    display(Image(image))
    
def get_image_size(image):
    """
        Return image size [height, width] (supporting different formats) 
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
    
@timer
def load_image(filename, target_shape = None, mode = None, channels = 3, bbox = None,
               dtype = tf.float32, preserve_aspect_ratio = False, resize_kwargs = {}, ** kwargs):
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
    if not isinstance(filename, tf.Tensor):
        if isinstance(filename, str): conv_dtype = tf.string
        elif isinstance(filename, np.ndarray):
            if filename.dtype == np.uint8:                      conv_dtype = tf.uint8
            elif filename.dtype in (np.float32, np.float64):    conv_dtype = tf.float32
            elif filename.dtype in (np.int32, np.int64):        conv_dtype = tf.int32
            else: raise ValueError('Unknown image type : {}'.format(filename.dtype))
        else: raise ValueError('Unknown image type : {}\n{}'.format(type(filename), filename))
        
        filename = tf.cast(filename, conv_dtype)

    if filename.dtype == tf.string:
        image = tf.io.read_file(filename)
        image = tf.image.decode_image(image, channels = channels, expand_animations = False)
    else:
        image = filename
    
    if bbox is not None:
        from utils.image.box_utils import crop_box
        image, _ = crop_box(image, bbox, ** kwargs)
    
    if image.dtype != dtype:
        image = tf.image.convert_image_dtype(image, dtype)
    
    if mode == 'gray' and image.shape[2] == 3:
        image = tf.image.rgb_to_grayscale(image)
    elif mode == 'rgb' and image.shape[2] == 1:
        image = tf.image.grayscale_to_rgb(image)
    
    if target_shape is not None:
        resize_kwargs['preserve_aspect_ratio'] = preserve_aspect_ratio
        image = resize_image(image, target_shape, ** resize_kwargs)
    
    return image

def convert_to_uint8(image, ** kwargs):
    """ Converts `image` to `np.uint8` format """
    if isinstance(image, np.ndarray) and image.dtype in (np.uint8, np.float32, np.float64):
        if image.dtype != np.uint8: image = (image * 255).astype(np.uint8)
    else:
        kwargs['dtype'] = tf.uint8
        image = load_image(image, ** kwargs).numpy()

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
    image = convert_to_uint8(image, ** kwargs)
    
    cv2.imwrite(filename, image[:, :, ::-1])
    return filename

@timer
def stream_camera(cam_id    = 0,
                  max_time  = 60,
                  nb_frames = -1,
                  
                  fps   = -1,
                  show_fps  = None,
                  
                  max_workers   = 0,
                  
                  id_format = None,
                  transform_fn  = None,
                  transformer_prod  = None,
                  
                  output_file   = None,
                  output_fps    = None,
                  output_shape  = None,
                  
                  show  = True,
                  play_audio    = False,
                  copy_audio    = True,
                  
                  view_graph        = False,
                  output_graph      = False,
                  graph_filename    = None,
                  
                  ** kwargs
                 ):
    """
        Streams either on a camera either on a video file (depending on `cam_id`) and applies `transformer_fn` on each frame.
        
        Arguments :
            - cam_id    : camera ID (0 is default camera) or video filename (str)
            - max_time  : the maximum streaming time (press 'q' to quit before if `show = True`)
            - nb_frames : number of frames to stream (quit after)
            
            - fps   : frames per second for the input stream
            - show_fps  : frames per second for the `cv2.imshow` call
            
            - max_workers   : number of workers for each `Consumer`
                - -1 / -2   : each function is executed in the main thread (no separated thread) 
                - 0     : each function is executed in a separate thread
                - > 0   : each function is executed in `n` separated threads
            
            - id_format : the name format for the ith frame. If provided, the generator's output (i.e. the input to `transform_fn`) is a dict {id: , image:} where `id_format` is formatted with the frame index
            - transform_fn  : valid argument to the `add_consumer` method. Its input is either a dict (if `id_format` is provided) either the frame
            - transformer_prod  : a `Consumer` object that outputs the transformed frames to display (default to `prod.add_consumer(transform_fn)`)
            
            - output_file   : the output filename to save the (transformed) video
            - output_fps    : the fps for the output video file
            - output_shape  : output shape of the transformed video (default to the camera / input file's shape)
            
            - show  : whether to display the result or not (with `cv2.imshow`)
            - play_audio    : whether to play the audio (ignored if `cam_id` is not a video file)
            - copy_audio    : whether to copy the input file's audio to the output's file
            
            - kwargs    : kwargs passed to the `transform_fn` pipeline
    """
    def should_continue(t0, t, n):
        run = True
        if max_time is not None:    run = run and t - t0 < max_time
        if nb_frames > 0:           run = run and n < nb_frames
        return run
    
    def frame_producer_new():
        """ Produces `fps` frames per second from `cap` """
        time_between_frames = 1. / fps
        
        n = 0
        t = time.time()
        next_time = t
        while should_continue(start_time, t, n):
            if t < next_time: time.sleep(next_time - t)
            next_time += time_between_frames
            ret, frame = cap.read()
            if ret: yield frame[..., ::-1]
            n += 1
            t = time.time()

    def frame_producer():
        """ Produces `fps` frames per second """
        n = 0
        start_iter_time = time.time()
        while should_continue(start_time, start_iter_time, n):
            ret, frame = cap.read()
            if ret:
                frame = frame[..., ::-1]
                yield frame if id_format is None else {
                    'id' : id_format.format(n), 'image' : frame
                }
            
            n += 1
            wait_time = wait_time_sec - (time.time() - start_iter_time)
            if fps > 0 and wait_time > 0: time.sleep(wait_time)
            start_iter_time = time.time()

    @timer
    def write_video(frame):
        """ Writes frames to the corresponding output file """
        frame = convert_to_uint8(frame, ** kwargs)
        video_writer.write(frame[:, :, ::-1])
    
    @timer
    def show_frame(frame, prev_time = -1):
        """ Displays `fps` frames per second with `cv2.imshow` """
        cv2.imshow(display_name, frame[..., ::-1])
        
        t = time.time()
        delay    = 0 if prev_time == -1 else (t - prev_time)
        delay_ms = int(delay * 1000)
        if cv2.waitKey(max(show_wait_time_ms - delay_ms, 1)) & 0xFF == ord('q'):
            raise StopIteration()
        return None, (time.time(), )
    
    def _play_first_frame(frame, first = True):
        """ Runs the audio once the first frame has been displayed """
        if first:
            from utils.audio import audio_io
            audio_io.play_audio(cam_id, block = False)
        return None, (False, )
    # variable initialization
    if max_time <= 0:        max_time = None
    if transform_fn is None:    id_format   = None
    
    display_name = '{} {}'.format('Camera' if isinstance(cam_id, int) else 'File', cam_id)
    cap = cv2.VideoCapture(cam_id)
    
    if output_fps is None: output_fps = fps
    # if streaming is not on camera but on video file
    if isinstance(cam_id, str):
        video_frames    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        nb_frames   = video_frames if nb_frames <= 0 else min(nb_frames, video_frames)
        output_fps  = cap.get(cv2.CAP_PROP_FPS)
        logger.info('Start streaming on a video with {:.3f} fps and {} frames'.format(
            output_fps, video_frames
        ))
    
    # set the output fps
    if output_file and output_fps == -1:
        logger.warning('When specifying an `output_file`, it is recommanded to specify `output_fps` as the `fps` can differ from the effective camera fps')
        output_fps  = cap.get(cv2.CAP_PROP_FPS) if fps == -1 else fps
    # set the display fps
    if play_audio and isinstance(cam_id, str):
        show_fps    = output_fps
    elif show_fps is None:
        show_fps    = fps
    # set the waiting time according to the frame rate
    wait_time_ms    = int(max(1000 // fps - DELAY_MS + 1, 1))
    wait_time_sec   = wait_time_ms / 1000.
    
    show_wait_time_ms    = int(max(1000 // show_fps - DELAY_MS + 1, 1))
    show_wait_time_sec   = show_wait_time_ms / 1000.

    #####################
    # Init the pipeline #
    #####################
    
    prod = Producer(
        frame_producer, run_main_thread = max_workers < 0, stop_listeners = cap.release
    )
    
    transformer = prod.add_consumer(
        transform_fn, link_stop = True, max_workers = max_workers, ** kwargs
    ) if transform_fn is not None else prod
    
    if transformer_prod is None: transformer_prod = transformer
    # Adds a frame writer if required
    if output_file is not None:
        if output_shape is None:
            frame_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        else:
            frame_h, frame_w = output_shape

        video_writer    = cv2.VideoWriter(
            output_file, cv2.VideoWriter_fourcc(*'MPEG'), output_fps, (frame_w, frame_h)
        )
        # Creates the video-writer consumer
        writer_cons     = transformer_prod.add_consumer(
            write_video, link_stop = True, start = True, max_workers = min(max_workers, 0),
            stop_listeners = video_writer.release
        )
        # Adds a listener to copy the video file's audio to the output file (if expected)
        if copy_audio:
            from utils.image import video_utils
            writer_cons.add_listener(
                lambda: video_utils.copy_audio(cam_id, output_file), on = 'stop'
            )
    
    # Adds a consumer to display frames (if expected)
    if show:
        cons = transformer_prod.add_consumer(
            show_frame, start = True, link_stop = True, stateful = True,
            max_workers     = min(max_workers, 0),
            start_listeners = lambda: cv2.namedWindow(display_name, cv2.WINDOW_NORMAL),
            stop_listeners  = cv2.destroyAllWindows
        )
        if play_audio and output_fps != -1 and isinstance(cam_id, str):
            cons.add_consumer(_play_first_frame, on = 'item', stateful = True, max_workers = -1)
    
    start_time = time.time()
    
    prod.start()
    graph = prod.plot(filename = graph_filename, view = view_graph)[0] if view_graph or graph_filename or output_graph else None
    # waits until all consumers are finished
    prod.join(recursive = True)
    
    total_time = time.time() - start_time
    logger.info("Streaming processed {} frames in {} ({:.2f} fps)".format(
        prod.size, time_to_string(total_time), prod.size / total_time
    ))
    return graph

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
        logger.error('`imageio` is not installed : run `pip install imageio`')
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
    """
        Writes all `images` (iterable of images) in a 'sprite' 
        A 'sprite' is a big square image showing all images as a table where image[i] is at position `(i // n, i % n)` where `n = ceil(sqrt(len(images)))`
    """
    if directory is not None: filename = os.path.join(directory, filename)
    n = math.ceil(math.sqrt(len(images)))
    
    sprite = np.zeros((n * image_size, n * image_size, 3), dtype = np.float32)
    
    for i, img in enumerate(images):
        img = load_image(img, target_shape = (image_size, image_size, 3), dtype = tf.float32).numpy()
        
        row, col = i // n, i % n
        
        sprite[
            row * image_size : (row + 1) * image_size,
            col * image_size : (col + 1) * image_size
        ] = img
    
    return save_image(filename = filename, image = sprite)
