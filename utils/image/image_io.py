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

import os
import cv2
import PIL
import sys
import glob
import math
import time
import inspect
import logging
import warnings
import collections
import numpy as np
import pandas as pd

from loggers import timer, time_logger
from utils.keras_utils import TensorSpec, ops, graph_compile
from utils import Producer, Consumer, time_to_string, partial
from utils.threading import Producer, Consumer
from utils.image.image_utils import resize_image

logger  = logging.getLogger(__name__)

DELAY_MS = 3
DELAY_SEC   = DELAY_MS / 1000.

_resize_kwargs  = set(list(inspect.signature(resize_image).parameters.keys())[1:])

def display_image(image):
    """
        Displays the image with `IPython.display.Image`
        You can also use `utils.plot(image)` or `plot_multiple(...)` to display (multiple) images(s)
    """
    from IPython.display import Image, display
    if ops.is_tensor(image): image = ops.convert_to_numpy(image)
    display(Image(image))
    
def get_image_size(image):
    """
        Returns the image size [height, width] (supporting different formats) 
        Arguments :
            - image : str (filename) or 2, 3 or 4-D `np.ndarray` / `Tensor` (raw image(s))
        Return :
            - [height, width]   : image size
    """
    if hasattr(image, 'shape'):
        shape = ops.shape(image)
        if len(shape) == 2:
            return shape[0], shape[1]
        else:
            return shape[-3], shape[-2]
    elif isinstance(image, str):
        from PIL import Image
        
        with Image.open(image) as img:
            return img.size[::-1]
    else:
        raise ValueError("Unknown image type : {}\n{}".format(type(image), image))
    
@timer
#@graph_compile(
#    support_xla = False, cast_kwargs = False, force_tensorflow = True
#)
def load_image(filename : TensorSpec(),

               channels = 3,
               mode     = None,
               dtype    = 'float32',
               to_tensor    = True,
               
               boxes    = None,
               
               ** kwargs
              ):
    """
        Load an image to a Tensor by supporting different formats / extensions
        
        Arguments :
            - filename  : either `str` (filename) either `np.ndarray` / `Tensor` (raw image)
            
            - dtype     : the expected output dtype
            - channels  : required kwarg for `tf.image.decode_image`
            - mode      : 'rgb', 'gray' or None, convert the image to the appropriate output type
                If gray, the last dimension will be 1 and if 'rgb' will be 3. If 'None' the last dimension will be either 1 or 3 depending on the original image format
            - to_tensor : converts the output image to `Tensor`
            
            - boxes     : [x, y, w, h] position to extract
            - kwargs    : forwarded to `utils.image.bounding_box.crop_box` if `bbox` is provided
        Return :
            - image : 3-D `Tensor` if `to_tensor == True`, `np.ndarray | Tensor` otherwise
        
        Note : if a filename is given, it loads the image with `tf.image.decode_image` (if tensorflow backend) or `PIL.Image.load` otherwise
    """
    assert mode in (None, 'rgb', 'gray')
    # Get filename / image from dict (if dict)
    if isinstance(filename, (dict, pd.Series)):
        filename = filename['image' if 'image' in filename else 'filename']

    # Convert filename to a Tensor (if necessary)
    # Note : inferring the output type then use `tf.cast` is faster than `tf.convert_to_tensor` and allows some type checking
    if ops.is_string(filename):
        if ops.is_tensorflow_backend() or ops.is_tensorflow_graph():
            import tensorflow as tf
            
            image = tf.io.read_file(filename)
            image = tf.image.decode_image(image, channels = channels, expand_animations = False)
        else:
            image = np.array(PIL.Image.open(filename))
            if to_tensor: image = ops.convert_to_tensor(image)
    else:
        image = filename
        if to_tensor: image = ops.convert_to_tensor(image)
        if len(ops.shape(image)) == 2: image = image[:, :, None]

    if boxes is not None:
        from utils.image.bounding_box import crop_box
        _, image = crop_box(image, boxes, ** kwargs)
    
    if dtype is not None:
        image = ops.convert_data_dtype(image, dtype = dtype)

    if mode == 'gray' and image.shape[2] == 3:
        image = ops.rgb_to_grayscale(image)
    elif mode == 'rgb' and image.shape[2] == 1:
        image = ops.grayscale_to_rgb(image)

    if any(kwargs.get(k, None) is not None for k in _resize_kwargs):
        image = resize_image(image, ** kwargs)
        if (isinstance(kwargs.get('target_shape', None), tuple)
            and all(s != -1 for s in kwargs['target_shape'])
            and not ops.executing_eagerly()
           ):
            image = ops.ensure_shape(image, kwargs['target_shape'][:2] + (image.shape[-1],))

    return image

def convert_to_uint8(image, ** kwargs):
    """ Converts `image` to `np.uint8` format (useful for subsequent `cv2` calls) """
    return ops.convert_to_numpy(load_image(
        image, dtype = 'uint8', to_tensor = False, run_eagerly = True, ** kwargs
    ))

@timer
def save_image(filename, image, ** kwargs):
    """
        Save given `image` to the given `filename`
        If `image` is not a np.ndarray, the function internally calls `load_image` with `kwargs`, to convert the image to `np.uint8` (which is required by `cv2.imwrite`)
        It means that you can apply different transformation (such as resizing / convert to grayscale) before saving. 
        Furthermore, this function can also be used to copy image (as the input to `load_image` can be a filename).
        
        Arguments :
            - filename  : filename where to save the image
            - image     : the image to save (any type supported by `load_image`)
            - kwargs    : kwargs passed to `load_image` when converting to uint8 (if not already a np.ndarray)
        Return :
            - filename  : the image filename (the argument)
    """
    image = convert_to_uint8(image, ** kwargs)
    
    cv2.imwrite(filename, image[:, :, ::-1])
    return filename

def frame_generator(cam_id,
                    fps         = None,
                    max_time    = None,
                    
                    nb_frames   = -1,
                    frames_step = 1,
                    frames_offset   = 0,
                    
                    add_copy    = False,
                    return_index    = False,
                    
                    max_failure = 5,
                    
                    ** kwargs
                   ):
    """
        Yields `fps` frames per second from the given camera (`cam_id`)
        
        Arguments :
            - cam_id    : the camera id (any value supported by `cv2.VideoCapture`) or the raw object
            - fps       : the number of frames to generate per second
            - max_time  : the maximum generation time
            - nb_frames : the maximum number of frames to generate
            - kwargs    : forwarded to `cv2.VideoCapture`
        Return :
            - n     : the number of generated frames
    """
    def should_continue(t0, t, n):
        run = True
        if max_time > 0:    run = run and t - t0 < max_time
        if nb_frames > 0:   run = run and n < nb_frames
        return run
    
    if not max_time: max_time = -1
    if not nb_frames: nb_frames = -1
    if max_time == -1 and nb_frames == -1: max_time = 60
    
    camera  = cam_id
    if isinstance(cam_id, (int, str)):
        camera = cv2.VideoCapture(cam_id, ** kwargs)
    
    failed      = 0
    idx, seen   = frames_offset, 0
    wait_time   = 1. / fps - DELAY_SEC if fps else 1e-3
    start_time  = time.time()
    start_iter_time = start_time
    
    for _ in range(frames_offset): camera.grab()
    while should_continue(start_time, start_iter_time, seen):
        with time_logger.timer('frame generation'):
            ret, frame = camera.read()
            if not ret:
                failed += 1
                if failed == max_failure: break
            if frames_step > 1 and idx % frames_step != 0:
                idx += 1
                continue

            if ret:
                frame = frame[..., ::-1]
                if add_copy or return_index:
                    data = {'image' : frame, 'frame_index' : idx - 1}
                    if add_copy: data['image_copy'] = frame.copy()
                    yield data
                else:
                    yield frame
            
            idx += 1
            seen += 1
            wait = max(1e-2, wait_time - (time.time() - start_iter_time))
            if fps > 0 and wait > 0: time.sleep(wait)
            start_iter_time = time.time()
    
    return seen

@timer
def stream_camera(cam_id    = 0,
                  *,
                  
                  max_time  = None,
                  nb_frames = -1,
                  frames_step   = 1,
                  frames_offset = 0,
                  
                  fps   = -1,
                  show_fps  = None,
                  use_multithreading    = True,
                  
                  add_copy  = False,
                  add_index = False,
                  buffer_size   = 5,
                  transform_fn  = None,
                  
                  copy_audio    = True,
                  output_fps    = None,
                  output_shape  = None,
                  output_file   = None,
                  transformed_file  = None,
                  
                  show  = True,
                  flags = cv2.WINDOW_AUTOSIZE  | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED,
                  play_audio    = True,
                  
                  view_graph        = False,
                  output_graph      = False,
                  graph_filename    = None,
                  
                  ** kwargs
                 ):
    """
        Streams either on a camera either on a video file (depending on `cam_id`) and applies `transformer_fn` on each frame.
        
        Arguments :
            - cam_id    : the video stream
                          - int : camera ID (0 is the default camera)
                          - str : video filename
                          - other   : should be a valid camera (i.e., implement `read()` method)
            
            - max_time  : maximum number of streaming time (in seconds)
            - nb_frames : maximum number of frames to stream
            - frames_step   : the number of frames to skip (e.g., `2` will take 1 frame out of 2)
            - frames_offset : the number of frames to skip at the start of the stream
            
            - fps   : number of frames to load per second
            - show_fps  : number of frames to display per second (only relevant if `show = True`)
            - use_multithreading    : whether to parallelize frames reading / display
            
            - add_copy  : add the `image_copy` field in the stream generator output
            - add_index : add the `frame_index` field in the stream generator output
            - buffer_size   : the buffer size of the transformer consumer (allows to control the number of frames in the waiting queue)
            - transform_fn  : a callable that takes 1 argument, and outputs a transformed frame
                              the input is a dict if `add_index or add_copy` or the raw frame
                              the output is used only if `show = True or transformed_file`
            
            - copy_audio    : whether to copy original video audio to the output file(s)
            - output_fps    : the frame rate for the output file(s)
            - output_shape  : the frame shape for the output file(s)
            - output_file   : where to save the raw stream
            - transformed_file  : where to save the transformed frame (require `transform_fn`)
            
            - show  : whether to display the transformed stream with `cv2.imshow
            - flags : the flags to `cv2.namedWindow` to initialize the display window
            - play_audio    : whether to play the original audio when showing the stream
            
            - {view / output}_graph : arguments forwarded to the `Producer.plot` method
            
            - kwargs    : forwarded to `transform_fn`
    """
    @timer
    def write_video_frame(writer, frame):
        """ Writes frames to the corresponding output file """
        frame = convert_to_uint8(frame)
        writer.write(frame[:, :, ::-1])
    
    @timer
    def show_frame(frame, prev_time = -1):
        """ Displays `fps` frames per second with `cv2.imshow` """
        if isinstance(frame, dict): frame = frame['image']
        if callable(frame):         frame = frame()
        cv2.imshow(display_name, frame[..., ::-1])
        
        t = time.time()
        delay    = 0 if prev_time == -1 else (t - prev_time)
        delay_ms = max(show_wait_time_ms - int(delay * 1000), 1) if use_multithreading else 1
        if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
            raise StopIteration()
        return None, (t, )
    
    def play_audio_on_first_frame(frame, first = True):
        """ Runs the audio once the first frame has been displayed """
        from utils.audio import audio_io
        audio_io.play_audio(cam_id, blocking = False)
        raise StopIteration()
    
    # variable initialization
    if isinstance(cam_id, (int, str)):
        display_name = '{} {}'.format('Camera' if isinstance(cam_id, int) else 'File', cam_id)
        cap = cv2.VideoCapture(cam_id)
    else:
        display_name    = str(cam_id)
        cap = cam_id
    
    # if streaming is not on camera but on video file
    if isinstance(cam_id, str):
        video_frames    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        nb_frames   = video_frames if nb_frames <= 0 else min(nb_frames, video_frames)
        output_fps  = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info('Start streaming on a video with {:.3f} fps and {} frames'.format(
            output_fps, video_frames
        ))
    
    # set the output fps
    if not output_fps: output_fps = fps
    if output_file or transformed_file:
        if output_shape is None:
            frame_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        else:
            frame_h, frame_w = output_shape

        if output_fps == -1:
            warnings.warn('When specifying an `output_file`, it is recommanded to specify `output_fps` as the `fps` can differ from the effective camera fps')
            output_fps  = cap.get(cv2.CAP_PROP_FPS) if fps == -1 else fps
    
    # set the display fps
    if play_audio and isinstance(cam_id, str):
        fps         = min(fps if fps != -1 else output_fps * 5, output_fps * 5)
        show_fps    = output_fps
    elif show_fps is None:
        show_fps    = fps
    # set the waiting time according to the frame rate
    show_wait_time_ms    = int(max(1000 // show_fps - DELAY_MS, 1))

    #####################
    # Init the pipeline #
    #####################
    
    prod = Producer(frame_generator(
        cap,
        fps     = fps,
        max_time    = max_time,
        nb_frames   = nb_frames,
        frames_step = frames_step,
        frames_offset   = frames_offset,
        
        add_copy    = add_copy,
        return_index = add_index
    ), run_main_thread = not use_multithreading, stop_listener = cap.release)
    
    transformer = prod.add_consumer(
        transform_fn,
        start   = True,
        link_stop   = True,
        buffer_size = buffer_size,
        run_main_thread = not use_multithreading,
        ** kwargs
    ) if transform_fn is not None else prod
    
    # Adds a consumer to display frames (if expected)
    if show:
        cons = transformer.add_consumer(
            show_frame,
            link_stop = True,
            stateful = True,
            run_main_thread = not use_multithreading,
            stop_no_more_listeners  = False,
            start_listener  = lambda: cv2.namedWindow(display_name, flags),
            stop_listener   = cv2.destroyAllWindows
        )
        if play_audio and output_fps != -1 and isinstance(cam_id, str):
            cons.add_listener(play_audio_on_first_frame)

    # Adds a frame writer if required
    if output_file is not None:
        os.makedirs(os.path.dirname(output_file), exist_ok = True)
        video_writer    = cv2.VideoWriter(
            output_file, cv2.VideoWriter_fourcc(*'MPEG'), output_fps, (frame_w, frame_h)
        )
        # Creates the video-writer consumer
        writer_cons     = prod.add_consumer(
            partial(write_video_frame, video_writer),
            link_stop = True,
            run_main_thread = not use_multithreading,
            stop_listener   = video_writer.release
        )
        # Adds a listener to copy the video file's audio to the output file (if expected)
        if copy_audio and isinstance(cam_id, str):
            from utils.image import video_utils
            writer_cons.add_listener(
                lambda: video_utils.copy_audio(cam_id, output_file), event = 'stop'
            )
    
    if transformed_file is not None:
        if transform_fn is None:
            raise RuntimeError('When `transformed_file` is provided, `transform_fn` must not be None')
        
        os.makedirs(os.path.dirname(transformed_file), exist_ok = True)

        transformed_video_writer    = cv2.VideoWriter(
            transformed_file, cv2.VideoWriter_fourcc(*'MPEG'), output_fps, (frame_w, frame_h)
        )
        # Creates the video-writer consumer
        writer_cons     = transformer.add_consumer(
            partial(write_video_frame, transformed_video_writer),
            link_stop = True,
            run_main_thread = not use_multithreading,
            stop_listener   = transformed_video_writer.release
        )
        # Adds a listener to copy the video file's audio to the output file (if expected)
        if copy_audio and isinstance(cam_id, str):
            from utils.image import video_utils
            writer_cons.add_listener(
                lambda: video_utils.copy_audio(cam_id, transformed_file), event = 'stop'
            )
    
    ####################
    #  Start pipeline  #
    ####################
    
    start_time = time.time()
    
    prod.start()
    
    graph   = None
    if view_graph or graph_filename or output_graph:
        graph = prod.plot(filename = graph_filename, view = view_graph)[0]
    
    # waits until all consumers are finished
    prod.join(recursive = True)
    
    total_time = time.time() - start_time
    logger.info("Streaming processed {} frames in {} ({:.2f} fps)".format(
        prod.size, time_to_string(total_time), prod.size / total_time
    ))
    return graph

def build_gif(images, filename = 'result.gif', n_repeat = 5, keep_frames = 1):
    """
        Creates a gif from all images in a given set of images
        
        Arguments :
            - images    : the images, a single image can be of any type supported by `load_image`
                - iterable      : an iterable of image
                - directory name    : a directory containing the images
                - filename with '*' : a filename pattern (as supported by `glob.glob`)
            - filename  : the sprite filename
            - n_repeat  : number of time to repeat each image (to have a slower animation)
            - keep_frames   : keep one image out of `keep_frames` (other are skipped)
        Return :
            - filename  : the .gif output file
    """
    try:
        import imageio
    except ImportError as e:
        logger.error('`imageio` is not installed : run `pip install imageio`')
        return None
    
    if isinstance(images, str):
        if os.path.isdir(images):
            images = sorted([os.path.join(images, f) for f in os.listdir(images)])
        elif '*' in images:
            images = sorted(glob.glob(images))
        else:
            raise ValueError('When `images` is a string, it must be a directory or a filename pattern (i.e. with a "*")')
    
    with imageio.get_writer(filename, mode = 'I') as writer:
        for i, image in enumerate(images):
            if i % keep_frames != 0 and i != len(images) - 1: continue
            
            if isinstance(image, str):
                image = imageio.imread(image)
            for _ in range(n_repeat):
                writer.append_data(image)

    return filename

def build_sprite(images, image_size = 128, directory = None, filename = 'sprite.jpg'):
    """
        Writes all `images` (iterable of images) in a 'sprite' 
        A 'sprite' is a big square image showing all images as a table where image[i] is at position `(i // n, i % n)` where `n = ceil(sqrt(len(images)))`
        
        Arguments :
            - images    : the images, a single image can be of any type supported by `load_image`
                - iterable      : an iterable of image
                - directory name    : a directory containing the images
                - filename with '*' : a filename pattern (as supported by `glob.glob`)
            - image_size    : a single image size in the sprite
            - directory : where to save the image
            - filename  : the sprite filename
        Return :
            - filename  : the sprite filename
    """
    if isinstance(images, str):
        if os.path.isdir(images):
            images = sorted([os.path.join(images, f) for f in os.listdir(images)])
        elif '*' in images:
            images = sorted(glob.glob(images))
        else:
            raise ValueError('When `images` is a string, it must be a directory or a filename pattern (i.e. with a "*")')
    
    if directory is not None: filename = os.path.join(directory, filename)
    n = math.ceil(math.sqrt(len(images)))
    
    sprite = np.zeros((n * image_size, n * image_size, 3), dtype = np.uint8)
    
    for i, img in enumerate(images):
        img = load_image(
            img, target_shape = (image_size, image_size, 3), dtype = 'uint8',
            to_tensor = False, run_eagerly = True
        )
        
        row, col = i // n, i % n
        
        sprite[
            row * image_size : (row + 1) * image_size,
            col * image_size : (col + 1) * image_size
        ] = img
    
    return save_image(filename = filename, image = sprite)
