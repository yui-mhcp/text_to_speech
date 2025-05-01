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

import os
import PIL
import sys
import glob
import math
import time
import queue
import inspect
import logging
import warnings
import collections
import numpy as np

from functools import partial

from ..keras import ops
from loggers import Timer, timer
from ..threading import Stream
from ..generic_utils import time_to_string
from .image_processing import resize_image

logger  = logging.getLogger(__name__)

DELAY_MS = 3
DELAY_SEC   = DELAY_MS / 1000.

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
        shape = image.shape
        if len(shape) == 2:
            return shape[0], shape[1]
        else:
            return shape[-3], shape[-2]
    elif isinstance(image, str):
        with PIL.Image.open(image) as img:
            return img.size[::-1]
    else:
        raise ValueError("Unknown image type : {}\n{}".format(type(image), image))
    
@timer
def load_image(filename,
               *,
               
               size = None,
               multiples    = None,
               
               dtype    = None,
               channels = 3,
               to_tensor    = False,
               
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
    assert channels in (1, 3)
    # Get filename / image from dict (if dict)
    if isinstance(filename, dict):
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
        if len(image.shape) == 2: image = image[:, :, None]

    if boxes is not None:
        from .bounding_box import crop_box
        _, image = crop_box(image, boxes, ** kwargs)
    
    if dtype is not None:
        image = ops.convert_data_dtype(image, dtype = dtype)

    if channels == 1 and image.shape[2] == 3:
        image = ops.rgb_to_grayscale(image)
    elif channels == 3 and image.shape[2] == 1:
        image = ops.grayscale_to_rgb(image)
    
    if size is not None or multiples is not None:
        image = resize_image(image, size, multiples = multiples, ** kwargs)

    return image

def convert_to_uint8(image, ** kwargs):
    """ Converts `image` to `np.uint8` format (useful for subsequent `cv2` calls) """
    return ops.convert_to_numpy(load_image(
        image, dtype = 'uint8', run_eagerly = True, ** kwargs
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
    import cv2
    
    image = convert_to_uint8(image, ** kwargs)
    
    cv2.imwrite(filename, image[:, :, ::-1])
    return filename

def set_video_audio(video_filename, audio_filename, codec = 'aac', bitrate = '128k'):
    import ffmpeg
    
    ffmpeg.output(
        ffmpeg.input(video_filename).video,
        ffmpeg.input(audio_filename, vn = None, dn = None).audio,
        video_filename[:-4] + '_audio.mp4',
        acodec = codec,
        audio_bitrate = bitrate,
        loglevel = 'error',
        scodec   = 'copy'
    ).overwrite_output().run()

@timer
def stream_camera(cam_id    = 0,
                  *,
                  
                  max_time  = None,
                  nb_frames = -1,
                  frames_step   = 1,
                  frames_offset = 0,
                  
                  fps   = -1,
                  show_fps  = None,
                  
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
                  flags = None,
                  play_audio    = True,
                  
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
    import cv2
    
    if flags is None: flags = cv2.WINDOW_AUTOSIZE  | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED
    
    # variable initialization
    if isinstance(cam_id, (int, str)):
        display_name = '{} {}'.format('Camera' if isinstance(cam_id, int) else 'File', cam_id)
        camera = cv2.VideoCapture(cam_id)
    else:
        display_name    = str(cam_id)
        camera = cam_id
    
    # if streaming is not on camera but on video file
    if isinstance(cam_id, str):
        video_frames    = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
        nb_frames   = video_frames if nb_frames <= 0 else min(nb_frames, video_frames)
        output_fps  = camera.get(cv2.CAP_PROP_FPS)
        
        logger.info('Start streaming on a video with {:.3f} fps and {} frames'.format(
            output_fps, video_frames
        ))
    elif not output_fps:
        output_fps = fps
    
    # set the output fps
    if output_file or transformed_file:
        if output_shape is None:
            frame_h     = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_w     = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        else:
            frame_h, frame_w = output_shape

        if output_fps == -1:
            warnings.warn('When specifying an `output_file`, it is recommanded to specify `output_fps` as the `fps` can differ from the effective camera fps')
            output_fps  = cap.get(cv2.CAP_PROP_FPS) if fps == -1 else fps
    
    # set the display fps
    if show and play_audio and isinstance(cam_id, str):
        fps         = min(fps if fps != -1 else output_fps * 5, output_fps * 5)
        show_fps    = output_fps
    elif show_fps is None:
        show_fps    = fps

    #####################
    # Init the pipeline #
    #####################
    
    stream_state = {'stop' : False}
    
    if show:
        stream_show = Stream(
            partial(_show_frame, fps = show_fps, display_name = display_name, stream_state = stream_state, audio_file = cam_id if isinstance(cam_id, str) and play_audio else None),
            
            max_workers = 1,
            start_callback  = lambda: cv2.namedWindow(display_name, flags),
            stop_callback   = cv2.destroyAllWindows
        )
    else:
        stream_show = lambda x: x
    
    if output_file:
        if os.path.dirname(output_file):
            os.makedirs(os.path.dirname(output_file), exist_ok = True)
        video_writer    = cv2.VideoWriter(
            output_file, cv2.VideoWriter_fourcc(*'MPEG'), output_fps, (frame_w, frame_h)
        )
        stream_write = Stream(
            partial(_write_video_frame, video_writer),
            max_workers = 1,
            stop_callback   = video_writer.release
        )
    else:
        stream_write = lambda x: x

    callbacks = []
    if transform_fn is not None:
        if show:
            callbacks.append(stream_show)
            stream_show = lambda x: x
        
        if transformed_file:
            if os.path.dirname(transformed_file):
                os.makedirs(os.path.dirname(transformed_file), exist_ok = True)
            video_writer    = cv2.VideoWriter(
                transformed_file, cv2.VideoWriter_fourcc(*'MPEG'), output_fps, (frame_w, frame_h)
            )
            callbacks.append(Stream(
                partial(_write_video_frame, video_writer),
                max_workers = 1,
                stop_callback   = video_writer.release
            ))

        stream_transform = Stream(
            transform_fn,
            stream  = queue.Queue(buffer_size),
            callback    = callbacks,
            max_workers = 1,
        )
    else:
        stream_transform = lambda x: x
    
    start_time = time.time()
    
    frame_stream = frame_generator(
        cam_id  = camera,
        fps     = fps,
        max_time    = max_time,
        nb_frames   = nb_frames,
        frames_step = frames_step,
        frames_offset   = frames_offset,

        add_copy    = add_copy,
        add_index   = add_index,
        stream_state    = stream_state
    )
    try:
        idx = 0
        for idx, frame in enumerate(frame_stream):
            if stream_state['stop'] or getattr(stream_transform, 'stopped', False): break
            stream_transform(frame)
            stream_show(frame)
            stream_write(frame)

        print('Stream stopped')
        for callback in callbacks + [stream_transform, stream_show, stream_write]:
            if isinstance(callback, Stream): callback.join(force = stream_state['stop'])
        
        if copy_audio and isinstance(cam_id, str):
            if output_file:         set_video_audio(output_file, cam_id)
            if transformed_file:    set_video_audio(transformed_file, cam_id)
    finally: # this is executed in all cases (error or not)
        frame_stream.close()
        camera.release()
        if 'player' in stream_state: stream_state['player'].stop()
        
        for callback in callbacks + [stream_transform, stream_show, stream_write]:
            if isinstance(callback, Stream): callback.join(force = True)
    
    total_time = time.time() - start_time
    logger.info("Streaming processed {} frames in {} ({:.2f} fps)".format(
        idx, time_to_string(total_time), idx / total_time
    ))

def frame_generator(cam_id,
                    *,
                    
                    fps     = None,
                    max_time    = None,
                    
                    nb_frames   = -1,
                    frames_step = 1,
                    frames_offset   = 0,
                    
                    add_copy    = False,
                    add_index   = False,
                    
                    max_failures    = 5,
                    
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
    import cv2
    
    if not max_time:  max_time = -1
    if not nb_frames: nb_frames = -1
    fps_time = None if not fps else 1. / fps
    
    camera  = cam_id
    if isinstance(cam_id, (int, str)):
        camera = cv2.VideoCapture(cam_id, ** kwargs)
    
    for _ in range(frames_offset): camera.read()
    
    start, idx, failed = time.time(), 0, 0
    start_iter_time = now = start
    while (failed <= max_failures) and (nb_frames == -1 or idx < nb_frames) and (max_time == -1 or now - start < max_time):
        with Timer('frame generation'):
            ret, frame = camera.read()
            if not ret:
                failed += 1
                print('failure : {}'.format(frame))
            elif frames_step > 0 and idx % frames_step != 0:
                failed, idx = 0, idx + 1
            else:
                failed, idx = 0, idx + 1

                frame = frame[..., ::-1]
                if add_copy or add_index:
                    data = {'image' : frame, 'frame_index' : idx - 1}
                    if add_copy: data['image_copy'] = frame.copy()
                    yield data
                else:
                    yield frame

            now = time.time()
            if fps_time:
                wait_time = fps_time - (now - start_iter_time) - DELAY_SEC
                if wait_time > 0:
                    time.sleep(wait_time)
                    now = time.time()
            start_iter_time = now

    if isinstance(cam_id, (int, str)): camera.release()

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

@timer
def _show_frame(frame, display_name = None, fps = None, audio_file = None, stream_state = {}):
    """ Displays `fps` frames per second with `cv2.imshow` """
    import cv2
    
    if 'last_show_time' not in stream_state and audio_file:
        from ..audio import play_audio
        stream_state['player'] = play_audio(audio_file, blocking = False)
    
    if isinstance(frame, dict): frame = frame['image']

    now = time.time()
    cv2.imshow(display_name, frame[..., ::-1])

    if fps and 'last_show_time' in stream_state:
        wait_time_ms = max(
            1, int(1000 * (1. / fps - (now - stream_state['last_show_time']))) - 8
        )
    else:
        wait_time_ms = 1
    
    if cv2.waitKey(wait_time_ms) & 0xFF == ord('q'):
        stream_state['stop'] = True
        raise StopIteration()

    stream_state['last_show_time'] = time.time()

@timer
def _write_video_frame(writer, frame):
    """ Writes frames to the corresponding output file """
    if isinstance(frame, dict): frame = frame['image']
    writer.write(convert_to_uint8(frame)[:, :, ::-1])
