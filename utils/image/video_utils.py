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
import logging
import subprocess
import numpy as np

from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class VideoInfos:
    fps : float
    nb_frames   : int
    frame_w     : int
    frame_h     : int

def get_video_infos(filename):
    """ Returns a `VideoInfos` dataclass with the video's information """
    cap = cv2.VideoCapture(filename)
    
    infos   = VideoInfos(
        fps = float(cap.get(cv2.CAP_PROP_FPS)),
        nb_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        frame_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        frame_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    )

    cap.release()
    return infos
    
def write_video(images, filename, fps = 16, show = False):
    """ Save a list of images as a video to `filename` """
    h, w, c = images[0].shape
    video = cv2.VideoWriter(filename, 0, fps, (w,h))
    for image in images:
        if image.dtype in (float, np.floating):
            image = (image * 255).astype(np.uint8)
        video.write(image)
        if show:
            cv2.imshow(filename, image)
            cv2.waitKey(1)
    if show: cv2.destroyAllWindows()
    video.release()
    
def load_youtube_playlist(url, directory = 'youtube_playlist', 
                          separate_audio = False, only_audio = False, ** kwargs):
    try:
        from pytube import Playlist
    except ImportError:
        logger.error("You must install pytube : `pip install pytube3`")
        return None
    os.makedirs(directory, exist_ok = True)
    
    playlist = Playlist(url)
    
    playlist.download_all(** kwargs)
    
    if only_audio or separate_audio:
        for video in os.listdir(directory):
            extract_audio(video)
            
            if only_audio: os.remove(video)
    
    return os.listdir(directory)

def load_youtube_video(url, filename = 'youtube.mp4', resolution = 'middle', 
                       separate_audio = False, only_audio = False):
    assert resolution in ('lowest', 'middle', 'highest')
    try:
        from pytube import YouTube
    except ImportError:
        logger.error("You must install pytube : `pip install pytube3`")
        return None

    video = YouTube(url)

    streams = video.streams.filter(file_extension = 'mp4', 
                                   progressive = True).order_by('resolution')
    
    if resolution == 'lowest' or only_audio:
        stream = streams[0]
    elif resolution == 'middle':
        stream = streams[len(streams) // 2]
    elif resolution == 'highest':
        stream = streams[-1]
    
    logger.info('Downloading stream : {}'.format(stream))
    
    video_filename = 'tmp_video.mp4' if only_audio else  filename
    if not video_filename.endswith('.mp4'): video_filename += '.mp4'
    video_filename = stream.download(filename = video_filename)
    
    if only_audio or separate_audio:
        exctract_audio(video_filename, filename)
    else:
        filename = video_filename
    
    if only_audio:
        os.remove(video_filename)
        
    return filename

def copy_audio(video_source, video_target):
    """ Copy audio from one video (`video_source`) to another (`video_target`) """
    audio_file  = extract_audio(video_source, filename = 'audio_tmp.mp3')
    res         = set_video_audio(audio_file, video_target)
    os.remove(audio_file)
    return res

def set_video_audio(audio_filename, video_filename):
    """ Set the audio in `audio_filename` to `video_filename` (with `ffmpeg`) """
    return subprocess.run([
        'ffmpeg', '-i', video_filename, '-i', audio_filename, '-c', 'copy', '-map', '0:v:0',
        '-map', '1:a:0', video_filename[:-4] + '_audio.mp4'
    ])

    
def extract_audio(video_filename, filename = None):
    """
        Extract audio from `video_filename` and copy it to `filename` (require `moviepy` library)
        Note that if you want the audio np.ndarray, you can use the `read_audio` from `utils.audio` instead
    """
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        logger.error("You must install moviepy : `pip install moviepy`")
        return None

    if filename is None: filename = '{}_audio.mp3'.format(video_filename[:-4])
    if not filename.endswith('.mp3'): filename += '.mp3'

    with VideoFileClip(video_filename) as video:
        video.audio.write_audiofile(filename)
    
    return filename
