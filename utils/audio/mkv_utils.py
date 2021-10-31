import os
import json
import subprocess

from utils.generic_utils import load_json, dump_json

def process_mkv(path, audio_stream = 1, subs_stream = -1, 
                output_dir = None, audio_filename = None, subs_filename = None, 
                map_file = None, verbose = True, ** kwargs):
    """
        Process .mkv file (or dir) by extracting audio and subtitles
        The function use ffmpeg to extract audio / subs so ffmpeg must be available
        
        Arguments : 
            - path  : path (or list of path) of the .mkv file (or dir of mkv files)
            - audio_stream  : audio stream to extract (default = 1)
            - subs_stream   : subtitles stream to extract (default = 2)
            - audio_filename    : output filename for audio (default = None)
            - subs_stream       : output filename for subtitles (.srt) (default = None)
            - output_dir        : output directory (default = directory of 'path')
            - map_file      : json file to save informations (and text alignment)
            - verbose       : verbosity
            - kwargs        : passed to the call to parse_subtitles(...)
        Returns : infos (dict) with keys : 
            {original_filename, audio_filename, subs_filename, alignment}
        
        Note : default filenames (if None) are : 
            path.replace('.mkv', '_audio.mp3')  for audio_filename
            path.replace('.mkv', '_subs.srt')   for subs_filename
    """
    if os.path.isdir(path) or isinstance(path, (list, tuple)):
        files = os.listdir(path) if not isinstance(path, (list, tuple)) else path
        files = [f for f in files if f.endswith('.mkv')]
        if verbose: print("Processing list of {} files...".format(len(files)))
        
        return [process_mkv(
            os.path.join(path, f), 
            audio_stream    = audio_stream, 
            subs_stream     = subs_stream, 
            output_dir      = output_dir,
            verbose = verbose
        ) for f in files]
    
    if output_dir is None: output_dir = os.path.dirname(path)
    if map_file is None: map_file = os.path.join(output_dir, 'map.json')
    
    
    audio_filename = extract_audio(
        path,
        stream      = audio_stream,
        output_dir  = output_dir,
        output_file = audio_filename,
        verbose     = verbose
    )
    
    subs_filename = extract_subtitles(
        path,
        stream      = subs_stream,
        output_dir  = output_dir,
        output_file = subs_filename,
        verbose     = verbose
    )
    
    alignment = parse_subtitles(subs_filename, ** kwargs)
    
    infos = {
        'original_filename' : path,
        'audio_filename'    : audio_filename,
        'subs_filename'     : subs_filename,
        'alignment'         : alignment
    }
    
    data = load_json(map_file, default = {})
    
    data[path] = infos
    
    dump_json(map_file, data, indent = 4)
    
    return infos

def extract_audio(path, output_dir = None, output_file = None, ** kwargs):
    if output_file is None:
        if output_dir is None: output_dir = os.path.dirname(path)
        basename = os.path.basename(path)
        
        output_file = os.path.join(
            output_dir, basename.replace('.mkv', '_audio.mp3')
        )
    
    return _extract(path, output_file, mode = 'a', ** kwargs)

def extract_subtitles(path, output_dir = None, output_file = None, ** kwargs):
    if output_file is None:
        if output_dir is None: output_dir = os.path.dirname(path)
        basename = os.path.basename(path)

        output_file = os.path.join(
            output_dir, basename.replace('.mkv', '_subs.srt')
        )
    elif not output_file.endswith('.srt'):
        output_file += '.srt'
    
    return _extract(path, output_file, mode = 's', ** kwargs)

        
def parse_subtitles(path, join_threshold = 0., add_time = 0.5):
    """
        Process a .srt file to extract all text alignment
        
        Arguments : 
            - path : path to the .rst file
            - join_threshold    : seconds between 2 subtitles to concat them
            - add_time  : nb of seconds to add after and before the specified time
        Returns : alignment (list of dict), each dict has keys : 
            {text, debut, fin, temps}
        
        Exemple (for join_threshold) : 
            2 subtitles : 
            1) debut = 0.25, fin = 0.30
            2) debut = 0.30.5, fin = 0.40
            The 2 subtitles are really closed so we can suppose they are from same speaker
            If join_threshold > 0.5, then the 2 subtitles will be treated as one unique subtitle by concatenating their respective text
    """
    def get_time(str_time):
        h, m, s = [float(t.replace(',', '.')) for t in str_time.split(':')]
        return h * 3600 + m * 60 + s
    
    if isinstance(path, (list, tuple)):
        alignments = []
        for p in path:
            parsed = parse_subtitles(p, join_threshold, add_time)
            alignments.extend([part for part in parsed if part not in alignments])
        return sorted(alignments, key = lambda t: t['start'])
    
    if not os.path.exists(path): return None
    
    with open(path, 'r', encoding = 'utf-8') as file:
        lines = file.read().split('\n')
    
    infos = []
    status, text, debut, fin = 0, [], 0., 0.
    for l in lines:
        if len(l) == 0 or l.isdigit():
            status = 0
            continue
        
        status += 1
        if status == 1:
            d, f = [get_time(t) for t in l.split(' --> ')]
            
            if join_threshold <= 0 or abs(fin - d) > join_threshold:
                if len(text) > 0:
                    fin = min(fin + add_time, d)
                    infos.append({'text' : ' '.join(text), 'start' : debut, 'end' : fin, 'time' : fin - debut})
                text, debut, fin = [], max(fin, d - add_time), 0.
            fin = f
        else:
            text.append(l)
    
    fin += add_time
    infos.append({'text' : ' '.join(text), 'start' : debut, 'end' : fin, 'time' : fin - debut})
    
    return infos

def _extract(path, output_file, mode, stream = 1, verbose = True, overwrite = False):
    if stream == -1:
        if '{}' not in output_file:
            output_file, ext = os.path.splitext(output_file)
            output_file = output_file + '_{}' + ext

        result = []
        stream = 0
        while stream < 5:
            output_i = _extract(path, output_file.format(stream), mode, stream, verbose, overwrite)
            if output_i is None: break
            result.append(output_i)
            stream += 1
        return result
    
    assert mode in ('a', 's')
    long_mode = 'audio' if mode == 'a' else 'subtitles'
    
    if os.path.exists(output_file):
        if not overwrite:
            if verbose: print("File {} already exists !".format(output_file))
            return output_file
        
        os.remove(output_file)
    
    if verbose:
        print("Extraction of {} (stream #{})...".format(long_mode, stream))
        
    c = subprocess.run(
        ['ffmpeg', '-i', path, '-map', '0:{}:{}'.format(mode, stream), output_file]
    ).returncode
    
    if verbose:
        if c == 0: print("{} successfully extracted !".format(long_mode))
        else: print("Error (code : {})".format(c))
        
    return output_file if c == 0 else None
