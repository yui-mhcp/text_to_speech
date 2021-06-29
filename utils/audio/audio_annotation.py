import os
import json
import time
import shutil
import numpy as np
import pandas as pd

from utils.plot_utils import plot, plot_embedding
from utils.generic_utils import dump_json, load_json, var_from_str, time_to_string
from utils.audio.audio_io import display_audio, read_audio, write_audio

_min_time       = 0.001
_timing_keys    = ('start', 'end', 'time')
_needed_keys    = ('id', 'start', 'end')

ANNOT_RULES     = """
Start add / modification of information. 

Here are the following option / notation to modify / add data :
- To quit : 'q'
- To go to the next part            : press Enter
- To go back to the previous part   : '!prec'
- To modify the default field       : simply enter the new value
- To modify other fields : 
    1) JSON format : '{"field1" : val1, "field2" : val2}'
    2) With ':' separator : field1:val1, field2:val2
    3) with ' = ' separator : field1 = val1, field2 = val2

Note : commas are used to separate fields. If you want to put a comma inside a value, do not put space after it, it will be automatically added (for instance 'names = a,b' becomes 'names = a, b')
"""

class AudioAnnotation(object):
    """
        Base class for audio speaker annotation on multi-speaker file
        
        It stores information on audio parts in a list of dict (`infos`) containing : 
            - start / end / time    : timing information of this specific sub-part
            - id    : speaker ID speaking at this part (either id or name)
            - ** infos  : whathever information you added (can be 'text', 'music', 'emotion', ...). 
        
        It also stores a second view : `alignment`. This is a regroupment of successibe sub-parts of `infos` which are connected (same speaker and connected in time*) and texts are contiguous (not different sentences)
        It can be useful to group multiple parts belonging to the same speaker to have complete text in a speech. 
        The regroupment can be controlled by `text_based_alignment` and `join_threshold`
        
        * i.e. the `end` of the previous sub-part is approximately equal to the `start` of the next one. 
        
        Important note : indexing / iterate over this class will iterate over `alignments` and **not** `infos`
        
        The main methods are `set_infos` (and its aliases `set_ids` and `set_text`) which allows you to listen to each (sub-part / alignment part) and modify information on it (speaker id, timing information, text, ...) or add new information
        Note that if you add an information to an "alignment part", it will be added to all its sub-part (parts from `infos` composing the alignment part). 
    """
    def __init__(self, filename, rate = None, infos = None, directory = 'results',
                 text_based_alignment = True, ids = None, alignment = None,
                 ** kwargs
                ):
        """
            Arguments : 
                - filename  : str (filename of the audio file) or ndarray (raw audio)
                - rate      : rate of the audio (only relevant if filename is raw audio)
                - infos     : list of dict with informations about each part
                    Must contains (at least) keys : {debut, fin, id}
                - directory : where to save results / config
                - text_based_alignment  : bool, whether to make alignment based on text or not.
                    If True, will combine parts if text are a continuous sentence. 
                
                - ids       : list of ids (len == N)        (normally passed when reconstructing from existing file)
                - alignment : list of dict with alignments  (normally passed when reconstructing from existing file)
        """
        self.directory  = directory
        self.filename   = filename
        self.rate       = rate
                
        self.infos      = infos
        self.text_based_alignment   = text_based_alignment
        
        self._ids       = ids
        self._alignment = alignment
        
        if ids is None:
            self.build_cluster(** kwargs)
        elif alignment is None:
            self._alignment = self.build_alignment()
            
    def _assert_valid_infos(self):
        for i, info in enumerate(self.infos):
            if not all([k in info for k in _needed_keys]):
                raise ValueError("L'information {} ne contient pas les informations nécessaires !\n  Reçu : {}".format(i, info))
    
    def _normalize_infos(self, timestep = 2.):
        if self.infos is None:
            if self._ids is None:
                rate, audio = self.load()
                self._ids = list(range(len(audio) // rate // timestep))
            self.infos = [{} for _ in range(len(self._ids))]
        elif self._ids is None:
            self._ids = [0] * len(self.infos)
        
        assert len(self._ids) == len(self.infos), "{} infos vs {} ids".format(len(self.infos), len(self._ids))
        
        for i, (info, id_i) in enumerate(zip(self.infos, self._ids)):
            info.setdefault('id', id_i)
            info.setdefault('start', i * timestep)
            info.setdefault('end', (i+1) * timestep)
            
            info['time'] = info['end'] - info['start']
            self._ids[i] = info['id']
    
    def build_cluster(self, ** kwargs):
        if self.infos is None:
            self._normalize_infos(** kwargs)
        else:
            self._assert_valid_infos()
            self._ids = [info['id'] for info in self.infos]

        self._alignment = self.build_alignment()
        return self._ids, self._alignment
        
    def build_alignment(self, join_threshold = 0.5, text_based = None):
        self._assert_valid_infos()
        if text_based is None: text_based = self.text_based_alignment
        # Make time alignment with predicted ids
        alignment = []
        start, fin, current_id, add_infos, start_idx = 0, 0, 0, {}, 0
        for i, pred_id in enumerate(self._ids):
            self.infos[i]['id'] = pred_id
            
            d, f = self.infos[i]['start'], self.infos[i]['end']
            is_new_text = True
            if len(self.infos[i].get('text', '')) > 0:
                is_new_text = self.infos[i]['text'] != add_infos.get('text', '') and self.infos[i]['text'][0].isupper()

            if pred_id == current_id and d - fin < join_threshold and join_threshold >= 0 and text_based and not is_new_text:
                fin = f
                for k, v in self.infos[i].items():
                    if k in _timing_keys: continue
                    elif k not in add_infos: add_infos[k] = v
                    elif k == 'text' and v != add_infos[k]: add_infos[k] += ' ' + v
                    elif v != add_infos[k]:
                        print("Warning : different parts should be for the same alignment but have different values for {} : {} and {}".format(k, add_infos[k], v))
                continue

            infos = {
                ** add_infos,
                'id'    : current_id,
                'start' : start,
                'end'   : fin,
                'time'  : fin - start,
                'indexes'   : list(range(start_idx, i))
            }
            if fin > 0: alignment.append(infos)

            start, fin, current_id, add_infos, start_idx = d, f, pred_id, self.infos[i].copy(), i

        last_infos = {
            ** add_infos,
            'id'    : current_id,
            'start' : start,
            'end'   : fin,
            'time'  : fin - start,
            'indexes'   : list(range(start_idx, len(self._ids)))
        }
        alignment.append(last_infos)
        
        return alignment
            
    @property
    def basename(self):
        return os.path.basename(self.filename).split('.')[0]
        
    @property
    def total_time(self):
        return sum([info['time'] for info in self.infos])
    
    @property
    def ids(self):
        return np.array(self._ids)
    
    @property
    def alignment(self):
        return pd.DataFrame(self._alignment)
    
    @property
    def speakers(self):
        return sorted(np.unique(self.ids))
    
    @property
    def speaker_infos(self):
        """
            Return all speakers with informations
            Return : infos : dict {speaker_id : {
                alignment   : list of dict {debut:, fin:, id:, duree:, ...}
                indexes     : list of indexes such as self.infos[i]['id'] == speaker_id
            }}
        """
        return {speaker_id : {
            'alignments' : self.get_speaker_alignment(speaker_id),
            'indexes'   : np.where(self.ids == speaker_id)
        } for speaker_id in self.speakers}
    
    def __len__(self):
        return len(self._alignment)
    
    def __str__(self):
        des = "Annotation of file {} :\n".format(self.filename)
        des += "- Total annotation time : {}\n".format(
            time_to_string(self.total_time)
        )
        des += "- Number of alignments : {} ({} sub-parts)\n".format(
            len(self._alignment), len(self.infos)
        )
        des += "- Speakers (n = {}) : {}\n".format(len(self.speakers), self.speakers)
        return des
    
    def __getitem__(self, idx):
        return self._alignment[idx]
    
    def __contains__(self, name):
        return name in self._ids
    
    def load(self, rate = None):
        """ Load the audio file at a given rate """
        if isinstance(self.filename, np.ndarray):
            assert self.rate is not None or rate is not None
            if rate is None: rate = self.rate
            audio = self.filename
            if self.rate and rate != self.rate:
                ratio = rate / self.rate
                audio = resample(audio, int(len(audio) * ratio))
            if self.rate is None: self.rate = rate
            return rate, audio
        return read_audio(self.filename, target_rate = rate)
    
    def describe(self):
        return self.alignment.describe()
    
    def _update_infos(self, idx, ** kwargs):
        """
            Update information at index 'idx' with 'kwargs' new values
            Note : 'start' / 'end' keys are treated as relative values so they will be added / removed from the original value. 
            It means 'start = 1' will add 1sec to the 'start' field and will *not* make 'start == 1' (except if 'start' was 0 :p)
        """
        assert idx >= 0 and idx < len(self.infos)
        
        time_modified = 'end' in kwargs or 'start' in kwargs
        
        if 'start' in kwargs and isinstance(kwargs['start'], int):
            kwargs['start'] /= self.rate
            
        if 'end' in kwargs and isinstance(kwargs['end'], int):
            kwargs['end'] /= self.rate
        
        if kwargs.get('end', 0) > 1:
            print("As a security, we disallow to modify time for more than 1sec")
            kwargs['end'] = 1.
        
        if kwargs.get('start', 0) > 1:
            print("As a security, we disallow to modify time for more than 1sec")
            kwargs['start'] = 1.
        
        self._ids[idx]  = kwargs.get('id', self._ids[idx])
        kwargs['start'] = self.infos[idx]['start'] + kwargs.get('start', 0.)
        kwargs['end']   = self.infos[idx]['end'] + kwargs.get('end', 0.)
        kwargs['time'] = kwargs['end'] - kwargs['start']
        
        if kwargs['time'] <= 0.:
            kwargs['end'] = kwargs['start'] + _min_time
            kwargs['time'] = _min_time
        
        #if idx > 0 and kwargs['start'] < self.infos[idx-1]['end']:
        #    self._update_infos(idx-1, fin = kwargs['start'])
        
        #if idx < len(self.infos)-1 and kwargs['end'] > self.infos[idx+1]['start']:
        #    self._update_infos(idx+1, debut = kwargs['end'])
        
        self.infos[idx].update(kwargs)

        return time_modified
    
    def set_infos(self,
                  default,
                  default_value = None,
                  
                  rate      = None,
                  transform_fn  = None,
                  
                  ids       = None,
                  filter_fn = None,

                  by_part   = False,
                  start_idx = 0,
                  replay_if_time_change = True,
                  
                  play      = True,
                  play_time = None,
                  play_original = False,
                  display   = False,
                  ** kwargs
                 ):
        """
            Start an annotation procedure that will show all desired parts and allow to add / modify information about them. 
            
            Arguments :
                - default   : the default field to modify if no key are given
                - default_value : default value to put for 'default' field if no value was given and this field is not already present
                
                - rate      : the rate to load the audio. Mostlly relevant if the 'transform_fn' needs a specific rate
                - transform_fn  : function to apply on all part before playing it
                
                - ids       : the ids for which to annotate (other will be skipped). 
                - filter_fn : callable that takes the current 'part' as argument and return False to skip it or True to display (and annotate) it. 
                
                - by_part   : whether to show alignments (False) or individual parts (True)
                - start_idx : the initial index to start annotation
                - replay_if_time_change : whether to replay the part if you modify 'start' / 'end' fields. 
                
                - play      : whether to autoplay the audio
                - play_time : maximal audio time to play for a 'part'
                - play_original : whether to display original audio (before transform_fn) or not. Relevant only if 'transform_fn' is not None
                - display   : whether to plot the audio or not
                - kwargs    : kwargs passed to the plot() function
        """
        if transform_fn is None: play_original = False
        if ids is not None and not isinstance(ids, (list, tuple)): ids = [ids]
        rate, audio = self.load(rate = rate)
        
        print(ANNOT_RULES)
        print("Default modified field is : {}".format(default))
        
        data = self.infos if by_part else self._alignment
        
        t0 = time.time()
        i, prec = start_idx, []
        
        while i < len(data):
            if ids is not None and data[i]['id'] not in ids:
                i += 1
                continue
            debut, fin = data[i]['start'], data[i]['end']
            
            if filter_fn and not filter_fn(data[i]):
                i += 1
                continue
            
            print('\n\nPart {} / {} :\n'.format(i, len(data)))
            start, end = int(debut * rate), int(fin * rate)
            
            audio_i = audio[start : end]
            
            if play_original:
                display_audio(audio_i, rate = rate, temps = play_time, play = False)
                
            if transform_fn: audio_i = transform_fn(audio_i, rate)
            
            display_audio(audio_i, rate = rate, temps = play_time, play = play)
            
            if display:
                plot(audio_i, ** kwargs)
            
            print("\n    id : {} - debut : {} - fin : {} - infos :".format(
                data[i]['id'], 
                time_to_string(debut), 
                time_to_string(fin)
            ))
            for k, v in data[i].items():
                if k == 'indexes' or k in _needed_keys: continue
                print("- {}\t: {}".format(k, v))
            print()
            
            if default in data[i]:
                print("Current value : {}".format(data[i][default]))
            
            new_infos = input("\n\nEnter new value :")
            if len(new_infos) == 0 and default is not None and default not in data[i]:
                new_infos = str(default_value)
            if len(new_infos) == 0:
                prec.append(i)
                i += 1
                continue
            elif new_infos == '!prec':
                i = prec.pop()
                continue
            elif new_infos == 'q': break
                        
            if '{' in new_infos and '}' in new_infos:
                infos = json.loads(new_infos)
            if ':' in new_infos or ' = ' in new_infos:
                sep = ' = ' if ' = ' in new_infos else ':'
                infos = {}
                for key_value_text in new_infos.split(', '):
                    if sep in key_value_text:
                        key = key_value_text.split(sep)[0]
                        value = sep.join(key_value_text.split(sep)[1:])
                    else: key, value = default, key_value_text

                    infos[key] = var_from_str(value)
            else:
                infos = {default : var_from_str(new_infos)}
            
            for k, v in infos.items():
                if isinstance(v, str) and ',' in v and ', ' not in v:
                    infos[k] = v.replace(',', ', ')
            print("New infos : {}".format(infos))
            if by_part:
                time_modified = self._update_infos(i, ** infos)
            else:
                time_modified = 'start' in infos or 'end' in infos
                # modified end timing only for the last information
                if infos.get('end', None):
                    last_idx = data[i]['indexes'][-1]
                    data[i]['end'] += infos['end']
                    self._update_infos(last_idx, fin = infos.pop('end'))
                    
                if infos.get('start', None):
                    first_idx = data[i]['indexes'][0]
                    data[i]['start'] += infos['start']
                    self._update_infos(first_idx, debut = infos.pop('start'))
                infos.pop('time', None)
                
                for idx in data[i]['indexes']:
                    self._update_infos(idx, ** infos)
            
            self.save()
            if not time_modified or not replay_if_time_change:
                prec.append(i)
                i += 1
        
        self._alignment = self.build_alignment()
        self.save()
        print("Modifications saved !")
        print("Annotation took {} !".format(time_to_string(time.time() - t0)))
        
    def set_ids(self, ** kwargs):
        """ Allias for 'set_infos' with default field as 'id' """
        kwargs['default'] = 'id'
        self.set_infos(** kwargs)
        
    def set_text(self, ** kwargs):
        """ Allias for 'set_infos' with default field as 'text' """
        kwargs['default'] = 'text'
        self.set_infos(** kwargs)
        
    def set_names(self):
        """
            Allow to modify the id of a given speaker by having a couple of audio from it 
        """
        for speaker_id in self.speakers:
            print("Audio samples for speaker '{}' (total {}) :\n\n".format(
                speaker_id, sum([1 for info in self.infos if info['id'] == speaker_id])
            ))
            self.display(speaker_id, verbose = False, max_display = 5)
            
            new_id = input("\n\nEnter new id for this speaker (q to quit) :")
            if len(new_id) == 0: continue
            elif new_id == 'q': break
            
            self.rename(speaker_id, new_id)
            print()
        print("Renaiming finished !")
    
    def rename(self, old_id, new_id):
        """ Set new id for the given speaker """
        self.set_speaker_infos(old_id, id = new_id)
        
    def set_speaker_infos(self, name, ** kwargs):
        """
            Set new information on all parts of a given speaker
            Can be useful to set general information about a speaker such as its 'sex'
        """
        for k, v in kwargs.items():
            if k in _timing_keys: continue
            
            for i in range(len(self.infos)): 
                if self.infos[i]['id'] == name:
                    self.infos[i][k] = v
                    if k == 'id': self._ids[i] = v
            
            for i in range(len(self._alignment)):
                if self._alignment[i]['id'] == name: self._alignment[i][k] = v
        
        self.save()
    
    def remove_speaker(self, name):
        """ Remove all parts of a given speaker """
        self._ids = [i for i in self._ids if i != name]
        self.infos = [info for info in self.infos if info['id'] != name]
        self._alignment = self.build_alignment()
        
    def get_speaker_alignment(self, name):
        """
            Return all parts of the speaker
            Return : dict {debut : [], fin : [], duree : []}
        """
        return [a.copy() for a in self._alignment if a['id'] == name]
    
    def get_speaker_infos(self, names):
        """
            Return self.speaker_infos only for ids in names
        """
        if not isinstance(names, (list, tuple)): names = [names]
        return {n : infos for n, infos in self.speaker_infos.items() if n in names}
    
    def get_speaker_audios(self, name, rate = None):
        """
            Return list of all audio samples for the given speaker
            Return a tuple (rate, list_audio_samples)
        """
        rate, audio = self.load(rate)
        
        return rate, [
            audio[int(info['start'] * rate) : int(info['end'] * rate)] 
            for info in self.get_speaker_alignment(name)
        ]
        
    def display(self, name = None, idx = None, max_display = None,
                by_part = False, verbose = 2):
        """
            Display audios parts
            Arguments :
                - max_display   : maximum samples to display
                - name          : id or list of speaker's ids to display
                - by_part       : whether to display alignments (False) or individual parts (True)
                - verbose       : verbosity level (0, 1, 2)
        """
        if name is not None and not isinstance(name, (list, tuple)): name = [name]
        if idx is not None and not isinstance(nidxaidxme, (list, tuple, np.ndarray)):
            idx = [idx]
        
        if verbose and name is not None:
            print("Audio samples for speaker(s) {} :\n".format(name))
            
        rate, audio = self.load()
        data = self.infos if by_part else self._alignment
        
        nb_display = 0
        for i, infos in enumerate(data):
            if max_display and nb_display >= max_display: break
            if name is not None and infos['id'] not in name: continue
            if idx is not None and i not in idx: continue
            
            if verbose:
                print("\n\nID : {} - début : {} - fin : {}{}\n".format(
                    time_to_string(infos['start']), 
                    time_to_string(infos['end']), 
                    infos['id'], 
                    ' - bonus infos : {}'.format({k : v for k, v in infos.items() if k not in _needed_keys}) if verbose == 2 else ''
                ))
            display_audio(
                audio, 
                rate  = rate, 
                debut = infos['start'], 
                fin   = infos['end']
            )
            nb_display += 1
        
    def get_config(self):
        return {
            'filename'  : self.filename,
            'rate'      : self.rate,
            'infos'     : self.infos,
            'text_based_alignment'  : self.text_based_alignment
        }
        
    def save(self, directory = None, with_alignment = True):
        assert directory or self.directory
        if not directory: directory = self.directory
        if self.directory is None: self.directory = directory
        
        os.makedirs(directory, exist_ok = True)
        print("Saving to directory {}".format(directory))
        self.save_config(directory, with_alignment)
    
    def save_config(self, directory, with_alignment = True):
        config_file = 'config.json'
        if directory is not None:
            config_file = os.path.join(directory, config_file)
        
        data = self.get_config()
        if with_alignment:
            data['ids']     = self._ids
            data['alignment']   = self._alignment
        print("Saving data to {}".format(config_file))
        dump_json(config_file, data, indent = 4)
    
    def save_wavs(self, directory, map_file = 'map.json', rate = None,
                  by_part = False, overwrite = False):
        """
            Save all audio samples in subdirectories (1 for each speaker)
            
            Arguments :
                - directory : main directory in which create sub-directories
                - map_file  : filename for mapping file (see below for details)
                - rate      : the rate to save audios
                - by_part   : whether to save individual parts audios (True) or alignments audios (False)
                - overwrite : whether to overwrite if audios already exists
            Return :
                - map_infos : list of dict containing mapping between audio_filename and id and all other information for it
                
            The mapping information contains, at least, fields : 
                - 'original_filename'   : the original filename from which this sample comesfrom
                - 'filename'    : the audio_filename for this sample
                - 'id'  : speaker id
                - 'start' / 'end' / 'time' : timing information from original file
            
            Audios will be saved as follow : 
            directory/
                speaker_1/
                    audio_0.wav
                    audio_1.wav
                    ...
                speaker_2/
                    audio_0.wav
                ...
        """
        wav_dir = os.path.join(directory, 'wavs')
        data    = self.infos if by_part else self._alignment
        
        new_infos = []
        audio = None
        if not os.path.exists(wav_dir) or overwrite:
            rate, audio = self.load(rate = rate)
        
        if os.path.exists(wav_dir) and overwrite: shutil.rmtree(wav_dir)
        os.makedirs(wav_dir, exist_ok = True)
        
        spk_nb = {}
        for i, info in enumerate(data):
            if info['id'] == '?' or info['time'] < 0.1: continue
            speaker_dir = os.path.join(wav_dir, str(info['id']))
            os.makedirs(speaker_dir, exist_ok = True)
            
            spk_nb.setdefault(info['id'], 0)
            audio_name = os.path.join(
                speaker_dir, 
                'audio_{}.wav'.format(spk_nb[info['id']])
            )
            spk_nb[info['id']] += 1

            if not os.path.exists(audio_name):
                d, f = int(rate * info['start']), int(rate * info['end'])
                write_audio(audio[d : f], audio_name, rate = rate)
            
            new_infos.append({
                'original_filename' : self.filename.replace(os.path.sep, '/'),
                'filename'  : audio_name.replace(os.path.sep, '/'),
                ** info
            })
        
        dump_json(os.path.join(directory, map_file), new_infos, indent = 4)
        
        return new_infos
            
    def save_as_dataset(self, path, overwrite = False, ** kwargs):
        """
            Save all datas (alignment and / or sub-parts) to a dataset format
            
            Dataset format is a sub directory (with os.basename(self.filename)) as name
            This directory contains : 
                - parts/ and alignments/ : sub-dir for single part of alignments
                    - wavs/     : contains directories with speakers audios
                    - map.json  : map between wavs and information
            
            The map.json file is structured as follow : 
            list of dict where each element correspond to a specific wav
            [
                {
                    "original_filename" : wav filename of the complete audio
                    "filename"  : wav filename for this part
                    "id"        : speaker's ID
                    "debut"     : start time of the audio part (in the complete audio)
                    "fin"       : end time of the audio part (in the complete audio)
                    "duree"     : time of the audio file of this part
                    ...         : additionnal informations given during annotation
                },
                ...
            ]
            
            Warning : the 'text' can be wrong for the sub-parts sub-parts of an alignment can contain the complete text for the whole alignment (if it was annotated in alignment mode). 
        """
        self.save_wavs(
            os.path.join(path, self.basename, 'parts'), by_part = True, 
            overwrite = overwrite, ** kwargs
        )
        self.save_wavs(
            os.path.join(path, self.basename, 'alignments'), by_part = False, 
            overwrite = overwrite, ** kwargs
        )
        
        return os.path.join(path, self.basename)

    
    @classmethod
    def load_from_file(cls, directory):
        """ Load from directory or config file """
        if not os.path.exists(directory):
            raise ValueError("Le répertoire {} n'existe pas !".format(directory))
        
        if directory.endswith('.json'):
            config_file = directory
            directory = directory[:-5]
        else:
            config_file = os.path.join(directory, 'config.json')
        
        config = load_json(config_file, default = {})
        
        return cls(directory = directory, ** config)

def load_annotation_dir(directory):
    """ Load a list of AudioAnnotation from a given directory """
    results = []
    files = [
        os.path.join(directory, f) for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f)) or f.endswith('.json')
    ]
    for file in files:
        try:
            results.append(AudioAnnotation.load_from_file(file))
        except:
            pass
    return results

def embed_annotation_dataset(directory, embed_fn, embedding_dim, rate, ** kwargs):
    """
        This method is used in main project for embedding the audio samples (such as for SV2TTS) 
    """
    from utils.embeddings import embed_dataset
    from datasets.custom_datasets.audio_datasets import preprocess_identification_annots

    for identification_dir in os.listdir(directory):
        print("\nStart embeddings for {}...".format(
            identification_dir
        ))
        for mode in ['parts', 'alignments']:            
            ds = preprocess_identification_annots(
                os.path.join(directory, identification_dir), by_part = mode == 'parts'
            )
            
            embed_dataset(
                os.path.join(directory, identification_dir, mode), ds, embed_fn,
                embedding_dim = embedding_dim, rate = rate, ** kwargs
            )
