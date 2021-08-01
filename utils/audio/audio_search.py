import os
import datetime
import itertools
import numpy as np

from utils.plot_utils import plot
from utils.generic_utils import time_to_string
from utils.audio.audio_io import display_audio
from utils.distance.distance_method import distance
from utils.audio.audio_annotation import AudioAnnotation

_zero_time = datetime.datetime(2000, 1, 1)

class AudioSearch(AudioAnnotation):
    def __init__(self, keyword, * args, distance_fn = None, threshold = 0.8, ** kwargs):
        super().__init__(* args, ** kwargs)
        
        self.keyword        = keyword
        self.threshold      = threshold
        self.distance_fn    = distance_fn
        
        self.probabilities  = None
        self.errors         = None
        self.scores         = None
        
        self.__search()
    
    def __search(self):
        self.probabilities  = np.zeros((len(self._alignment),))
        self.errors     = []
        self.scores     = []
        
        for i, info in enumerate(self._alignment):
            if 'text' not in info or len(info['text']) == 0:
                self.errors.append([])
                self.scores.append([])
                continue

            dist, matrix = self.distance(info['text'])

            self.probabilities[i] = 1. - dist
            self.errors.append(matrix[-1, 1:])
            self.scores.append(1. - matrix[-1, 1:] / len(self.keyword))
    
    @property
    def index(self):
        return np.where(self.probabilities > self.threshold)[0]
    
    @property
    def timestamps(self):
        timestamps = []
        for idx in self.index:
            info    = self._alignment[idx]
            scores  = self.scores[idx]
            errors  = self.errors[idx]
            
            indexes = []
            current = 0
            for v, vals in itertools.groupby(scores > self.threshold):
                n = len(list(vals))
                if v: indexes.append(current + np.argmax(scores[current : current + n]))
                current += n
            indexes = np.array(indexes)
            start_indexes = np.maximum(0, indexes - len(self.keyword) - errors[indexes])
            
            prop    = start_indexes / len(scores)
            
            for start_idx, end_idx, p in zip(start_indexes, indexes, prop):
                timestamps.append({
                    'text'  : info['text'][int(start_idx) : end_idx + len(self.keyword)],
                    'start' : info['start'] + info['time'] * p,
                    'probability'   : scores[end_idx],
                })
        return timestamps
    
    @property
    def nb_occurences(self):
        return len(self.timestamps)
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        return super().__getitem__(self.index[idx])
    
    def __str__(self):
        des = super().__str__()
        
        timestamps = self.timestamps
        if len(timestamps) == 0:
            des += "No occurence of '{}' found\n".format(self.keyword)
        else:
            des += "\nOccurences of '{}' ({}, threshold = {:.2f}%) :\n".format(
                self.keyword, len(timestamps), self.threshold * 100
            )
            for timestamp in timestamps:
                des += "- Timestamp {} (p = {:.2f} %) : [...] {} [...]\n".format(
                    time_to_string(timestamp['start']),
                    timestamp['probability'] * 100,
                    timestamp['text']
                )
        
        return des
    
    def __lt__(self, value):
        return len(self) < len(value) or self.filename < value.filename
    
    def set_threshold(self, threshold):
        self.threshold = threshold
        self.__search()
    
    def search(self, keyword, threshold = None):
        if threshold is not None: self.threshold = threshold
        self.keyword = keyword
        self.__search()
    
    def distance(self, text):
        kwargs = {
            'hypothesis' : self.keyword, 'truth' : text, 'method' : 'edit', 
            'partial' : True, 'normalize' : True, 'return_matrix' : True
        }
        if self.distance_fn is not None:
            return self.distance_fn(** kwargs)
        return distance(** kwargs)
    
    def max(self):
        return np.max(self.probabilities)
    
    def min(self):
        return np.min(self.probabilities)
    
    def contains(self):
        return len(self.index) > 0
    
    def plot(self, by_alignment = True, ** kwargs):
        plot(
            self.probabilities if by_alignment else np.concatenate(self.scores),
            ylim = (0, 1), title = "Probability of {} for each alignment".format(self.keyword),
            xlabel = 'Alignment index', ylabel = "Probability (%)", ** kwargs
        )
    
    def display(self, before = 2.5, display_time = 10, max_display = None, verbose = 2):
        rate, audio = self.load()
        
        nb_display = 0
        for i, timestamp in enumerate(self.timestamps):
            display_audio(
                audio, 
                rate    = rate, 
                debut   = timestamp['start'] - before,
                temps   = display_time
            )
            nb_display += 1






class SearchResult(object):
    def __init__(self, * results):
        self.__results  = results
    
    @property
    def results(self):
        return self.__results
    
    @property
    def keyword(self):
        return self[0].keyword
    
    @property
    def filenames(self):
        return [result.filename for result in self]
    
    @property
    def nb_contains(self):
        return len(self.containing_files)
    
    @property
    def nb_occurences(self):
        return sum([result.nb_occurences for result in self])
    
    @property
    def containing_files(self):
        return [res for res in self if res.contains()]
    
    def __len__(self):
        return len(self.results)
    
    def __str__(self):
        des = "Result for searching keyword '{}' :\n".format(self.keyword)
        des += "Number of files : {} / {}\n".format(self.nb_contains, len(self))
        des += "Total number of occurences : {}\n".format(self.nb_occurences)
        des += "Files : {}".format('\n\n'.join([str(f) for f in self.containing_files]))
        return des
        
    def __getitem__(self, idx):
        return self.results[idx]
    
    def remove_empty(self):
        removed = [result for result in self if not result.contains()]
        self.results = [result for result in self if result.contains()]
        return removed
    
    def set_threshold(self, * args, ** kwargs):
        for res in self: res.set_threshold(* args, ** kwargs)

    def search(self, * args, ** kwargs):
        for res in self: res.search(* args, ** kwargs)
    
    def plot(self, * args, ** kwargs):
        for res in self: res.plot(*args, **kwargs)

    def display(self, * args, ** kwargs):
        for result in self:
            result.display(*args, **kwargs)
            print('\n')
            
