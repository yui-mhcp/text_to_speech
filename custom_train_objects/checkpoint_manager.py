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
import copy
import logging
import warnings

from functools import cache

from utils import load_json, dump_json

logger = logging.getLogger(__name__)

_default_ckpt_format    = 'ckpt.weights.h5'
_empty_checkpoint_infos = {
    'counter' : 0, 'loaded' : -1, 'checkpoints' : [], 'best_checkpoint' : {}
}

class CheckpointManager:
    """ This class manages the checkpoint for all the tracked models """
    def __init__(self,
                 core   = None,
                 directory  = None,
                 max_to_keep    = 3,
                 checkpoint_format  = _default_ckpt_format
                ):
        """
            Initializes a checkpoint manager
            
            Arguments :
                - core  : any data structure (typically a `BaseModel` instance)
                    The object must have the following attributes :
                        - save_dir  : the saving directory
                        - epochs    : the current number of training epochs
                        - steps     : the current number of training steps
                - max_to_keep   : the number of checkpoint files to keep (older are removed)
                - checkpoint_format : checkpoint filename format
                    The file format is formatted with the following keys :
                        - epochs    : the number of training epochs
                        - steps     : the number of training steps
                        - counter   : the number of already saved checkpoints
                        - model     : the model name (the `key` provided in the `add` call)
        """
        self.core   = core
        self._directory = directory
        self.max_to_keep    = max_to_keep
        self.checkpoint_format  = checkpoint_format
        
        self._state = load_json(
            self.checkpoint_file, default = copy.deepcopy(_empty_checkpoint_infos)
        )
        self.model  = None
    
    step    = property(lambda self: self.core.steps if self.core is not None else -1)
    epoch   = property(lambda self: self.core.epochs if self.core is not None else -1)
    directory   = property(
        lambda self: self.core.save_dir if self.core is not None else self._directory
    )
    best_checkpoint_path    = property(
        lambda self: os.path.join(self.directory, 'best.weights.h5')
    )
    
    counter = property(lambda self: self._state['counter'])
    loaded  = property(lambda self: self._state['loaded'])
    checkpoints = property(lambda self: self._state['checkpoints'])
    latest_checkpoint_infos = property(lambda self: self[-1])
    
    @property
    def checkpoint_file(self):
        return os.path.join(self.directory, 'checkpoint.json')

    @property
    def infos(self):
        return {'epoch' : self.epoch, 'step' : self.step, 'counter' : self.counter}
    
    @property
    def latest_checkpoint(self):
        return None if len(self) == 0 else self.get_filename(self[-1])
    
    @property
    def loaded_checkpoint(self):
        if self.loaded == 'best': return self.best_checkpoint
        return None if len(self) == 0 else self.get_filename(self[self.loaded])
    
    @property
    def best_checkpoint(self):
        if os.path.exists(self.best_checkpoint_path):
            return self.best_checkpoint_path
        return self.latest_checkpoint
    
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        if not self.checkpoints:
            return '<CheckpointManager directory={} no checkpoint created>'.format(self.directory)
        loaded = self.loaded
        if isinstance(loaded, int):
            loaded, infos = loaded + 1, self[loaded]
        elif loaded == 'best':
            infos = self._state['best_checkpoint'] if self._state['best_checkpoint'] else self[-1]
        
        return '<CheckpointManager directory={} loaded={} ({} saved) (epoch {})>'.format(
            self.directory, loaded, len(self), infos['epoch']
        )
    
    def __len__(self):
        """ Return the number of tracked checkpoints """
        return len(self.checkpoints)
    
    def __getitem__(self, idx):
        """ Return the information of the checkpoint at the given index """
        return self.checkpoints[idx]
    
    def __update_state(self):
        if len(self) == self.max_to_keep: self.delete(force = True)
        
        self._state.update({
            'counter'   : self.counter + 1,
            'loaded'    : len(self),
            'checkpoints'   : self.checkpoints + [self.infos]
        })

    def set_best_checkpoint_infos(self, epoch, logs):
        self._state['best_checkpoint'] = {** logs, 'epoch' : epoch}
        self.save_state()

    def get_filename(self, infos = None, ckpt_format = None):
        """ Return the effective checkpoint filename with the given configuration """
        if not ckpt_format:
            ckpt_format = standardize_checkpoint_format(self.checkpoint_format)
        
        if infos is None: infos = self.infos
        return os.path.join(self.directory, ckpt_format.format(** infos))
    
    def save_state(self):
        dump_json(filename = self.checkpoint_file, data = self._state, indent = 4)

    def save(self, directory = None, *, ckpt_format = None, ** kwargs):
        if not directory and not self.checkpoints: os.makedirs(self.directory, exist_ok = True)
        
        filename = self.get_filename(ckpt_format = ckpt_format)
        if directory: filename = os.path.join(directory, os.path.basename(filename))

        logger.info('Save model weights to `{}`'.format(filename))
        if filename.endswith('.keras'):
            self.model.save(filename, ** kwargs)
        elif filename.endswith('.weights.h5'):
            self.model.save_weights(filename, ** kwargs)
        else:
            raise ValueError('Unsupported weights extension : {}'.format(filename))
        
        if not directory and not ckpt_format:
            if not any(self.infos == ckpt for ckpt in self):
                self.__update_state()
                self.save_state()
        
        return filename
    
    def load(self, checkpoint = None, *, epoch = None, ** kwargs):
        if epoch is not None:
            for i, ckpt in enumerate(self):
                if ckpt['epoch'] == epoch:
                    checkpoint = i
                    break
            
            if checkpoint is None:
                raise ValueError('No checkpoint found for epoch {}'.format(epoch))
        
        from_tf = False
        if checkpoint is None:
            if len(self) != 0 and os.path.exists(self.loaded_checkpoint):
                checkpoint = self.loaded_checkpoint
            elif any(f.endswith('.index') for f in os.listdir(self.directory)):
                from_tf     = True
                checkpoint  = self.directory
        elif isinstance(checkpoint, int):
            self._state['loaded'] = checkpoint
            checkpoint = [self.get_filename(k, self[checkpoint]) for k in self._models]
        elif checkpoint == 'best':
            self._state['loaded'] = 'best'
            checkpoint = self.best_checkpoint

        filename = checkpoint
        
        logger.info('Loading weights from `{}`'.format(filename))
        if filename.endswith(('.keras', '.weights.h5')):
            self.model.load_weights(filename, ** kwargs)
        elif from_tf or os.path.exists(f'{filename}.index'):
            try:
                import tensorflow as tf
                
                from models.weights_converter import (
                    name_based_partial_transfer_learning, load_saved_model_variables
                )
                
                name_based_partial_transfer_learning(
                    model, load_saved_model_variables(filename), source = 'saved_model'
                )
            except Exception as e:
                warnings.warn('Unable to restore weights from tensorflow checkpoint : {}\nMake sure to manually restore weights, as the model is currently randomly initialized !'.format(e))
        else:
            raise RuntimeError('Unsupported checkpoint file format : {}'.format(filename))
        
        if len(self) == 0 or from_tf:
            self._state['counter'] = 0
            self.save()
    
    def delete(self, index = 0, force = False):
        """ Deletes the checkpoint at the given `index` (default 0, the oldest checkpoint) """
        if index == self.loaded and self.max_to_keep != 1:
            if not force:
                raise RuntimeError('You are trying to delete the currently loaded checkpoint !')
        
        weight_file = self.get_filename(self[index])
        logger.debug('Removing weight file {}'.format(weight_file))
        os.remove(weight_file)
        
        self._state['checkpoints'].pop(index)
        if isinstance(self._state['loaded'], int) and index <= self._state['loaded']:
            self._state['loaded'] -= 1

    def clear(self):
        for i in reversed(range(len(self))):
            if i != self.loaded_index: self.delete(i)
    
@cache
def standardize_checkpoint_format(ckpt_format):
    parts = ckpt_format.split('.')
    ckpt_format = parts[0]
    ext = '.'.join(parts[1:]) if len(parts) > 1 else None
    if not ext: ext = '.keras'
    else:       ext = '.' + ext
    if '{counter' not in ckpt_format: ckpt_format += '-{counter:04d}'
    return ckpt_format + ext