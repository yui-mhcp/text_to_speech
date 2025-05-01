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
import logging

from .ops import get_backend

logger = logging.getLogger(__name__)

_limited_memory = False

def set_gpu_config(backend = None, precision = None, gpu_memory = None, visible_devices = None, ** _ ):
    if backend:     set_backend(backend)
    if precision:   set_default_precision(precision)
    if visible_devices is not None: set_visible_devices(visible_devices)
    if gpu_memory:  limit_gpu_memory(gpu_memory)
    
def set_backend(backend):
    if 'keras' in sys.modules: raise RuntimeError('`keras` has already been imported !')
    
    os.environ['KERAS_BACKEND'] = backend

def set_default_precision(precision):
    import keras
    keras.mixed_precision.set_global_policy(precision)

def set_visible_devices(devices):
    if not isinstance(devices, list): devices = [devices]
    
    if get_backend() == 'tensorflow':
        import tensorflow as tf
        available = tf.config.list_physical_devices('GPU')
        tf.config.set_visible_devices([available[dev] for dev in devices], 'GPU')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(dev) for dev in devices])

def _limit_gpu_memory_tf(limit):
    """ Limits the tensorflow visible GPU memory on each available physical device """
    global _limited_memory
    if _limited_memory or not limit: return
    
    import tensorflow as tf
    
    gpus = tf.config.list_physical_devices('GPU')
    try:
        for gpu in gpus:
            tf.config.set_logical_device_configuration(gpu, [
                tf.config.LogicalDeviceConfiguration(memory_limit = limit)
            ])
        _limited_memory = True
    except Exception as e:
        logger.error("Error while limiting tensorflow GPU memory : {}".format(e))

def limit_gpu_memory(limit):
    logger.info('Memory limited to {}Mb'.format(limit))
    if get_backend() == 'tensorflow':
        _limit_gpu_memory_tf(limit)
    else:
        logger.warning('`limi_gpu_memory` is not implemented for {}'.format(_limit_gpu_memory_tf()))
        

def _get_memory_usage_tf(gpu = 0, reset = True):
    import tensorflow as tf

    if isinstance(gpu, int): gpu = 'GPU:{}'.format(gpu)
    
    mem_usage = tf.config.experimental.get_memory_info(gpu)
    if reset: tf.config.experimental.reset_memory_stats(gpu)
    return mem_usage

def _get_memory_usage_pt(gpu = 0, reset = True):
    import torch

    if isinstance(gpu, int): gpu = 'cuda:{}'.format(gpu)
    
    current = torch.cuda.memory_allocated(gpu)
    peak    = torch.cuda.max_memory_allocated(gpu)
    if reset: torch.cuda.reset_peak_memory_stats(gpu)
    return {'current' : current, 'peak' : peak}

def get_memory_usage(gpu = 0, backend = None, ** kwargs):
    if not backend: backend = get_backend()
        
    if backend == 'tensorflow':
        return _get_memory_usage_tf(gpu, ** kwargs)
    elif backend == 'torch':
        return _get_memory_usage_pt(gpu, ** kwargs)
    else:
        logger.warning('`limi_gpu_memory` is not implemented for {}'.format(_limit_gpu_memory_tf()))
        return {}

def show_memory(message = '', ** kwargs):
    mem_usage = get_memory_usage(** kwargs)
    
    logger.info('{}{}'.format(message if not message else message + '\t: ', {
        k : '{:.3f} Gb'.format(v / 1024 ** 3) for k, v in mem_usage.items()
    }))
    return mem_usage

def get_gpu_memory_infos(gpu = 0):
    try:
        import nvidia_smi
    except:
        logger.error('This function requires `nvidia_smi` : please run `pip install nvidia-ml-py3`')
        return {}
    
    nvidia_smi.nvmlInit()

    device = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu)
    infos  = nvidia_smi.nvmlDeviceGetMemoryInfo(device)
    return {'total' : infos.total, 'free' : infos.free, 'used' : infos.used}

def show_gpu_memory_infos(message = '', gpu = 0):
    mem_infos = get_gpu_memory_infos(gpu = gpu)
    
    logger.info('{}{}'.format(message if not message else message + '\t: ', {
        k : '{:.3f} Gb'.format(v / 1024 ** 3) for k, v in mem_infos.items()
    }))
    return mem_infos
