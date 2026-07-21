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
import sys
import atexit
import logging

from .ops import get_backend

logger = logging.getLogger(__name__)

__all__ = [
    'set_gpu_config',
    'set_backend',
    'set_default_precision',
    'set_visible_devices',
    'limit_gpu_memory',
    'get_memory_usage',
    'show_memory',
    'get_gpu_memory_infos',
    'show_gpu_memory_infos',
]

_limited_memory   = False
_nvml_initialized = False

def set_gpu_config(*, backend = None, precision = None, gpu_memory = None, gpu = None, ** _ ):
    if backend:     set_backend(backend)
    if precision:   set_default_precision(precision)
    if gpu is not None: set_visible_devices(gpu)
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
        # Both `torch` and `jax` honor `CUDA_VISIBLE_DEVICES` (as long as it is set
        # before CUDA is initialized), so a single branch covers them.
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(dev) for dev in devices)

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
    if get_backend() == 'tensorflow':
        logger.info('Memory limited to {}Mb'.format(limit))
        _limit_gpu_memory_tf(limit)
    else:
        logger.warning('`limit_gpu_memory` is not implemented for {}'.format(get_backend()))


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

def _get_memory_usage_jax(gpu = 0, reset = True):
    import jax

    stats = jax.devices()[gpu].memory_stats()
    # `jax` exposes no public API to reset the peak counter, so `reset` is a no-op here.
    return {'current' : stats.get('bytes_in_use', 0), 'peak' : stats.get('peak_bytes_in_use', 0)}

def get_memory_usage(gpu = 0, backend = None, ** kwargs):
    if not backend: backend = get_backend()

    if backend == 'tensorflow':
        return _get_memory_usage_tf(gpu, ** kwargs)
    elif backend == 'torch':
        return _get_memory_usage_pt(gpu, ** kwargs)
    elif backend == 'jax':
        return _get_memory_usage_jax(gpu, ** kwargs)
    else:
        logger.warning('`get_memory_usage` is not implemented for {}'.format(backend))
        return {}

def show_memory(message = '', ** kwargs):
    mem_usage = get_memory_usage(** kwargs)
    
    logger.info('{}{}'.format(message if not message else message + '\t: ', {
        k : '{:.3f} Gb'.format(v / 1024 ** 3) for k, v in mem_usage.items()
    }))
    return mem_usage

def get_gpu_memory_infos(gpu = 0):
    """
        Returns the global GPU memory (in bytes) as ``{'total', 'free', 'used'}``.

        Unlike `get_memory_usage` (which reports the current process' allocation), this returns
        the device-wide memory, which is what matters e.g. to size a TRT-LLM KV-cache.

        If `torch` is *already* imported, its `cuda.mem_get_info` is used : it is free (no extra
        dependency), and it honors `CUDA_VISIBLE_DEVICES`. Otherwise, we fall back to `NVML`
        (the official `nvidia-ml-py` / `pynvml`), without ever importing a backend.
    """
    if 'torch' in sys.modules:
        infos = _get_gpu_memory_infos_torch(gpu)
        if infos: return infos
    return _get_gpu_memory_infos_nvml(gpu)

def _get_gpu_memory_infos_torch(gpu = 0):
    import torch

    try:
        free, total = torch.cuda.mem_get_info(gpu)
    except Exception as e:
        # e.g. no CUDA device available for `torch` : let the caller fall back to `NVML`
        logger.warning('Unable to get GPU memory infos from `torch` : {}'.format(e))
        return {}
    return {'total' : total, 'free' : free, 'used' : total - free}

def _get_gpu_memory_infos_nvml(gpu = 0):
    try:
        import pynvml
    except ImportError:
        try:
            import nvidia_smi as pynvml    # legacy `nvidia-ml-py3` fallback
        except ImportError:
            logger.error(
                'This function requires `pynvml` : please run `pip install nvidia-ml-py`'
            )
            return {}

    _init_nvml(pynvml)

    device = pynvml.nvmlDeviceGetHandleByIndex(gpu)
    infos  = pynvml.nvmlDeviceGetMemoryInfo(device)
    return {'total' : infos.total, 'free' : infos.free, 'used' : infos.used}

def _init_nvml(nvml):
    """ Initializes `NVML` once, registering a proper shutdown at exit """
    global _nvml_initialized
    if _nvml_initialized: return

    nvml.nvmlInit()
    atexit.register(nvml.nvmlShutdown)
    _nvml_initialized = True

def show_gpu_memory_infos(message = '', gpu = 0):
    mem_infos = get_gpu_memory_infos(gpu = gpu)
    
    logger.info('{}{}'.format(message if not message else message + '\t: ', {
        k : '{:.3f} Gb'.format(v / 1024 ** 3) for k, v in mem_infos.items()
    }))
    return mem_infos