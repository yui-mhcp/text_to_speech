
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
import glob
import json
import tensorflow as tf

from utils import get_object, print_objects, partial
from custom_layers.custom_activations import _activations
from custom_architectures.simple_models import classifier

def __load():
    for module_name in os.listdir('custom_architectures'):
        if module_name.startswith(('__', '.')): continue
        module_name = 'custom_architectures.' + module_name.replace('.py', '')

        module = __import__(
            module_name, fromlist = ['custom_objects', 'custom_functions']
        )
        if hasattr(module, 'custom_objects'):
            custom_objects.update(module.custom_objects)
        if hasattr(module, 'custom_functions'):
            _custom_architectures.update(module.custom_functions)

def get_architecture(architecture_name, * args, ** kwargs):
    return get_object(
        architectures, architecture_name, * args, print_name = 'architecture', err = True, ** kwargs
    )

def print_architectures():
    print_objects(architectures, 'model architectures')

custom_objects = _activations.copy()
_custom_architectures = {}

__load()

_keras_architectures = {
    'densenet121'       : partial(classifier, tf.keras.applications.DenseNet121),
    'densenet169'       : partial(classifier, tf.keras.applications.DenseNet169),
    'densenet201'       : partial(classifier, tf.keras.applications.DenseNet201),
    'inceptionresnetv2' : partial(classifier, tf.keras.applications.InceptionResNetV2),
    'inceptionv3'       : partial(classifier, tf.keras.applications.InceptionV3),
    'mobilenet'         : partial(classifier, tf.keras.applications.MobileNet),
    'mobilenetv2'       : partial(classifier, tf.keras.applications.MobileNetV2),
    'nasnetlarge'       : partial(classifier, tf.keras.applications.NASNetLarge),
    'resnet50'          : partial(classifier, tf.keras.applications.ResNet50),
    'resnet50v2'        : partial(classifier, tf.keras.applications.ResNet50V2),
    'resnet101'         : partial(classifier, tf.keras.applications.ResNet101),
    'resnet101v2'       : partial(classifier, tf.keras.applications.ResNet101V2),
    'resnet152'         : partial(classifier, tf.keras.applications.ResNet152),
    'resnet152v2'       : partial(classifier, tf.keras.applications.ResNet152V2),
    'vgg16'             : partial(classifier, tf.keras.applications.VGG16),
    'vgg19'             : partial(classifier, tf.keras.applications.VGG19),
    'xception'          : partial(classifier, tf.keras.applications.Xception)
}

architectures = {**_keras_architectures, **_custom_architectures}