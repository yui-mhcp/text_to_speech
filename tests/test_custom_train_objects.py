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

import keras
import unittest

from keras import tree

from . import CustomTestCase
try:
    err = None
    from custom_train_objects.losses import _losses, get_loss
    from custom_train_objects.metrics import _metrics, get_metrics
    from custom_train_objects.callbacks import _callbacks, get_callbacks
    from custom_train_objects.optimizers import DivideByStep, _optimizers, get_optimizer
except Exception as e:
    err = e
    _losses = _metrics = _callbacks = _optimizers = {}

def normalize_object(obj):
    if hasattr(obj, 'get_config'):
        return tree.map_structure(normalize_object, obj.get_config())
    elif callable(obj):
        return obj.__name__
    return obj

@unittest.skipIf(err is not None, 'The module import failed due to {}'.format(err))
class TestCustomObject(CustomTestCase):
    items   = None
    serialize   = None
    keras_type  = None
    
    def load(self, * args, ** kwargs):
        raise NotImplementedError()

    def assertEqual(self, value, target):
        return super().assertEqual(normalize_object(value), normalize_object(target))
    
    def _get_original(self, name):
        try:
            obj = self.load(name, name = name)
            self.assertTrue(isinstance(obj, self.keras_type), str(obj))
            return obj
        except:
            return None

    def test_from_config(self):
        if not self.items: return
        for name in self.items.keys():
            obj = self._get_original(name)
            if obj is not None:
                config = {k : v for k, v in obj.get_config().items() if k != 'fn'}
                self.assertEqual(self.load(name, ** config), obj)
    
    def test_from_serialization(self):
        if not self.items: return
        for name in self.items.keys():
            obj = self._get_original(name)
            if obj is not None:
                with self.subTest(name = name):
                    self.assertEqual(self.load(self.module.serialize(obj)), obj)
            
class TestLoss(TestCustomObject):
    items   = _losses
    module  = keras.losses
    keras_type  = keras.losses.Loss
    
    def load(self, * args, ** kwargs):
        return get_loss(* args, ** kwargs)

class TestMetrics(TestCustomObject):
    items   = _metrics
    module  = keras.metrics
    keras_type  = keras.metrics.Metric
    
    def load(self, * args, ** kwargs):
        return get_metrics(* args, ** kwargs)


class TestOptimizer(TestCustomObject):
    items   = _optimizers
    module  = keras.optimizers
    keras_type  = keras.optimizers.Optimizer

    def load(self, * args, ** kwargs):
        return get_optimizer(* args, ** kwargs)

    def test_learning_rate(self):
        self.assertEqual(get_optimizer('adam', lr = 1.).learning_rate, 1.)
        
        scheduler = DivideByStep(factor = 5)
        optimizer = get_optimizer('adam', lr = scheduler)
        self.assertEqual(optimizer._learning_rate, scheduler)
        
        self.assertEqual(get_optimizer(
            'adam', lr = {'name' : 'DivideByStep', ** scheduler.get_config()}
        ), optimizer)
        self.assertEqual(
            get_optimizer('adam', ** optimizer.get_config()), optimizer
        )
        self.assertEqual(
            get_optimizer(keras.optimizers.serialize(optimizer)), optimizer
        )
