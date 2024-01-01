# Copyright (C) 2022-now yui-mhcp project's author. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import tensorflow as tf
    
from unitests import CustomTestCase
from custom_train_objects import _losses, _metrics, _optimizers, _schedulers
from custom_train_objects import get_loss, get_metrics, get_optimizer
from custom_train_objects.optimizers import DivideByStep

class TestCustomObjects(CustomTestCase):
    def test_losses(self):
        for loss in _losses.keys():
            with self.subTest(loss = loss):
                try:
                    loss_obj = get_loss(loss, name = loss)
                except TypeError:
                    continue
                
                self.assertTrue(isinstance(loss_obj, tf.keras.losses.Loss), str(loss_obj))
                self.assertEqual(
                    get_loss(tf.keras.losses.serialize(loss_obj)), loss_obj
                )
                self.assertEqual(get_loss({
                    'class_name' : loss_obj.__class__.__name__, 'config' : loss_obj.get_config()
                }), loss_obj)
                config = loss_obj.get_config()
                config.pop('fn', None)
                self.assertEqual(get_loss(loss, ** config), loss_obj)
                
    def test_metrics(self):
        for metric in _metrics.keys():
            with self.subTest(metric = metric):
                try:
                    metric_obj = get_metrics(metric, name = metric)
                except TypeError:
                    continue
                
                self.assertTrue(isinstance(metric_obj, tf.keras.metrics.Metric), str(metric_obj))
                self.assertEqual(
                    get_metrics(tf.keras.metrics.serialize(metric_obj)), metric_obj
                )
                self.assertEqual(get_metrics({
                    'class_name' : metric_obj.__class__.__name__, 'config' : metric_obj.get_config()
                }), metric_obj)
                config = metric_obj.get_config()
                config.pop('fn', None)
                self.assertEqual(get_metrics(metric, ** config), metric_obj)

    def test_optimizers(self):
        for optim in _optimizers.keys():
            with self.subTest(optimizer = optim):
                optimizer = get_optimizer(optim, name = optim)
                
                self.assertTrue(isinstance(optimizer, tf.keras.optimizers.Optimizer), str(optimizer))
                self.assertEqual(
                    get_optimizer(tf.keras.optimizers.serialize(optimizer)), optimizer
                )
                self.assertEqual(get_optimizer({
                    'class_name' : optimizer.__class__.__name__, 'config' : optimizer.get_config()
                }), optimizer)
                self.assertEqual(get_optimizer(optim, ** optimizer.get_config()), optimizer)

        scheduler = DivideByStep(factor = 5)
        optimizer = get_optimizer('adam', lr = scheduler)
        if hasattr(optimizer, '_learning_rate'):
            self.assertEqual(optimizer._learning_rate, scheduler)
        else:
            self.assertEqual(optimizer.learning_rate, scheduler)
        self.assertEqual(get_optimizer(
            'adam', lr = {'name' : scheduler.__class__.__name__, ** scheduler.get_config()}
        ), optimizer)
        self.assertEqual(
            get_optimizer('adam', ** optimizer.get_config()), optimizer
        )
        self.assertEqual(
            get_optimizer(tf.keras.optimizers.serialize(optimizer)), optimizer
        )
