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

import time
import unittest

from . import CustomTestCase
try:
    err = None
    from models.nlu.workflows import *
except Exception as e:
    err = e

@unittest.skipIf(err is not None, 'The module import failed due to {}'.format(err))
class TestWorkflows(CustomTestCase):
    def test_sequential(self):
        graph = Graph(
            FunctionNode(lambda context: context.setdefault('numbers', [10, 42, 100])),
            FunctionNode(lambda context: context.setdefault('result', sum(context['numbers'])))
        )
        
        final_context, res = graph.start()
        
        self.assertEqual({'numbers' : [10, 42, 100], 'result' : 152}, final_context)
        self.assertEqual(152, res)
    
    def test_parallel(self):
        def fn_with_delay(context):
            time.sleep(0.1)
            return sum(context['numbers'])
        
        graph = Graph(
            FunctionNode(lambda context: context.setdefault('numbers', [10, 42, 100])),
            ParallelExecution(
                FunctionNode(fn_with_delay, output_key = 'sum_1'),
                FunctionNode(fn_with_delay, output_key = 'sum_2')
            )
        )
        
        t0 = time.time()
        final_context, res = graph.start()
        t1 = time.time()
        
        self.assertEqual({'numbers' : [10, 42, 100], 'sum_1' : 152, 'sum_2' : 152}, final_context)
        self.assertEqual([152, 152], res)
        self.assertTrue(t1 - t0 < 0.2, 'The graph should take less than 0.2 second, but took {:.3f} sec'.format(t1 - t0))

    def test_output_key(self):
        graph = Graph(
            FunctionNode(lambda context: [10, 42, 100], output_key = 'numbers'),
            FunctionNode(lambda context: sum(context['numbers']), output_key = 'result')
        )
        
        final_context, res = graph.start()
        
        self.assertEqual({'numbers' : [10, 42, 100], 'result' : 152}, final_context)
        self.assertEqual(152, res)
    
    def test_condition(self):
        graph = ConditionNode(
            lambda context: context['number'] > 0,
            ValueNode(True, output_key = 'positif'),
            ValueNode(False, output_key = 'positif')
        )
        
        ctx, res = graph.start({'number' : 5})
        
        self.assertEqual({'number' : 5, 'positif' : True}, ctx)
        self.assertTrue(res)

        ctx, res = graph.start({'number' : -5})
        
        self.assertEqual({'number' : -5, 'positif' : False}, ctx)
        self.assertFalse(res)
    
    def test_loop(self):
        def counter(context):
            context['counter'] *= 2
            return context['counter']
        
        graph = LoopNode(
            counter, lambda context: context['counter'] < 10
        )
        
        ctx, res = graph.start({'counter' : 1})
        
        self.assertEqual({'counter' : 16}, ctx)
        self.assertEqual(16, res)

        graph = LoopNode(
            counter, max_iter = 10
        )
        
        ctx, res = graph.start({'counter' : 1})
        
        self.assertEqual({'counter' : 1024}, ctx)
        self.assertEqual(1024, res)
        
        