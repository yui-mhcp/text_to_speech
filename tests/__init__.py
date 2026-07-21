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

import unittest

# re-exported for backward compatibility (existing tests do `from . import ...`)
from ._helpers import (
    data_dir, temp_dir, reproductibility_dir,
    is_tensorflow_available, get_graph_function, get_xla_function, convert_to_tf_tensor,
    _graph_failed
)
from . import asserts

class CustomTestCase(unittest.TestCase):
    """ `TestCase` whose domain assertions delegate to the standalone `tests.asserts`.

        Keeping the logic in `tests.asserts` lets the very same checks be used from
        plain `pytest` functions / fixtures, while this class stays a thin, backward
        compatible wrapper (it only adds `unittest` niceties such as `subTest`).
    """
    def assertEqual(self, target, value, msg = None, *, max_err = 1e-6, ** kwargs):
        asserts.assert_equal(target, value, msg = msg, max_err = max_err, ** kwargs)

    def assertNotEqual(self, target, value, max_err = 1e-6, ** kwargs):
        asserts.assert_not_equal(target, value, max_err = max_err, ** kwargs)

    def assertReproductible(self, value, file, max_err = 1e-6, ** kwargs):
        asserts.assert_reproducible(value, file, max_err = max_err, ** kwargs)

    def assertArray(self, x):
        asserts.assert_array(x)

    def assertTensor(self, x):
        asserts.assert_tensor(x)

    def assertTfTensor(self, x):
        asserts.assert_tf_tensor(x)

    def assertGraphCompatible(self, fn, * args, jit_compile = False, ** kwargs):
        if fn in _graph_failed: return
        elif not is_tensorflow_available():
            self.skipTest('`tensorflow` should be available')

        name = fn.name if hasattr(fn, 'name') else fn.__name__
        with self.subTest('{} compatible : {}'.format('XLA' if jit_compile else 'Graph', name)):
            asserts.assert_graph_compatible(fn, * args, jit_compile = jit_compile, ** kwargs)

    def assertXLACompatible(self, fn, * args, ** kwargs):
        self.assertGraphCompatible(fn, * args, jit_compile = True, ** kwargs)
