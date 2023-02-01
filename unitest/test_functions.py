
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

from unitest.test import get_current_test

def set_sequential(sequential = True):
    get_current_test().sequential = sequential

def assert_true(value, * args, ** kwargs):
    assert_equal(True, value, * args, ** kwargs)

def assert_false(value, * args, ** kwargs):
    assert_equal(False, value, * args, ** kwargs)

def assert_none(value, * args, ** kwargs):
    assert_equal(None, value, * args, ** kwargs)

def assert_not_none(value, * args, ** kwargs):
    assert_not_equal(None, value, * args, ** kwargs)
    
def assert_model_output(fn, * args, max_err = 1e-3, ** kwargs):
    kwargs.setdefault('compare_kwargs', {'max_eer' : max_err})
    assert_function(fn, * args, ** kwargs)
    
def assert_function(fn, * args, ** kwargs):
    get_current_test().run_test(fn, None, 'is_equal', * args, is_consistency = True, ** kwargs)

def assert_exception(fn, exception, * args, ** kwargs):
    get_current_test().run_test(
        fn, None, 'is_equal', * args, expected_exception = exception, ** kwargs
    )

def assert_contains(target, value, * args, ** kwargs):
    get_current_test().run_test(target, value, 'is_in', * args, ** kwargs)
    
def assert_smaller(target, value, * args, ** kwargs):
    get_current_test().run_test(target, value, 'is_smaller', * args, ** kwargs)

def assert_greater(target, value, * args, ** kwargs):
    get_current_test().run_test(target, value, 'is_greater', * args, ** kwargs)

def assert_equal(target, value, * args, ** kwargs):
    get_current_test().run_test(target, value, 'is_equal', * args, ** kwargs)

def assert_not_equal(target, value, * args, ** kwargs):
    get_current_test().run_test(target, value, 'is_diff', * args, ** kwargs)
