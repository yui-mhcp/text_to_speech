
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
import time
import shutil
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from utils import time_to_string, load_data, dump_data, load_json, dump_json, load_pickle, dump_pickle
from utils.comparison_utils import is_in, is_equal, is_diff, is_smaller, is_greater

logger = logging.getLogger(__name__)

RANDOM_WARNING = """
/!\ WARNING /!\ This test contains randomness, make sure to set explicit seed to have reproducible results.
Furthermore, if multiple functions are random, make sure to put different seeds for each call.

If it is the first run, it is normal that you have errors, run tests again to check consistency / reproducibility
"""

_default_root   = os.path.join('unitest', 'test_results')

_all_tests     = {}
_current_test  = None

_test_config    = {
    'name'  : None,
    'compare_fn'    : None,
    'compare_kwargs'    : {},
    'target'    : None,
    'value'     : None, 
    'args'  : [],
    'kwargs'    : {},
    'should_fail'   : False,
    'should_throw_exception'    : False,
    'stop_if_fail'  : False
}
_test_result    = {
    'passed' : None, 'time' : 0, 'message' : None
}

def _maybe_call(data, * args, ** kwargs):
    if callable(data):
        logger.debug('Calling function {}'.format(data))
        return data(* args, ** kwargs)
    return data

def Test(fn = None, ** kwargs):
    if fn is not None: return TestSuite(fn, ** kwargs)
    return lambda fn: Test(fn, ** kwargs)

class TestSuite:
    def __init__(self,
                 fn     = None,
                 name   = None,
                 
                 sequential = False,
                 model_dependant    = None,
                 contains_randomness    = False,
                 
                 root   = _default_root,
                 overwrite  = False,

                 ** kwargs
                ):
        self.fn     = fn
        self.name   = name if name is not None else fn.__name__
        self.kwargs = kwargs
        
        self.root   = root
        self.overwrite  = overwrite
        
        self.sequential = sequential
        self.model_dependant    = model_dependant
        self.contains_randomness    = contains_randomness
        
        self.__fn   = fn
        self.__built    = False
        self.__stopped  = False
        
        self.__order    = []
        self.__tests    = {}
        self.__results  = {}
        
        if model_dependant:
            try:
                from models.model_utils import is_model_name
            except ImportError as e:
                logger.error("You try to make a model-dependant test without the `models` module !\n{}".format(e))
                is_model_name = lambda * args, ** kwargs: False
                self.__stopped  = True
            if not is_model_name(model_dependant):
                logger.warning('Test {} depends on model {} which does not exist. Skipping the test'.format(self.name, model_dependant))
                self.__stopped  = True
        
        if not self.__stopped:
            if overwrite and os.path.exists(self.directory): shutil.rmtree(self.directory)

            global _all_tests
            _all_tests[self.name] = self
    
    def build(self):
        if self.built or self.stopped: return
        
        global _current_test
        _current_test = self

        self.__fn()
        self.__built   = True
        
        _current_test = None
        
        if not self.overwrite and os.path.exists(self.tests_file): self.restore()
    
    @property
    def directory(self):
        return os.path.join(self.root, self.name)
    
    @property
    def config_file(self):
        return os.path.join(self.directory, 'config.json')
    
    @property
    def tests_file(self):
        return os.path.join(self.directory, 'tests.pkl')
    
    @property
    def built(self):
        return self.__built
    
    @property
    def stopped(self):
        return self.__stopped
    
    @property
    def finished(self):
        return self.stopped or all([passed is not None for passed in self.passed])
    
    @property
    def test_names(self):
        if not self.built:
            raise ValueError("You must build test suites to get tests")
        return self.__order
    
    @property
    def tests(self):
        if not self.built:
            raise ValueError("You must build test suites to get tests")
        return [self.__tests[name] for name in self]
    
    @property
    def results(self):
        if not self.finished:
            raise ValueError("You must run tests suite to access information results")
        return [self.__results[name] for name in self]

    @property
    def succeed(self):
        return self.built and all(self.passed)
    
    @property
    def passed(self):
        return [self.is_success(name) for name in self]
    
    @property
    def passed_tests(self):
        return [name for name in self if self.is_success(name)]

    @property
    def failed_tests(self):
        return [name for name in self if self.is_failure(name)]
    
    @property
    def executed_tests(self):
        return [name for name in self if self.is_executed(name)]
    
    @property
    def not_executed_tests(self):
        return [name for name in self if not self.is_executed(name)]
    
    @property
    def time(self):
        return sum([res['time'] for res in self.results])
    
    def __len__(self):
        return 0 if not self.built else len(self.tests)
    
    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self.__tests[idx]

        return self.__order[idx]
    
    def __str__(self):
        des = "===== {} =====\n".format(self.name)
        if self.stopped and not self.built:
            des += "Test not built because something is missing"
        elif not self.built:
            des += "Test suite not initialized yet"
        elif not self.finished:
            des += "{} / {} tests not executed yet".format(len(self.not_executed_tests), len(self))
        elif self.stopped:
            last_fail   = self.failed_tests[-1]
            des += "Test interrupted after a failure at test {} : {}".format(
                last_fail, self.get_result(last_fail)['message']
            )
        else:
            des += "{} tests executed in {} ({} passed)\n".format(
                len(self), time_to_string(self.time), len(self.passed_tests)
            )
            failed = self.failed_tests
            if len(failed) > 0:
                des += "Failed tests ({}) :\n".format(len(failed))
                for fail in failed[:5]:
                    des += "- Test {} failed with message {}\n".format(
                        fail, self.get_result(fail)['message']
                    )
                    
                if len(failed) > 5: des += "... ({} more)\n".format(len(failed) - 5)
                if self.contains_randomness:
                    des += "\n" + RANDOM_WARNING
        
        return des
    
    def __call__(self, ** kwargs):
        return self.run(** kwargs)
    
    def __contains__(self, name):
        return name in self.__tests
    
    def is_success(self, name):
        return self.__results[name]['passed'] == True
    
    def is_failure(self, name):
        return self.__results[name]['passed'] == False
    
    def is_executed(self, name):
        return self.__results[name]['passed'] is not None
    
    def get_result(self, name):
        return self.__results[name]
    
    def append(self,
               compare_fn,
               target   = None,
               value    = None,
               fn   = None,
               name = None,
               args = None,
               kwargs   = None,

               should_fail  = False,
               should_throw_exception   = False,
               
               overwrite    = False,
               stop_if_fail = False,
               compare_kwargs   = None,
               ** other_kwargs
              ):
        if name is None:
            if fn is not None:
                name = fn.__name__ if hasattr(fn, '__name__') else fn.__class__.__name__
            elif callable(target):
                name = target.__name__ if hasattr(target, '__name__') else target.__class__.__name__
            elif callable(value):
                name = value.__name__ if hasattr(value, '__name__') else value.__class__.__name__
            else: name = 'test'
        
        if name == '<lambda>': name = name[1:-1]
        
        if name in self:
            idx = 1
            while '{}_{}'.format(name, idx) in self: idx += 1
            name = '{}_{}'.format(name, idx)
        
        if callable(compare_fn):
            global _test_fn
            if compare_fn.__name__ not in _test_fn: _test_fn[compare_fn.__name__] = compare_fn
            compare_fn = compare_fn.__name__
        
        if kwargs is None: kwargs = {}
        
        test_infos = _test_config.copy()
        test_infos.update({
            'compare_fn'    : compare_fn,
            'target'    : target if value is not None else fn,
            'value'     : value if value is not None else fn,
            'args'  : [] if args is None else args,
            'kwargs'    : {** other_kwargs, ** kwargs},
            
            'overwrite' : overwrite,
            'should_fail'   : should_fail,
            'stop_if_fail'  : stop_if_fail,
            'compare_kwargs'    : {} if compare_kwargs is None else compare_kwargs
        })
        
        logger.debug('Adding test {} to test suite {}'.format(name, self.name))
        
        self.__order.append(name)
        
        self.__tests[name]  = test_infos
        self.__results[name]    = _test_result.copy()
    
    def run_test(self, name, ** kwargs):
        if name not in self:
            raise ValueError('Test suite {} does not contain test {}'.format(self.name, name))
        
        test = self[name]
        
        global _test_fn
        if test['compare_fn'] not in _test_fn:
            raise ValueError('Unknown compare fn for test {}\n  Accepted : {}\n  Got : {}'.format(
                name, tuple(_test_fn.keys()), test['compare_fn']
            ))
        
        compare_fn  = _test_fn[test['compare_fn']]
        stop_if_fail    = test['stop_if_fail'] or self.sequential
        
        t0 = time.time()
        
        try:
            logger.debug('run test {} with target : {} - value : {}'.format(
                name, self.name, test['target'], test['value']
            ))
            
            test['target']  = _maybe_call(test['target'],   * test['args'], ** test['kwargs'])
            test['value']   = _maybe_call(test['value'],    * test['args'], ** test['kwargs'])
            
            if test['should_throw_exception']:
                passed, msg = False, 'should throw an exception but did not'
            else:
                passed, msg = compare_fn(test['target'], test['value'], ** test['compare_kwargs'])
                if test['should_fail'] and passed:
                    passed, msg = False, 'Test passed but was expected to fail'
        except Exception as e:
            if test['should_throw_exception']:
                passed, msg = True, ''
            else:
                passed, msg = False, 'An exception occured : {}'.format(e)
        
        self.__results[name].update({
            'passed' : passed, 'message' : msg, 'time' : time.time() - t0
        })
        if not passed and stop_if_fail:
            self.__stopped = True
        
        return passed
    
    def run(self, to_run = 'failed', ** kwargs):
        assert to_run in ('all', 'failed')
        
        self.build()
        
        names   = [
            name for name in self if to_run == 'all' or not self.is_success(name)
        ]
        
        logger.info('Running test {} ({} tests to run)'.format(self.name, len(names)))
        
        for name in names:
            if self.stopped: break
            self.run_test(name)
        
        self.save()

    def get_config(self):
        return {
            'name'  : self.name,
            'kwargs'    : self.kwargs,

            'root'  : self.root,
            'overwrite' : self.overwrite,

            'sequential'    : self.sequential,
            'model_dependant'   : self.model_dependant,
            'contains_randomness'   : self.contains_randomness
        }
    
    def restore(self):
        try:
            data  = load_pickle(self.tests_file)
            restored_tests, results = data['tests'], data['results']
            
            for name, test in restored_tests.items():
                if self.__tests[name]['overwrite'] or self.overwrite or 'target' not in test:
                    if 'target' in test:
                        logger.info('Overwriting test {} from {}'.format(name, self.name))
                    continue
                else:
                    logger.debug('{} from {} successfully restored !'.format(name, self.name))
                
                test['target'] = load_data(test['target'])
                self.__tests[name].update(test)
                self.__results[name].update(results[name])
        
        except Exception as e:
            logger.error("Failed to restore due to {} : {}".format(e.__class__.__name__, e))

    def save(self):
        os.makedirs(self.directory, exist_ok = True)
        
        self.save_config()
        self.save_tests()
    
    def save_tests(self):
        if not self.built: return
        
        data    = {
            'tests' : {name : test.copy() for name, test in self.__tests.items()},
            'results'   : self.__results
        }
        
        for name, test in data['tests'].items():
            if not test['should_throw_exception']:
                filename        = os.path.join(self.directory, '{}_target'.format(name))
                overwrite_test  = self.overwrite or test.get('overwrite', False)
                test_file       = glob.glob(filename + '.*')
                if test_file and not overwrite_test:
                    test['target'] = test_file[0]
                    continue
                else:
                    for file in test_file: os.remove(file)
                
                logger.debug('Saving {} from {}'.format(name, self.name))
                
                test['target']  = _maybe_call(test['target'], * test['args'], ** test['kwargs'])

                if not callable(test['target']):
                    test['target']  = dump_data(
                        filename    = filename,
                        data        = test['target'],
                        overwrite   = overwrite_test
                    )
            
            for k in ['value', 'overwrite']: test.pop(k)
            if callable(test['target']): test.pop('target')

        dump_pickle(self.tests_file, data)
    
    def save_config(self):
        dump_json(self.config_file, self.get_config(), indent = 4)

class TestResult:
    def __init__(self, tests = None, ** kwargs):
        self.kwargs  = kwargs
        self.__tests = tests if tests is not None else list(_all_tests.values())
    
    @property
    def tests(self):
        return self.__tests
    
    @property
    def time(self):
        return sum([t.time for t in self])
    
    @property
    def succeed(self):
        return all([t.succeed for t in self])
    
    @property
    def passed(self):
        return [t for t in self if t.succeed]

    @property
    def failures(self):
        return [t for t in self if not t.succeed]
    
    @property
    def total_tests(self):
        return sum([len(t) for t in self])
    
    @property
    def total_passed(self):
        return sum([len(t.passed_tests) for t in self])
    
    @property
    def total_failures(self):
        return sum([len(t.failed_tests) for t in self])

    def __str__(self):
        des = "Tests summary :\n"
        des += "{} tests executed in {} ({} passed)\n".format(
            self.total_tests, time_to_string(self.time), self.total_passed
        )
        
        failed = self.failures
        if len(failed) > 0:
            des += "Failed tests ({}) :".format(self.total_failures)
            for fail in failed:
                des += "\n\n" + str(fail)
        
        return des
        
    def __len__(self):
        return len(self.tests)
    
    def __getitem__(self, idx):
        return self.tests[idx]
    
    def run(self, * args, ** kwargs):
        for t in self: t.run(* args, ** kwargs)
    
    def assert_succeed(self):
        assert self.succeed, "\n{}".format(self)
        logger.info("All tests succeed !")
    
    def save(self):
        for t in self: t.save()


def set_sequential():
    global _current_test
    _current_test.sequential = True

def assert_true(value, * args, ** kwargs):
    assert_equal(True, value, * args, ** kwargs)

def assert_false(value, * args, ** kwargs):
    assert_equal(False, value, * args, ** kwargs)

def assert_none(value, * args, ** kwargs):
    assert_equal(None, value, * args, ** kwargs)

def assert_not_none(value, * args, ** kwargs):
    assert_not_equal(None, value, * args, ** kwargs)
    
def assert_model_output(fn, * args, max_err = 0.01, normalize = True, ** kwargs):
    kwargs.setdefault('compare_kwargs', {'max_eer' : max_err, 'normalize' : normalize})
    assert_function(fn, * args, ** kwargs)
    
def assert_function(fn, * args, ** kwargs):
    _current_test.append(is_equal, fn = fn, args = args, ** kwargs)

def assert_exception(fn, exception, * args, ** kwargs):
    kwargs['should_throw_exception'] = True
    _current_test.append(is_equal, value = fn, args = args, ** kwargs)

def assert_contains(target, value, * args, ** kwargs):
    _current_test.append(is_in, target = target, value = value, args = args, ** kwargs)
    
def assert_smaller(target, value, * args, ** kwargs):
    _current_test.append(is_smaller, target = target, value = value, args = args, ** kwargs)

def assert_greater(target, value, * args, ** kwargs):
    _current_test.append(is_greater, target = target, value = value, args = args, ** kwargs)

def assert_equal(target, value, * args, ** kwargs):
    _current_test.append(is_equal, target = target, value = value, args = args, ** kwargs)

def assert_not_equal(target, value, * args, ** kwargs):
    _current_test.append(is_diff, target = target, value = value, args = args, ** kwargs)

def run_tests(* args, ** kwargs):
    tests = TestResult()
    tests.run(* args, ** kwargs)
    
    return tests

_test_fn    = {
    'is_in'     : is_in,
    'is_equal'  : is_equal,
    'is_diff'   : is_diff,
    'is_smaller'    : is_smaller,
    'is_greater'    : is_greater
}