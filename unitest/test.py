
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
import time
import shutil
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from utils import time_to_string, load_json, dump_json, load_pickle, dump_pickle
from utils.comparison_utils import is_equal, is_diff, is_smaller, is_greater
from models.model_utils import is_model_name

RANDOM_WARNING = """
/!\ WARNING /!\ This test contains randomness, make sure to set explicit seed to have reproducible results.
Furthermore, if multiple functions are random, make sure to put different seeds for each call.

If it is the 1st run, it is normal that you have errors, run tests again to check consistency / reproducibility
"""

_default_root   = os.path.join('unitest', 'tests')

_all_tests     = {}

_current_test  = None

def _maybe_call(data, * args, ** kwargs):
    if callable(data):
        logging.debug('Calling function {}'.format(data))
        return data(* args, ** kwargs)
    return data

def _maybe_save_target(filename, data, overwrite = False):
    if isinstance(data, tf.Tensor): data = data.numpy()
    if isinstance(data, np.ndarray):
        filename += '.npy'
        if os.path.exists(filename) and not overwrite: return filename
        np.save(filename, data)
        data = filename
    elif isinstance(data, pd.DataFrame):
        filename += '.csv'
        if os.path.exists(filename) and not overwrite: return filename
        data.to_csv(filename)
        data = filename
    elif isinstance(data, (list, tuple)):
        filename += '.pkl'
        if os.path.exists(filename) and not overwrite: return filename
        dump_pickle(filename, data)
        data = filename
    
    return data
        
def Test(fn = None, ** kwargs):
    if fn is not None: return TestSuite(fn, ** kwargs)
    return lambda fn: Test(fn, ** kwargs)

class TestSuite:
    def __init__(self, fn = None, sequential = False, name = None, root = _default_root,
                 contains_randomness = False, model_dependant = None, overwrite = False, ** kwargs):
        if model_dependant:
            try:
                from models.model_utils import is_model_name
            except ImportError:
                logging.error("You try to make a model-dependant test without the `models` module !")
                return
            if not is_model_name(model_dependant):
                logging.warning('Test {} depends on model {} which does not exist. Skipping the test'.format(fn.__name__, model_dependant))
                return
        
        self.name   = name if name is not None else fn.__name__
        self.root   = root
        self.kwargs = kwargs
        self.overwrite  = overwrite
        self.sequential = sequential
        self.contains_randomness    = contains_randomness
        
        self.__fn   = fn
        self.__tests    = []
        self.__infos    = None
        self.__built    = False
        self.__executed = False
        
        if not overwrite and os.path.exists(self.config_file): self.restore()
        if overwrite and os.path.exists(self.directory): shutil.rmtree(self.directory)
        
        valid_test = self.build()
        

        global _all_tests
        _all_tests[self.name] = self
    
    def build(self):
        if self.__built: return
        global _current_test

        self.__test_idx = 0
        
        _current_test = self
        self.__fn()
        _current_test = None
        
        if self.__infos is None: self.__infos = []
        self.__infos += [None for _ in range(len(self.__tests) - len(self.__infos))]

        self.__built    = True
    
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
    def executed(self):
        return self.__executed
    
    @property
    def tests(self):
        if not self.built:
            raise ValueError("You must build test suites to get tests")
        return self.__tests
    
    @property
    def infos(self):
        if not self.executed:
            raise ValueError("You must run tests suite to access information results")
        return self.__infos

    @property
    def interrupted(self):
        return self.executed and self.sequential and not all([i is not None for i in self.infos])
    
    @property
    def succeed(self):
        return all(self.results)
    
    @property
    def results(self):
        return [i is not None and i['passed'] for i in self.infos]
    
    @property
    def passed(self):
        return [t for t, success in zip(self.tests, self.results) if success]
    
    @property
    def failed(self):
        return [t for t, success in zip(self.tests, self.results) if not success]
    
    @property
    def time(self):
        return sum([i['time'] for i in self.infos if i is not None])
    
    def __len__(self):
        return -1 if not self.built else len(self.tests)
    
    def __getitem__(self, idx):
        return self.tests[idx]
    
    def __str__(self):
        des = "===== {} =====\n".format(self.name)
        if not self.built:
            des += "Test suite not initialized yet"
            return des
        elif not self.executed:
            des += "{} tests not executed yet".format(len(self))
            return des
        elif self.interrupted:
            executed = [i for i in self.infos if i is not None]
            des += "Interrupted by a failure at index {} with message {}\n".format(
                len(executed) - 1, executed[-1]['message']
            )
            return des
        
        des += "{} tests executed in {} ({} passed)\n".format(
            len(self), time_to_string(self.time), len(self.passed)
        )
        
        if len(self.failed) > 0:
            indexes = [i for i, info in enumerate(self.infos) if not info['passed']]
            des += "Failed tests ({}) :\n".format(len(indexes))
            for idx in indexes[:5]:
                des += "- Test {} failed with message {}\n".format(idx, self.infos[idx]['message'])
            if len(indexes) > 5: des += "... ({} more)\n".format(len(indexes) - 5)
            if self.contains_randomness:
                des += "\n" + RANDOM_WARNING
        
        return des
    
    def __call__(self, ** kwargs):
        return self.run(** kwargs)
    
    def append(self, compare_fn, target = None, value = None, fn = None, exception = None,
               args = [], kwargs = {}):
        compare_fn_name = compare_fn if isinstance(compare_fn, str) else compare_fn.__name__
        test = {
            'compare_fn' : compare_fn_name, 'target' : target, 'value' : value, 'fn' : fn,
            'exception' : exception, 'args' : args, 'kwargs' : kwargs,
            'compare_kwargs' : kwargs.pop('compare_kwargs', {})
        }
        logging.debug('Append new test on {} : {}'.format(self.name, test))
            
        if self.__test_idx == len(self.__tests):
            self.__tests.append(test)
        else:
            test.pop('target')
            self.__tests[self.__test_idx].update(test)
        
        self.__test_idx += 1
        
        if not isinstance(compare_fn, str):
            global _test_fn
            if compare_fn.__name__ not in _test_fn:
                _test_fn[compare_fn.__name__] = compare_fn
        else:
            assert compare_fn in _test_fn, "Unknown comparison fn {}".format(compare_fn)
        
    def run_test(self, idx, ** kwargs):
        logging.debug("{} run test index {}".format(self.name, idx))
        test = self[idx]
        
        if test['fn'] is not None:
            test_filename = test['target'] if test['target'] is not None else os.path.join(
                os.path.join(self.directory, 'test_{}_target'.format(idx))
            )
            if test.get('exception', None) is not None:
                test['target'] = test['fn']
            elif not isinstance(test_filename, str) or not os.path.exists(test_filename):
                logging.info("Initializing function consistency test for {} at index {}".format(self.name, idx))
                test.update({'target' : self.save_test_result(idx), 'value' : test['fn']})
            else:
                test.update({'target' : test_filename, 'value' : test['fn']})
        
        compare_fn  = _test_fn[test['compare_fn']]

        t0 = time.time()
        
        try:
            logging.debug('target : {} - value : {}'.format(test['target'], test['value']))
            target  = _maybe_call(test['target'], * test['args'], ** test['kwargs'])
            value   = _maybe_call(test['value'], * test['args'], ** test['kwargs'])
            
            if test.get('exception', None) is None:
                passed, msg = compare_fn(target, value, ** test.get('compare_kwargs', {}))
            else:
                passed, msg = False, 'should throws an exception but did not'
        except Exception as e:
            if test.get('exception', None) is not None:
                if isinstance(e, test['exception']):
                    passed, msg = True, ''
                else:
                    passed, msg = False, 'the function do not raised the right exception : got {} - expected {}'.format(test['exception'], type(e))
            else:
                passed, msg = False, '{} : {} with inputs {} and {}'.format(
                    e.__class__.__name__, e, test.get('target', None), test.get('value', None)
                )
        
        self.__infos[idx] = {
            'passed' : passed, 'message' : msg, 'time' : time.time() - t0
        }
        return passed
    
    def run(self, to_run = 'failed', ** kwargs):
        assert to_run in ('all', 'failed')
        
        indexes = range(len(self)) if to_run == 'all' else [
            i for i, info in enumerate(self.__infos) if info is None or not info['passed']
        ]
        
        for idx in indexes:
            passed = self.run_test(idx, ** kwargs)
            if not passed and self.sequential:
                break
        
        self.__executed = True
        self.save()

    def get_config(self):
        return {
            'nam'   : self.name,
            'directory' : self.directory,
            'sequential'    : self.sequential,
            'contains_randomness'   : self.contains_randomness,
            'kwargs'    : self.kwargs,
            
            'tests'     : self.__tests,
            'infos'     : self.__infos
        }
    
    def restore(self):
        try:
            config = load_json(self.config_file)
            tests  = load_pickle(self.tests_file)

            self.__tests    = tests
            self.__infos    = config['infos']
        except Exception as e:
            logging.error("Failed to restore due to {} : {}".format(e.__class__.__name__, str(e)))

    def save(self):
        os.makedirs(self.directory, exist_ok = True)
        
        self.save_config()
        self.save_tests()
    
    def save_test_result(self, idx):
        os.makedirs(self.directory, exist_ok = True)
        test = self[idx]
        
        key     = 'fn' if test['fn'] is not None else 'target'
        target  = _maybe_call(test[key], * test['args'], ** test['kwargs'])
        
        filename = os.path.join(self.directory, 'test_{}_target'.format(idx))
        return _maybe_save_target(filename, target)
        
    def save_tests(self):
        if not self.built: return
        
        tests = self.tests.copy()
        for i, test in enumerate(tests):
            if test.get('exception', None) is None:
                test['target'] = _maybe_call(test['target'], * test['args'], ** test['kwargs'])

                test['target'] = _maybe_save_target(
                    os.path.join(self.directory, 'test_{}_target'.format(i)), test['target']
                )
            else:
                test['target'] = None
            test.pop('value', None)
            test['fn'] = None
        
        dump_pickle(self.tests_file, tests)
    
    def save_config(self):
        config  = self.get_config()
        config.pop('tests')
        
        dump_json(self.config_file, config, indent = 4)
    

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
        return sum([len(t.passed) for t in self])
        
    @property
    def total_failures(self):
        return sum([len(t.failed) for t in self])

    def __str__(self):
        des = "Tests summary :\n"
        des += "{} tests executed in {} ({} passed)\n".format(
            self.total_tests, time_to_string(self.time), self.total_passed
        )
        
        failed = self.failures
        if len(failed) > 0:
            des += "Failed tests ({}) :\n".format(self.total_failures)
            for fail in failed:
                des += "\n" + str(fail)
        
        return des
        
    def __len__(self):
        return len(self.tests)
    
    def __getitem__(self, idx):
        return self.tests[idx]
    
    def run(self, ** kwargs):
        for t in self: t.run(** kwargs)
    
    def assert_succeed(self):
        assert self.succeed, "\n{}".format(self)
        logging.info("All tests succeed !")
    
    def save(self):
        for t in self: t.save()
    

def set_sequential():
    _current_test.sequential = True
        
def assert_true(value, * args, ** kwargs):
    assert_equal(True, value, * args, ** kwargs)

def assert_false(value, * args, ** kwargs):
    assert_equal(False, value, * args, ** kwargs)

def assert_none(value, * args, ** kwargs):
    assert_equal(None, value, * args, ** kwargs)

def assert_not_none(value, * args, ** kwargs):
    assert_not_equal(None, value, * args, ** kwargs)
    
def assert_model_output(fn, * args, max_err = 0.01, err_mode = 'norm', ** kwargs):
    kwargs.setdefault('compare_kwargs', {'max_eer' : max_err, 'err_mode' : err_mode})
    assert_function(fn, * args, ** kwargs)
    
def assert_function(fn, * args, ** kwargs):
    _current_test.append(is_equal, fn = fn, args = args, kwargs = kwargs)

def assert_exception(fn, exception, * args, ** kwargs):
    _current_test.append(is_equal, exception = exception, fn = fn, args = args, kwargs = kwargs)

def assert_smaller(target, value, * args, ** kwargs):
    _current_test.append(is_smaller, target = target, value = value, args = args, kwargs = kwargs)

def assert_greater(target, value, * args, ** kwargs):
    _current_test.append(is_greater, target = target, value = value, args = args, kwargs = kwargs)

def assert_equal(target, value, * args, ** kwargs):
    _current_test.append(is_equal, target = target, value = value, args = args, kwargs = kwargs)

def assert_not_equal(target, value, * args, ** kwargs):
    _current_test.append(is_diff, target = target, value = value, args = args, kwargs = kwargs)

def run_tests(** kwargs):
    tests = TestResult()
    tests.run(** kwargs)
    
    return tests

_test_fn    = {
    'is_equal'  : is_equal,
    'is_diff'   : is_diff,
    'is_smaller'    : is_smaller,
    'is_greater'    : is_greater
}