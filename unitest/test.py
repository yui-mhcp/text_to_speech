
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
import re
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

RUN_MESSAGE     = 'Running test {name} ({success} success, {fail} failure)'

RANDOM_WARNING  = """
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

def set_current_test(test):
    global _current_test
    _current_test = test

def get_current_test():
    global _current_test
    return _current_test

def _maybe_call(data, * args, ** kwargs):
    return data(* args, ** kwargs) if callable(data) else data

def Test(fn = None, ** kwargs):
    if fn is not None: return TestSuite(fn, ** kwargs)
    return lambda fn: Test(fn, ** kwargs)

class TestSuite:
    def __init__(self,
                 fn     = None,
                 name   = None,
                 
                 model  = None,
                 random = None,
                 sequential = False,
                 
                 root   = _default_root,
                 overwrite  = False,

                 ** kwargs
                ):
        """
            Represents a suite of multiple tests
            
            Arguments :
                - fn    : the decorated function that internally calls `assert_[...]`
                - name  : the test's name (default to `fn.__name__`)
                
                - model : model's name on which the test is based
                - random    : whether this test contains randomness or not (if `True`, make sure to fix the seed at the beginning of `fn` !)
                - sequential    : whether the test is sequential or not (i.e. if an internal test fails, stops)
                
                - root  : the root directory to save results / information
                - overwrite : whether to overwrite if this test already exists
        """
        self.fn     = fn
        self.name   = name if name is not None else fn.__name__
        self.kwargs = kwargs
        
        self.root   = root
        self.overwrite  = overwrite
        
        self.model  = model
        self.random = random
        self.sequential = sequential
        
        self.__stopped_reason   = None
        self.__finished = False
        
        self.__time     = -1.
        self.__order    = []
        self.__tests    = {}
        self.__results  = {}
        
        if model:
            try:
                from models.model_utils import is_model_name
                if not is_model_name(model):
                    self.__stopped_reason   = 'Model `{}` not found'.format(model)
                
            except ImportError as e:
                self.__stopped_reason   = 'Module `models` not found'
        
        if not self.stopped:
            if overwrite and os.path.exists(self.directory): shutil.rmtree(self.directory)

            global _all_tests
            _all_tests[self.name] = self
    
    @property
    def directory(self):
        return os.path.join(self.root, self.name)
    
    @property
    def config_file(self):
        return os.path.join(self.directory, 'config.json')
    
    @property
    def stopped(self):
        return self.__stopped_reason is not None
    
    @property
    def finished(self):
        return self.__finished or self.stopped
    
    @property
    def test_names(self):
        if not self.finished: raise RuntimeError("This test has not been executed yet")
        return self.__order
    
    @property
    def tests(self):
        if not self.finished: raise RuntimeError("This test has not been executed yet")
        return self.__tests
    
    @property
    def results(self):
        if not self.finished: raise RuntimeError("This test has not been executed yet")
        return self.__results

    @property
    def passed(self):
        if not self.finished: raise RuntimeError("This test has not been executed yet")
        return not self.stopped and all(res['passed'] for res in self.results.values())
    
    @property
    def passed_tests(self):
        return [(name, res) for name, res in self.__results.items() if res['passed']]

    @property
    def failed_tests(self):
        return [(name, res) for name, res in self.__results.items() if not res['passed']]
    
    @property
    def time(self):
        return self.__time
    
    def __len__(self):
        return -1 if not self.finished else len(self.tests)
    
    def __getitem__(self, idx):
        if isinstance(idx, int): idx = self.__order[idx]
        return self.__tests[idx]
    
    def __str__(self):
        success, failures = self.passed_tests, self.failed_tests
        des = "===== {} =====\n".format(self.name)
        des += "Success : {} ({} success, {} failures)\n".format(
            self.passed, len(success), len(failures)
        )
        if self.stopped:
            des += "Stopped reason : {}\n".format(self.__stopped_reason)
        if failures:
            des += "Failures :\n"
            for test_name, res in failures:
                des += "- {} (time {}) : {}\n".format(
                    test_name, time_to_string(res['time']), res['message']
                )
        return des
    
    def __call__(self, * args, ** kwargs):
        return self.run(* args, ** kwargs)
    
    def __contains__(self, name):
        return name in self.__tests
    
    def run_test(self,
                 target,
                 value,
                 compare_fn,
                 * args,
                 
                 is_consistency = False,
                 random     = False,
                 should_fail    = False,
                 stop_if_fail   = False,
                 expected_exception  = None,
                 
                 compare_kwargs = None,
                 
                 name   = 'test_{}',
                 ** kwargs
                ):
        if compare_kwargs is None: compare_kwargs = {}
        
        if name in self.__order: name += '_{}'
        if '{}' in name:
            name = name.format(len([
                k for k in self.__tests if re.match(name.replace('{}', '\\d+'), k) is not None
            ]))
        
        self.__order.append(name)
        
        if isinstance(compare_fn, str):
            global _test_fn
            if compare_fn not in _test_fn:
                raise ValueError('Unknown compare fn for test {}\n  Accepted : {}\n  Got : {}'.format(
                    name, tuple(_test_fn.keys()), test['compare_fn']
                ))
            compare_fn  = _test_fn[compare_fn]
        
        random  = random or self.random
        stop_if_fail    = stop_if_fail or self.sequential
        
        t0 = time.time()
        try:
            is_first_run = False # checks whether it is the 1st run of a consistency test
            if is_consistency:
                compare_kwargs['raw_compare_if_filename'] = True
                
                target_file = os.path.join(self.directory, name)
                file_ext    = glob.glob(target_file + '.*')
                
                if len(file_ext) > 1:
                    raise RuntimeError('Multiple target files found : {}'.format(file_ext))
                elif len(file_ext) == 1:
                    target, value = load_data(file_ext[0]), target
                else:
                    logger.info('Creating consistency for {} in {}'.format(name, target_file))
                    
                    os.makedirs(self.directory, exist_ok = True)
                    
                    is_first_run    = True
                    target, value   = _maybe_call(target, * args, ** kwargs), target
                    dump_data(target_file, target)
            
            
            target  = _maybe_call(target, * args, ** kwargs)
            
            if is_consistency and random and is_first_run:
                # by default set the test as failed because calling 2 times the same function with the same seed will automatically give different results if it is done in the same run
                passed, msg = False, 're-run the test to check consistency'
            elif expected_exception is not None:
                # if an exception was expected and not raised, it is an error so test failed
                passed, msg = False, 'should throw an exception but did not'
            else:
                # build the `value` (if needed) and calls the comparison function
                value   = _maybe_call(value, * args, ** kwargs)
                
                passed, msg = compare_fn(target, value, ** compare_kwargs)
                if should_fail and passed:
                    passed, msg = False, 'Test passed but was expected to fail'
        
        except Exception as e:
            if expected_exception is not None:
                if isinstance(e, expected_exception):
                    passed, msg = True, '{} has well been raised'.format(e.__class__.__name__)
                else:
                    passed, msg = False, '{} was expected but raised {}'.format(
                        expected_exception, e
                    )
            else:
                passed, msg = False, 'unexpected exception raised : {}'.format(e)
        
        self.__results[name] = {
            'passed' : passed, 'message' : msg, 'time' : time.time() - t0
        }
        self.__tests[name]  = {
            'target'    : target,
            'value'     : value,
            'compare_fn'    : compare_fn,
            'args'  : args,
            'kwargs'    : kwargs,
            
            'should_fail'   : should_fail,
            'stop_if_fail'  : stop_if_fail,
            'expected_exception'    : expected_exception,
            'compare_kwargs'    : compare_kwargs
        }
        if self.verbose:
            print(RUN_MESSAGE.format(
                name = self.name, fail = len(self.failed_tests), success = len(self.passed_tests)), end = '\r'
            )
        
        # checks whether it should stop or not
        # if it is a random-based test and it is the first run, it will automatically fail
        if stop_if_fail and not (random and is_first_run):
            assert passed, '{} has failed'.format(name)
        
        return passed
    
    def run(self, verbose = True, ** kwargs):
        """ Runs the test by running `fn` """
        set_current_test(self)
        
        start   = time.time()
        try:
            self.verbose = verbose
            if verbose: print(RUN_MESSAGE.format(name = self.name, fail = 0, success = 0), end = '\r')
            self.fn()
            self.__finished = True
            if self.verbose: print()
        except Exception as e:
            if verbose: print()
            self.__stopped_reason = str(e)
        
        self.__time = time.time() - start
        
        set_current_test(None)
        self.save()
        return self

    def get_results(self):
        return {
            'name'  : self.name,
            'time'  : self.time,
            'finished'  : self.__finished,
            'success'   : self.passed,
            'stopped_reason'    : self.__stopped_reason,
            'test_results'  : self.__results
        }
    
    def save(self):
        os.makedirs(self.directory, exist_ok = True)
        
        dump_json(os.path.join(self.directory, 'results.json'), self.get_results(), indent = 4)

class TestResult:
    def __init__(self, tests):
        self.tests = tests

    @property
    def passed(self):
        return [t for t in self if t.passed]
    
    @property
    def failed(self):
        return [t for t in self if not t.passed]
    
    @property
    def success(self):
        return len(self.failed) == 0
    
    @property
    def summary(self):
        return pd.DataFrame(self.get_summary(self.tests)).T
    
    def __len__(self):
        return len(self.tests)
    
    def __getitem__(self, idx):
        return self.tests[idx]
    
    def __repr__(self):
        return repr(self.summary)

    def __str__(self):
        passed, failed = self.passed, self.failed
        des = "===== Tests summary =====\n"
        des += "Success : {} ({} passed, {} failed)\n".format(
            len(failed) == 0, len(passed), len(failed)
        )
        if failed:
            des += "\nFailed :\n{}\n".format('\n'.join(str(t) for t in failed))
        return des

    def get_summary(self, tests):
        summary = {}
        for t in tests:
            infos = t.get_results()
            infos['# success'] = len([t for _, t in infos['test_results'].items() if t['passed']])
            infos['# failures'] = len([t for _, t in infos['test_results'].items() if not t['passed']])
            for k in ('name', 'test_results', 'finished'): infos.pop(k)
            summary[t.name] = infos
        return summary

def run_tests(tests = None, verbose = True):
    results = []
    for test_name, test in _all_tests.items():
        if tests is not None:
            if isinstance(tests, (list, tuple)) and test_name not in tests: continue
            elif re.match(tests, test_name) is None: continue
            
        results.append(test(verbose = verbose))
    
    results = TestResult(results)
    if verbose: print(results)
    return results

_test_fn    = {
    'is_in'     : is_in,
    'is_equal'  : is_equal,
    'is_diff'   : is_diff,
    'is_smaller'    : is_smaller,
    'is_greater'    : is_greater
}