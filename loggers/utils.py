
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

import logging
import datetime

def time_to_string(secondes):
    """ return a string representation of a time (given in seconds) """
    h = int(secondes // 3600)
    h = "" if h == 0 else "{}h ".format(h)
    m = int((secondes % 3600) // 60)
    m = "" if m == 0 else "{}min ".format(m)
    s = ((secondes % 300) % 60)
    s = "{:.3f} sec".format(s) if m == "" and h == "" else "{}sec".format(int(s))
    return "{}{}{}".format(h, m, s)        

def get_object(available_objects, obj_name, * args,
               print_name = 'object', err = False, 
               allowed_type = None, ** kwargs):
    """
        Get corresponding object based on a name and dict of object names with associated class / function to call
        Arguments : 
            - available_objects : dict of objects names with their associated class / function
            - obj_name      : the objectto construct (either a list, str or instance of 'allowed_type')
            - args / kwargs : the args and kwargs to pass to the constructor
            - print_name    : name for printing if object was not found
            - err   : whether to raise error if object is not available or not
            - allowed_type  : expected return type
        Return : 
            - instance (or list of instance) of the object created
    """
    if allowed_type is not None and isinstance(obj_name, allowed_type):
        return obj_name
    elif obj_name is None:
        return [get_object(
            available_objects, n, *args, print_name = print_name, 
            err = err, allowed_type = allowed_type, ** kw
        ) for n, kw in kwargs.items()]
    
    elif isinstance(obj_name, (list, tuple)):
        return [get_object(
            available_objects, n, *args, print_name = print_name, 
            err = err, allowed_type = allowed_type, ** kwargs
        ) for n in obj_name]
    
    elif isinstance(obj_name, dict):
        return [get_object(
            available_objects, n, *args, print_name = print_name, 
            err = err, allowed_type = allowed_type, ** kwargs
        ) for n, args in obj_name.items()]
    
    elif isinstance(obj_name, str) and obj_name.lower() in to_lower_keys(available_objects):
        return to_lower_keys(available_objects)[obj_name.lower()](*args, **kwargs)
    else:
        if err:
            raise ValueError("{} is not available !\n  Accepted : {}\n  Got :{}".format(
                print_name, tuple(available_objects.keys()), obj_name
            ))
        else:
            logging.warning("{} : '{}' is not available !".format(print_name, obj_name))
        return obj_name

def to_lower_keys(dico):
    return {k.lower() : v for k, v in dico.items()}
