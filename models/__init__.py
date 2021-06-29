import os
import glob

from utils.generic_utils import get_object, print_objects

def __load():
    for module_name in glob.glob('models/*'):
        if not os.path.isdir(module_name): continue
        module_name = module_name.replace(os.path.sep, '.')

        module = __import__(
            module_name, fromlist = ['_models']
        )
        if hasattr(module, '_models'):
            _models.update(module._models)

def get_model(model_name, *args, **kwargs):
    return get_object(_models, model_name, *args, 
                      print_name = 'models', err = True, **kwargs)

def print_models():
    print_objects(_models, 'models')

_models = {}

__load()