import os
import glob

def __load():
    for module_name in glob.glob(os.path.join('custom_architectures', 'transformers_arch/*.py')):
        if os.path.basename(module_name) in ['__init__.py']: continue
        module_name = module_name.replace(os.path.sep, '.')[:-3]

        module = __import__(
            module_name, fromlist = ['custom_objects', 'custom_functions']
        )
        if hasattr(module, 'custom_objects'):
            custom_objects.update(module.custom_objects)
        if hasattr(module, 'custom_functions'):
            custom_functions.update(module.custom_functions)

    
custom_objects = {}
custom_functions = {}

__load()

