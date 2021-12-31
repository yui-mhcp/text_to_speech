import os
import glob

def __load():
    for test_filename in glob.glob('test/*.py'):
        if '__init__' in test_filename: continue
            
        test_filename = test_filename.replace(os.path.sep, '.')[: -3]

        module = __import__(test_filename)

__load()
