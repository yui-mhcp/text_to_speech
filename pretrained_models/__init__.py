import os
import json

from models import _models
from utils import load_json

def get_pretrained(model_name):
    config_filename = os.path.join("pretrained_models", model_name, "config.json")
    if not os.path.exists(config_filename):
        raise ValueError("Le modèle pré-entrainé {} n'existe pas !".format(model_name))
    
    model_class = load_json(config_file).get('class_name', None)
    if model_class is None:
        raise ValueError("Wrong configuration in {} : missing 'class_name' information !".format(config_file))
    
    if model_class in _models:
        return _models[model_class](nom = model_name)
    else:
        raise ValueError("Modèle de classe inconnue : {}".format(model_class))

def print_pretrained():
    folders = [f for f in os.listdir('pretrained_models') if os.path.exists(os.path.join('pretrained_models', f, 'config.json'))]
    print("Existing pre-trained models : {}".format(sorted(folders)))
    
def update_models():
    names = [f for f in os.listdir('pretrained_models') if os.path.exists(os.path.join('pretrained_models', f, 'config.json'))]
    
    for name in names:
        print("Update model '{}'".format(name))
        model = get_pretrained(name)
        model.save(save_ckpt = False)
        print(model)
        del model
    