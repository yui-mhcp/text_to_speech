
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

import tensorflow as tf

from models.tts import Tacotron2
from datasets import get_dataset, train_test_split

gpus = tf.config.list_physical_devices('GPU')

model_name = "tacotron2_siwis"

print("Tensorflow version : {}".format(tf.__version__))
print("GPU disponibles ({}) : {}".format(len(gpus), gpus))


model = Tacotron2(nom = model_name)

model.compile(
    optimizer = 'adam', 
    optimizer_config = {
        'lr' : {'name' : 'WarmupScheduler', 'maxval' : 1e-3,'minval' : 5e-4, 'factor' : 128, 'warmup_steps' : 512},
    }
)


print(model)

datasets = 'siwis'
dataset = get_dataset(datasets, shuffle = True)

train, valid = None, None

print("Dataset length : {}".format(len(dataset)))

""" Classic hyperparameters """
epochs     = 5
batch_size = 32
valid_batch_size = 2 * batch_size
train_prop = 0.9
train_size = 640 #int(len(dataset) * train_prop)
valid_size = 640 #min(len(dataset) - train_size, 250 * valid_batch_size)

shuffle_size    = 1024
pred_step       = -1 # make a prediction after every epoch
augment_prct    = 0.25

""" Custom training hparams """
trim_audio      = False
reduce_noise    = False
trim_threshold  = 0.075
max_silence     = 0.25
trim_method     = 'window'
trim_mode       = 'start_end'

trim_mel     = False
trim_factor  = 0.6
trim_mel_method  = 'max_start_end'

# These lengths corresponds to approximately 5s audio
# This is the max my GPU supports for an efficient training but is large enough for the SIWIS dataset
max_output_length = 512
max_input_length = 75

""" Training """

# this is to normalize dataset usage so that you can use a pre-splitted dataset or not
# without changing anything in the training configuration
if train is None or valid is None:
    train, valid = train_test_split(
        dataset, train_size = train_size, valid_size = valid_size, shuffle = True
    )

print("Training samples   : {} - {} batches".format(
    len(train), len(train) // batch_size
))
print("Validation samples : {} - {} batches".format(
    len(valid), len(valid) // valid_batch_size
))

model.train(
    train, validation_data = valid, 
    train_size = train_size, valid_size = valid_size, 

    epochs = epochs, batch_size = batch_size, valid_batch_size = valid_batch_size,
    
    max_input_length = max_input_length, max_output_length = max_output_length,
    pred_step = pred_step, shuffle_size = shuffle_size, augment_prct = augment_prct,
    
    trim_audio = trim_audio, reduce_noise = reduce_noise, trim_threshold = trim_threshold,
    max_silence = max_silence, trim_method = trim_method, trim_mode = trim_mode,
    
    trim_mel = trim_mel, trim_factor = trim_factor, trim_mel_method = trim_mel_method,
)