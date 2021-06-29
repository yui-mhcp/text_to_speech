import tensorflow as tf

from custom_train_objects.optimizers.lr_schedulers import *

_optimizers = {
    'Adam'          : tf.keras.optimizers.Adam,
    'Ddadelta'      : tf.keras.optimizers.Adadelta,
    'Ddagrad'       : tf.keras.optimizers.Adagrad,
    'Ddam'          : tf.keras.optimizers.Adam,
    'RMSprop'       : tf.keras.optimizers.RMSprop,
    'SGD'           : tf.keras.optimizers.SGD,
}

_schedulers = {
    'DivideByStep'      : DivideByStep, 
    'SinScheduler'      : SinScheduler,
    'TanhDecayScheduler'    : TanhDecayScheduler,
    'WarmupScheduler'   : WarmupScheduler
}