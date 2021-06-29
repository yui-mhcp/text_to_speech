import tensorflow as tf

from custom_train_objects.losses.masked_loss import MaskedMSE, MaskedMAE
from custom_train_objects.losses.ctc_loss import CTCLoss
from custom_train_objects.losses.ge2e_loss import GE2ELoss
from custom_train_objects.losses.tacotron_loss import TacotronLoss

_keras_losses = {
    'BinaryCrossentropy'        : tf.keras.losses.BinaryCrossentropy,
    'binary_crossentropy'       : tf.keras.losses.BinaryCrossentropy,
    'bce'                       : tf.keras.losses.BinaryCrossentropy,

    'CategoricalCrossentropy'   : tf.keras.losses.CategoricalCrossentropy,
    'categorical_crossentropy'  : tf.keras.losses.CategoricalCrossentropy,
    
    'Huber'                     : tf.keras.losses.Huber,
    
    'KLDivergence'              : tf.keras.losses.KLDivergence,
    
    'MeanAbsoluteError'         : tf.keras.losses.MeanAbsoluteError,
    'mean_absolute_error'       : tf.keras.losses.MeanAbsoluteError, 
    'mae'                       : tf.keras.losses.MeanAbsoluteError,
    
    'MeanSquaredError'          : tf.keras.losses.MeanSquaredError,
    'mean_squared_error'        : tf.keras.losses.MeanSquaredError,
    'mse'                       : tf.keras.losses.MeanSquaredError,
    
    'SparseCategoricalCrossentropy'     : tf.keras.losses.SparseCategoricalCrossentropy,
    'sparse_categorical_crossentropy'   : tf.keras.losses.SparseCategoricalCrossentropy
}

_losses = {
    'MaskedMAE'         : MaskedMAE,
    'MaskedMSE'         : MaskedMSE,
    'CTCLoss'           : CTCLoss,
    'GE2ELoss'          : GE2ELoss,
    'TacotronLoss'      : TacotronLoss,
    ** _keras_losses
}
