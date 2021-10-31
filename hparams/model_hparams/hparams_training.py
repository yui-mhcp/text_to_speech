from hparams.hparams import HParams

_dataset_config = {
    'batch_size'    : 64,
    'train_batch_size'  : None,
    'valid_batch_size'  : None,
    'test_batch_size'   : 1,
    'shuffle_size'      : 1024
}

HParamsTraining = HParams(   
    ** _dataset_config,
    epochs      = 10,
    
    verbose     = 1,
    
    train_times = 1,
    valid_times = 1,
    
    train_size  = None,
    valid_size  = None,
    test_size   = 4,
    pred_step   = -1
)

HParamsTesting  = HParams(
    ** _dataset_config
)

