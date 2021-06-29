from hparams.hparams import HParams

HParamsTraining = HParams(    
    epochs      = 10,
    batch_size  = 32,
    valid_batch_size    = 32,
    test_batch_size     = 1,
    
    verbose     = 1,
    
    train_times = 1,
    valid_times = 1,
    
    train_size  = None,
    valid_size  = None,
    test_size   = 4,
    pred_step   = -1,
    
    shuffle_size    = 1024
)

HParamsPrediction   = HParams(
    
)

HParamsTesting  = HParams(

)

