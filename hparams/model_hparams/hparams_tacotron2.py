from hparams.hparams import HParams

HParamsTacotron2Encoder = HParams(
    _prefix     = 'encoder',
    
    embedding_dims  = 512,
    n_convolutions  = 3,
    kernel_size     = 5,
    use_bias        = True,
    
    bnorm           = 'after',
    epsilon         = 1e-5,
    momentum        = 0.1,

    drop_rate       = 0.5,
    activation      = 'relu',
    
    n_speaker       = 1,
    speaker_embedding_dim   = None,
    concat_mode         = 'concat',
    linear_projection   = False,
    
    name    = 'encoder'
)

HParamsTacotron2Prenet  = HParams(
    _prefix = 'prenet',
    
    sizes       = [256, 256],
    use_bias    = False,
    activation  = 'relu', 
    drop_rate   = 0.5,
    deterministic   = False,
    name        = 'prenet'
)

HParamsTacotron2Postnet = HParams(
    _prefix     = 'postnet',
    
    n_convolutions  = 5,
    filters     = 512,
    kernel_size = 5,
    use_bias    = True,
    
    bnorm       = 'after',
    epsilon     = 1e-5,
    momentum    = 0.1,
    
    drop_rate   = 0.5,
    activation  = 'tanh',
    final_activation    = None,
    linear_projection   = False,
    name    = 'postnet'
)

HParamsTacotron2LSA = HParams(
    _prefix     = 'lsa',
    
    attention_dim   = 128,
    attention_filters   = 32,
    attention_kernel_size   = 31,
    probability_function    = 'softmax',
    concat_mode     = 2,
    cumulative      = True,
    name    = 'location_sensitive_attention'
)

HParamsTacotron2Sampler = HParams(
    gate_threshold     = 0.5,
    max_decoder_steps  = 1024,
    early_stopping     = True,
    add_go_frame       = False,
    remove_last_frame  = False,
    
    teacher_forcing_mode    = 'constant',
    init_ratio      = 1.0,
    final_ratio     = 0.75,
    init_decrease_step  = 50000,
    decreasing_steps    = 50000
)

HParamsTacotron2Decoder = HParams(
    n_mel_channels  = 80,
    with_logits     = True,
    n_frames_per_step   = 1,
    pred_stop_on_mel    = False,
    
    attention_rnn_dim  = 1024, 
    p_attention_dropout    = 0.,
        
    decoder_n_lstm     = 1,
    decoder_rnn_dim    = 1024,
    p_decoder_dropout  = 0.
) + HParamsTacotron2Prenet + HParamsTacotron2LSA + HParamsTacotron2Sampler

def HParamsTacotron2(vocab_size = 148, ** kwargs):
    hparams = HParamsTacotron2Encoder + HParamsTacotron2Decoder + HParamsTacotron2Postnet
    return hparams(vocab_size = vocab_size, ** kwargs)
