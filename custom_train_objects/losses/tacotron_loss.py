import tensorflow as tf

class TacotronLoss(tf.keras.losses.Loss):
    def __init__(self,
                 from_logits        = False, 
                 label_smoothing    = 0,
                 
                 mel_loss   = 'mse',
                 mask_mel_padding   = True,
                 similarity_function    = None,
                 
                 finish_weight      = 1.,
                 not_finish_weight  = 1.,
                 
                 reduction  = 'none', 
                 name   = 'tacotron_loss',
                 **kwargs
                ):
        """
            Loss for the Tacotron-2 model architecture
            
            Arguments : 
                - from_logits / label_smooting  : for the binary_crossentropy loss computed on gates outputs
                - mask_mel_padding  : whether to mask padding or not (in mel losses)
                
                - finish_weight     : weight of binary_crossentropy loss for 1-gate
                - not_finish_weight : weight of binary_crossentropy loss for 0-gate
        """
        super().__init__(name = name, reduction = 'none', **kwargs)
        self.mask_mel_padding   = mask_mel_padding
        self.label_smoothing    = label_smoothing
        self.from_logits        = from_logits
        self.finish_weight      = finish_weight
        self.not_finish_weight  = not_finish_weight
        self.mel_loss   = mel_loss
        self.similarity_function    = similarity_function
        
    @property
    def loss_names(self):
        loss_fn_names = self.mel_loss
        if not isinstance(loss_fn_names, (list, tuple)): loss_fn_names = [loss_fn_names]

        names = ['loss']
        names += ['{}_mel_loss'.format(l) for l in loss_fn_names]
        names += ['{}_mel_postnet_loss'.format(l) for l in loss_fn_names]
        return names + ['gate_loss']
    
    def compute_mel_loss(self, y_true, y_pred, loss = None, mask = None):
        if loss is None: loss = self.mel_loss
        if isinstance(loss, (list, tuple)):
            return [self.compute_mel_loss(y_true, y_pred, l, mask = mask) for l in loss]
        
        if loss == 'similarity':
            if mask is not None: y_pred *= tf.cast(mask, y_pred.dtype)
            similarity = self.similarity_function(tf.stop_gradient(y_true), y_pred)
            return tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                tf.ones_like(similarity), similarity
            ))
        elif callable(loss):
            error   = loss(y_true, y_pred)
        elif loss == 'mse':
            error   = tf.square(y_true - y_pred)
        elif loss == 'mae':
            error   = tf.abs(y_true - y_pred)
        else:
            raise ValueError("Unknown loss : {}".format(loss))
        
        if mask is None:
            return tf.reduce_mean(error)
        
        mask = tf.cast(mask, error.dtype)
        return tf.reduce_sum(error * mask) / (tf.reduce_sum(mask) * tf.cast(tf.shape(error)[-1], mask.dtype))
    
    def call(self, y_true, y_pred):
        """
            Compute the Tacotron-2 loss as following : 
                loss = loss_mel + loss_mel_postnet + loss_gate
                - loss_mel          = MSE on mel-spectrogram (before postnet)
                - loss_mel_postnet  = MSE on mel-spectrogram (after postnet)
                - loss_gate         = binary_crossentropy on gates
            
            Arguments : 
                - y_true : [mel_target, gate_target], expected spectrogram / gate
                - y_pred : [mel, mel_postnet, gates, attention], prediction of Tacotron2
            
            Shapes :
                - mel_target, mel, mel_postnet  : [batch_size, seq_len, n_mel_channels]
                - gate_target, gates    : [batch_size, seq_len]
        """
        mel_target, gate_target = y_true
        mel_pred, mel_postnet_pred, gate_pred = y_pred[:3]
        
        reshaped_gate_target = tf.reshape(gate_target, [-1, 1])
        reshaped_gate_pred   = tf.reshape(gate_pred, [-1, 1])
        
        finish_mask     = reshaped_gate_target * self.finish_weight
        not_finish_mask = (1 - reshaped_gate_target) * self.not_finish_weight
        gate_loss = tf.keras.losses.binary_crossentropy(
            reshaped_gate_target, reshaped_gate_pred, 
            from_logits = self.from_logits, 
            label_smoothing = self.label_smoothing
        )
        gate_loss = gate_loss * finish_mask + gate_loss * not_finish_mask

        mask = 1. - tf.expand_dims(gate_target, -1) if self.mask_mel_padding else None
        
        mel_loss            = self.compute_mel_loss(mel_target, mel_pred, mask = mask)
        mel_postnet_loss    = self.compute_mel_loss(mel_target, mel_postnet_pred, mask = mask)
        
        if not isinstance(mel_loss, list): mel_loss = [mel_loss]
        if not isinstance(mel_postnet_loss, list): mel_postnet_loss = [mel_postnet_loss]
        
        gate_loss = tf.reduce_mean(gate_loss)
        
        loss  = tf.reduce_sum(mel_loss) + tf.reduce_sum(mel_postnet_loss) + gate_loss

        return [loss] + mel_loss + mel_postnet_loss + [gate_loss]
    
    def get_config(self):
        config = super().get_config()
        config['mel_loss']      = self.mel_loss
        config['mask_mel_padding']  = self.mask_mel_padding
        config['label_smoothing']   = self.label_smoothing
        config['finish_weight']     = self.finish_weight
        config['not_finish_weight'] = self.not_finish_weight
        config['from_logits']   = self.from_logits
        return config

