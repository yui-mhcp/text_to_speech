# Copyright (C) 2022-now yui-mhcp project author. All rights reserved.
# Licenced under a modified Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import keras
import keras.ops as K

from custom_layers import get_activation
from .text_transformer_arch import TextTransformerEncoder, HParamsTextTransformerEncoder

HParamsBaseBERT = HParamsTextTransformerEncoder(
    pooler_as_output    = False,
    pooler_activation   = 'tanh',
    poolers = -1
)

HParamsBertMLM  = HParamsBaseBERT(
    transform_activation    = 'gelu'
)

HParamsBertClassifier   = HParamsBaseBERT(
    num_classes = -1,
    process_tokens  = False,
    process_first_token = False,
    final_drop_rate = 0.1,
    final_activation    = None,
    classifier_type     = 'dense',
    classifier_kwargs   = {}
)

#HParamsBertEmbedding   = HParamsBaseBERT(
#    ** HParamsEmbeddingHead, process_tokens = True
#)

@keras.saving.register_keras_serializable('transformers')
class BaseBERT(TextTransformerEncoder):
    default_params = HParamsBaseBERT

@keras.saving.register_keras_serializable('transformers')
class BertMLM(BaseBERT):
    default_params = HParamsBertMLM

    def __init__(self, vocab_size, embedding_dim, ** kwargs):
        kwargs['poolers'] = None
        super().__init__(
            vocab_size = vocab_size, embedding_dim = embedding_dim, ** kwargs
        )
        
        self.dense  = keras.layers.Dense(embedding_dim, name = 'dense_transform')
        self.act    = get_activation(self.hparams.transform_activation)
        self.final_norm = keras.layers.LayerNormalization(epsilon = self.hparams.epsilon)
        
    def build(self, input_shape):
        super().build(input_shape)
        self.dense.build((None, None, self.embedding_dim))
        self.final_norm.build((None, None, self.embedding_dim))
        with keras.name_scope(self.name):
            self.bias = self.add_weight(
                shape = [self.hparams.vocab_size], initializer = "zeros", name = "bias"
            )
    
    def compute_output(self, output, mask = None, training = False, ** kwargs):
        output = super().compute_output(output, mask = mask, training = training, ** kwargs)
        
        output = self.dense(output)
        if self.act is not None: output = self.act(output)
        output = self.final_norm(output, training = training and self.norm_training)
        
        return self.embeddings.linear(output) + self.bias

@keras.saving.register_keras_serializable('transformers')
class BertClassifier(BaseBERT):
    default_params  = HParamsBertClassifier
    _attr_to_set    = BaseBERT._attr_to_set + ['process_tokens', 'process_first_token']
    
    def __init__(self, num_classes, vocab_size, embedding_dim, ** kwargs):
        super().__init__(
            vocab_size = vocab_size, embedding_dim = embedding_dim, num_classes = num_classes, ** kwargs
        )
        
        if self.hparams.classifier_type == 'dense':
            self.classifier = keras.layers.Dense(
                num_classes, name = 'classification_layer', ** self.hparams.classifier_kwargs
            )
        elif self.hparams.classifier_type == 'lstm':
            self.classifier = keras.layers.LSTM(
                num_classes, name = 'classification_layer', ** self.hparams.classifier_kwargs
            )
        elif self.hparams.classifier_type == 'bi-lstm':
            self.classifier = keras.layers.Bidirectional(keras.layers.LSTM(
                num_classes, name = 'classification_layer', ** self.hparams.classifier_kwargs
            ))
        elif self.hparams.classifier_type == 'gru':
            self.classifier = keras.layers.GRU(
                num_classes, name = 'classification_layer', ** self.hparams.classifier_kwargs
            )

        self.act        = get_activation(self.hparams.final_activation)
        self.dropout    = keras.layers.Dropout(self.hparams.final_drop_rate)
    
    def build(self, input_shape):
        super().build(input_shape)
        self.classifier.build((None, None, self.embedding_dim))
    
    def compute_output(self, output, mask = None, training = False, ** kwargs):
        output, pooled_out = super(BertClassifier, self).compute_output(
            output, mask = mask, training = training, ** kwargs
        )
        
        if not self.process_tokens:
            output = pooled_out
        elif self.process_first_token:
            output = output[:, 0]
        
        if self.dropout is not None: output = self.dropout(output, training = training)
        output = self.classifier(output, training = training)
        if self.act is not None: output = self.act(output)
            
        return output

@keras.saving.register_keras_serializable('transformers')
class BertNSP(BertClassifier):
    def __init__(self, vocab_size, embedding_dim, num_classes = 2, ** kwargs):
        kwargs.update({'use_pooling' : True, 'process_tokens' : False})
        super().__init__(
            num_classes = 2, vocab_size = vocab_size, embedding_dim = embedding_dim, ** kwargs
        )
    
@keras.saving.register_keras_serializable('transformers')
class BertEmbedding(BaseBERT):
    #default_params  = HParamsBertEmbedding
    _attr_to_set    = BaseBERT._attr_to_set + ['process_tokens']

    def __init__(self, output_dim, vocab_size, embedding_dim, ** kwargs):
        super().__init__(
            vocab_size = vocab_size, embedding_dim = embedding_dim, output_dim = output_dim, ** kwargs
        )
        if 'process_first_token' in kwargs:
            kwargs['token_selector'] = 'first' if kwargs['process_first_token'] else None
        self.embedding_head = EmbeddingHead(** self.hparams)

    def compute_output(self, output, mask = None, training = False, ** kwargs):
        output, pooled_out = super(BertEmbedding, self).compute_output(
            output, mask = mask, training = training, ** kwargs
        )
        if not self.process_tokens: output = pooled_out
        
        return self.embedding_head(output, mask = mask, training = training)

@keras.saving.register_keras_serializable('transformers')
class BertQA(BertClassifier):
    def __init__(self, * args, ** kwargs):
        kwargs.update({'num_classes' : 2, 'process_tokens' : True, 'process_first_token' : False})
        super().__init__(* args, ** kwargs)
    
    def compute_output(self, output, mask = None, training = False, ** kwargs):
        output = super().compute_output(
            output, mask = mask, training = training, ** kwargs
        )

        probs   = tf.nn.softmax(output, axis = 1)
        
        return probs[:, :, 0], probs[:, :, 1]

@keras.saving.register_keras_serializable('transformers')
class DPR(BertEmbedding):
    @classmethod
    def from_pretrained(cls,
                        pretrained_name,
                        pretrained_task = 'question_encoder',
                        pretrained      = None,
                        ** kwargs
                       ):
        if pretrained is None:
            pretrained = transformers_bert(pretrained_name, pretrained_task)
            
        kwargs.setdefault('output_dim', pretrained.config.projection_dim)
        kwargs.update({
            'process_tokens'    : True,
            'token_selector'    : 'first',
            'hidden_layer_type' : 'dense',
            'final_pooling'     : None,
            'use_final_dense'   : False
        })
        
        return BertEmbedding.from_pretrained(
            pretrained_name = pretrained_name,
            pretrained_task = pretrained_task,
            pretrained  = pretrained,
            ** kwargs
        )
    
def transformers_bert(name, task = 'base'):
    import transformers
    if task == 'base':
        return transformers.BertModel.from_pretrained(name)
    elif task == 'question_encoder':
        return transformers.DPRQuestionEncoder.from_pretrained(name)
    elif task in ('lm', 'mlm'):
        return transformers.BertForMaskedLM.from_pretrained(name)
    elif task == 'nsp':
        return transformers.BertForNextSentencePrediction.from_pretrained(name)
    else:
        raise ValueError("Unknown task !\n  Accepted : {}\n  Got : {}".format(
            tuple(_transformers_pretrained_task.keys()), task
        ))

