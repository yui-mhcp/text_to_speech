import numpy as np
import tensorflow as tf

def naive_ge2e(inp, ids, normalize = True):
    uniques = np.unique(ids)
    nb_speakers = len(uniques)
    embedding_dim = inp.shape[-1]
    
    if normalize: inp = inp / np.linalg.norm(inp, axis = -1, keepdims = True)
    inp = np.reshape(inp, [nb_speakers, -1, embedding_dim])
    
    nb_utterances = inp.shape[1]
    
    centroids = np.mean(inp, axis = 1)
    
    sim_matrix = []
    for speaker_id, speaker in enumerate(inp):
        cs_row = []
        for utt_idx, utterance in enumerate(speaker):
            new_centroids = []
            for i, centroid in enumerate(centroids):
                if i == speaker_id:
                    excl = (np.sum(speaker, axis = 0) - utterance) / (nb_utterances - 1)
                    new_centroids.append(excl)
                else:
                    new_centroids.append(centroid)
            new_centroids = np.array(new_centroids)
            
            new_centroids = new_centroids / np.linalg.norm(new_centroids, axis = 1, keepdims = True)
            cs_row.append(np.dot(np.transpose(np.expand_dims(utterance, 1)), np.transpose(new_centroids)))
        sim_matrix.append(np.array(cs_row))
    return np.squeeze(np.array(sim_matrix))

class GE2ELoss(tf.keras.losses.Loss):
    def __init__(self, init_w = 70.8929, init_b = -4.1807, method = 'softmax', 
                 name = 'ge2e_loss', ** kwargs):
        assert method in ('softmax', 'contrast')
        super(GE2ELoss, self).__init__(name = name, ** kwargs)
        self.init_w = init_w
        self.init_b = init_b
        self.method = method
        
        if method == 'softmax':
            self.loss_fn = self.softmax_loss
        else:
            self.loss_fn = self.contrast_loss
        
        self.w = tf.Variable(init_w, trainable = True)
        self.b = tf.Variable(init_b, trainable = True)
    
    def simple_similarity_matrix(self, speaker_embedded):
        """
            Implementation of the similarity matrix of the equation 5
        """
        nb_speakers = tf.shape(speaker_embedded)[0]
        utterances_per_speaker = tf.shape(speaker_embedded)[1]
        embedding_dim = tf.shape(speaker_embedded)[2]

        # Shape == (nb_speakers, 1, embedded_dim)
        centroids_incl = tf.reduce_mean(speaker_embedded, axis = 1, keepdims = True)
        centroids_incl = centroids_incl / tf.norm(centroids_incl, axis = 2, keepdims = True)

        # Shape == (nb_speakers, speaker_utterances, embedded_dim) == speaker_embedded.shape
        centroids_excl = tf.reduce_sum(speaker_embedded, axis = 1, keepdims = True) - speaker_embedded
        centroids_excl = centroids_excl / tf.cast(utterances_per_speaker - 1, tf.float32)
        centroids_excl = centroids_excl / tf.norm(centroids_excl, axis = 2, keepdims = True)

        # Similarity between each utterance and each centroids (inclusive)
        # Shape == (nb_speakers, utterances_per_speaker, nb_speakers)
        sim_matrix = tf.matmul(speaker_embedded, tf.squeeze(centroids_incl, axis = 1), transpose_b = True)

        return sim_matrix

    def similarity_matrix(self, speaker_embedded):
        """
            Implementation of the similarity matrix with exclusive centroids (equation 9)
        """
        nb_speakers = tf.shape(speaker_embedded)[0]
        utterances_per_speaker = tf.shape(speaker_embedded)[1]
        embedding_dim = tf.shape(speaker_embedded)[2]

        # Shape == (nb_speakers, embedded_dim)
        centroids_incl = tf.reduce_mean(speaker_embedded, axis = 1)
        centroids_incl = tf.l2normalize(centroids_incl, axis = -1)
        # Reshape to have shape (nb_speakers, utterances_per_speaker, nb_speakers, embedding_dim) to multiply with mask
        centroids_incl = tf.reshape(centroids_incl, [1, 1, nb_speakers, embedding_dim])
        centroids_incl = tf.tile(centroids_incl, [nb_speakers, utterances_per_speaker, 1, 1])

        # Shape == (nb_speakers, speaker_utterances, embedded_dim) == speaker_embedded.shape
        centroids_excl = tf.reduce_sum(speaker_embedded, axis = 1, keepdims = True) - speaker_embedded
        centroids_excl = centroids_excl / tf.cast(utterances_per_speaker - 1, tf.float32)
        centroids_excl = tf.l2_normalize(centroids_excl, axis = -1)
        # Reshape to have shape (nb_speakers, utterances_per_speaker, nb_speakers, embedding_dim) to multiply with mask
        centroids_excl = tf.expand_dims(tf.transpose(centroids_excl, [1, 0, 2]), axis = 0)
        centroids_excl = tf.repeat(centroids_excl, nb_speakers, axis = 0)

        # Compute mask (shape = (nb_speakers, utterances_per_speaker, nb_speakers, 1))
        eye = tf.eye(nb_speakers)
        mask = tf.repeat(eye, repeats = utterances_per_speaker, axis = 0)
        mask = tf.reshape(mask, [nb_speakers, utterances_per_speaker, nb_speakers, 1])

        # Shape == (nb_speakers, utterances_per_speakers, nb_speakers, embedding_dim)
        centroids = mask * centroids_excl + (1 - mask) * centroids_incl

        # Shape == (nb_speaker, utterances_per_speaker, 1, embedding_dim)
        speaker_embedded = tf.expand_dims(speaker_embedded, 2)

        # Get Similarity matrix by dot product between  speaker_embedded and centroids
        # Shape == (nb_speakers, utterances_per_speaker, 1, nb_speaker)
        sim_matrix = tf.matmul(speaker_embedded, centroids, transpose_b = True)
        # Shape == (nb_speakers, utterances_per_speaker, nb_speaker)
        sim_matrix = tf.squeeze(sim_matrix, axis = 2)

        return sim_matrix
    
    def softmax_loss(self, idx, similarity_matrix):
        nb_speakers = tf.shape(similarity_matrix)[-1]
        similarity_matrix = tf.reshape(similarity_matrix, [-1, nb_speakers])
        
        return tf.keras.losses.sparse_categorical_crossentropy(
            idx, similarity_matrix, from_logits = True
        )
    
    def contrast_loss(self, idx, similarity_matrix):
        raise NotImplementedError()
        
    def call(self, y_true, y_pred):
        """
            Implementation of the GE2E loss function
            Arguments : 
                - y_true : the ids of the speakers with shape (None, 1)
                    Ids can be any values but, for simplicity, they should be stacked
                    Exemple : [1, 1, 2, 2, 0, 0] are valid ids (3 speakers, 2 utterances)
                - y_pred : list of embedded vectors with shape (None, embedding_dim)
                    y_pred[i] should be an embedded vector of speaker y_pred[i]
            Return : 
                - Scalar : loss
        """        
        uniques, idx = tf.unique(tf.reshape(y_true, [-1]))
        nb_speakers = tf.size(uniques)
        
        # L2 norm preds
        #y_pred = y_pred / (tf.norm(y_pred, axis = -1, keepdims = True) + 1e-6)
        
        # Shape == (nb_speakers, utterances_per_speakers, embedded_dim)
        speaker_embedded = tf.reshape(y_pred, [nb_speakers, -1, tf.shape(y_pred)[-1]])
        
        cos_sim_matrix = self.similarity_matrix(speaker_embedded)
        cos_sim_matrix = cos_sim_matrix * tf.maximum(self.w, 1e-6) + self.b
        
        loss = self.loss_fn(idx, cos_sim_matrix)
        
        return tf.reduce_mean(loss)

    def get_config(self):
        config = super(GE2ELoss, self).get_config()
        config['init_w'] = self.w.value().numpy()
        config['init_b'] = self.b.value().numpy()
        config['method'] = self.method
        return config
    

if __name__ == '__main__':
    # 3 speakers, 2 utterances, embedding_dim = 5
    inputs = np.reshape(np.arange(30, dtype = np.float32), (6, 5))
    ids = np.array([1, 1, 2, 2, 3, 3])

    loss = GE2ELoss()
    
    naive_matrix = naive_ge2e(inputs, ids, normalize = False)
    matrix = loss.similarity_matrix(np.reshape(inputs, [3, 2, 5]))
    
    print("Similarity matrix are equals : {}".format(np.allclose(naive_matrix, matrix)))
    print("Naive cosine similarity matrix :\n {}".format(naive_matrix))
    print("Tensorflow cosine similarity matrix :\n {}".format(matrix))
