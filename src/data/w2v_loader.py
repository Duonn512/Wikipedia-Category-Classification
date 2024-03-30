from gensim.models import KeyedVectors
import numpy as np

class W2VLoader():
    def __init__(self, w2v_path):
        self.embed_model = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
        self.embedding_dim = self.embed_model.vectors.shape[1]

        UNKNOWN_VEC = np.zeros(shape=(1, self.embedding_dim))
        PADDING_VEC = np.zeros(shape=(1, self.embedding_dim))
        # Add '<PAD>' token
        self.embed_model.add_vectors(['<PAD>'], PADDING_VEC)
        # Add '<UNK>' token
        self.embed_model.add_vectors(['<UNK>'], UNKNOWN_VEC)
        self.vocab_size = self.embed_model.vectors.shape[0]
    
    def get_model(self):
        return self.embed_model