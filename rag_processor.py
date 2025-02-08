from google.cloud import vision
#from langchain_google_vertexai import VertexAIEmbeddings
import faiss
import numpy as np
from gensim.models import KeyedVectors
import faiss
import numpy as np
import os

class RAGProcessor:
    def __init__(self, glove_path='VideoGeneration/glove.6B.300d.txt'):
        # Load GloVe embeddings
        self.glove = self._load_glove(glove_path)
        self.index = faiss.IndexFlatL2(300)  # GloVe uses 300-dimensional vectors
        self.image_data = []

    def _load_glove(self, file_path):
        """Load pre-trained GloVe embeddings"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"GloVe file not found at {file_path}")
            
        return KeyedVectors.load_word2vec_format(file_path, binary=False, no_header=True)

    def _text_to_vector(self, text):
        """Convert text to average GloVe vector"""
        words = text.lower().split()
        vectors = []
        for word in words:
            if word in self.glove:
                vectors.append(self.glove[word])
        return np.mean(vectors, axis=0) if vectors else np.zeros(300)

    def process_image(self, image_bytes):
        description = self._describe_image(image_bytes)
        embedding = self._text_to_vector(description)
        self._add_to_index(embedding, description)

    def retrieve_context(self, query, k=3):
        query_vec = self._text_to_vector(query).astype('float32')
        distances, indices = self.index.search(np.expand_dims(query_vec, 0), k)
        return [self.image_data[i] for i in indices[0]]

    
    def _describe_image(self, image_bytes):
        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=image_bytes)
        response = client.annotate_image({
            'image': image,
            'features': [vision.Feature.Type.LABEL_DETECTION],
        })
        return ", ".join([label.description for label in response.label_annotations])
    
    def _add_to_index(self, embedding, description):
        vector = np.array(embedding).astype('float32').reshape(1, -1)
        self.index.add(vector)
        self.image_data.append(description)
    
    def retrieve_context(self, query, k=3):
        try:
            clean_query = query.lower().replace(",", " ").strip()
            query_embedding = self._text_to_vector(clean_query)
            query_embedding = query_embedding.astype('float32').reshape(1, -1)
        
            distances, indices = self.index.search(query_embedding, k)
        
            return [self.image_data[i] for i in indices[0]]
        except Exception as e:
            print(f"Error in retrieve_context: {e}")
            return []