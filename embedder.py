from sentence_transformers import SentenceTransformer
class Embedder:
    """
      Sentence Transformers SBERT used in Graph, Store, R1
      for unified encoder to calculate similarity scores
    """
    def __init__(self):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

embedder = Embedder()
embedding_model = embedder.embedding_model