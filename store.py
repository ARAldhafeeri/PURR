from chromadb import PersistentClient
from embedder import embedding_model
from rl_reterival_agent import rra
from graph import graph
class Store:
    """
      ChromaDB presisted store for active learning, continous learning.
      It's used across the solution for semantic search and presisted memory
      for the
    """
    def __init__(self, embedding_model):
        self.chroma_client = PersistentClient(path="bio_knowledge")
        self.embedding_model = embedding_model
        self.knowledge_base = self._init_chroma()
        self.rl_agent = rra
        self.graph = graph
        
    def _init_chroma(self):
        """Initialze chroma db instance"""
        collection = self.chroma_client.get_or_create_collection(
            "bio_facts",
            metadata={"hnsw:space": "cosine"}
        )
        if collection.count() == 0:
            self._seed_knowledge(collection)
        return collection
    
    def _seed_knowledge(self, collection):
        """Seeds the facts knowledge inside chroma db to be retreived by the AI"""
        embeddings = self.embedding_model.encode(facts)
        collection.add(
            ids=[str(i) for i in range(len(facts))],
            documents=facts,
            embeddings=embeddings
        )

    def neural_retrieval(self, query):
        """Reterive results using sematic search -> graph reasoning -> reinforcement """
        query_embed = self.embedding_model.encode([query])
        chroma_results = self.knowledge_base.query(
            query_embeddings=query_embed.tolist(),
            n_results=5
        )['documents'][0]
        graph_concepts = self.graph.graph_reasoning(query_embed)
        all_concepts = list(set(chroma_results + graph_concepts))
        
        # RL-based selection
        selected_concepts = self.rl_agent.select_concepts(all_concepts, query_embed)
        return selected_concepts

store = Store(embedding_model)