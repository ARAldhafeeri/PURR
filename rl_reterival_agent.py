from sklearn.linear_model import SGDClassifier
import numpy as np
from embedder import embedding_model
import random

class RLRetrievalAgent:
    """
    A retrieval agent that uses reinforcement learning principles and epsilon-greedy exploration
    to select concepts based on an input query embedding. The model is trained incrementally
    using rewards to refine its selection accuracy over time.
    """
    
    def __init__(self, embedding_dim=384):       
        """
        Initializes the RLRetrievalAgent with a sentence embedding model and a logistic regression classifier.
        
        Args:
            embedding_dim (int): Dimensionality of the embeddings. Default is 384.
        """
        self.model = SGDClassifier(loss='log_loss')
        self.embedding_dim = embedding_dim
        self.epsilon = 0.1  # Exploration rate for epsilon-greedy strategy
        self.embedding_model = embedding_model
        
    def get_features(self, query_embed, concept_embed):
        """
        Concatenates the query embedding and concept embedding to form feature vectors
        for model prediction.
        
        Args:
            query_embed (np.ndarray): Embedding of the query.
            concept_embed (np.ndarray): Embedding of the concept.
            
        Returns:
            np.ndarray: A concatenated feature vector of query and concept embeddings.
        """
        return np.concatenate([query_embed, concept_embed])
    
    def select_concepts(self, concepts, query_embed, k=5):
        """
        Selects concepts relevant to a query using epsilon-greedy exploration with a logistic
        regression model. The agent either selects randomly (exploration) or based on
        the model's prediction probability (exploitation).
        
        Args:
            concepts (list[str]): List of concept strings to evaluate.
            query_embed (np.ndarray): Embedding of the query.
            k (int): Maximum number of concepts to select. Default is 5.
        
        Returns:
            list[str]: List of up to `k` selected concepts.
        """
        selected = []
        for concept in concepts:
            concept_embed = self.embedding_model.encode(concept)
            features = self.get_features(query_embed, concept_embed)
            
            # Epsilon-greedy exploration
            if random.random() < self.epsilon:
                selected.append(concept)
            else:
                prob = self.model.predict_proba([features])[0][1]  # Probability of relevance
                if prob > 0.5:  # Select if the probability is above 0.5
                    selected.append(concept)
            
            if len(selected) >= k:
                break
                
        return selected[:k]
    
    def update(self, concept, query_embed, reward):
        """
        Updates the model based on the reward received for a specific concept-query pair.
        The model is refit incrementally using the new data.
        
        Args:
            concept (str): The concept for which feedback is being provided.
            query_embed (np.ndarray): Embedding of the query.
            reward (int): Reward label (0 or 1) indicating whether the concept was relevant (1) or not (0).
        """
        concept_embed = self.embedding_model.encode(concept)
        features = self.get_features(query_embed, concept_embed)
        self.model.partial_fit([features], [reward], classes=[0, 1])
