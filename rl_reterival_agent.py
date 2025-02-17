import numpy as np
import random
from sklearn.linear_model import SGDClassifier
from embedder import embedding_model

class RLRetrievalAgent:
    """
    A retrieval agent that uses reinforcement learning principles and epsilon-greedy exploration
    to select concepts based on an input query embedding. The model is trained incrementally
    using rewards to refine its selection accuracy over time.
    """
    
    def __init__(self, embedding_model):       
        """
        Initializes the RLRetrievalAgent with a sentence embedding model and a logistic regression classifier.
        
        Args:
            embedding_model: The embedding model used to encode queries and concepts.
        """
        self.model = SGDClassifier(loss='log_loss')
        self.epsilon = 0.1  # Exploration rate for epsilon-greedy strategy
        self.embedding_model = embedding_model
        
    def get_features(self, query_embed, concept_embed):
        """
        Concatenates the query embedding and concept embedding to form feature vectors
        for model prediction, ensuring a 2D output shape.
        
        Args:
            query_embed (np.ndarray): Embedding of the query (1D array).
            concept_embed (np.ndarray): Embedding of the concept (1D array).
            
        Returns:
            np.ndarray: A concatenated feature vector of shape (1, embedding_dim * 2).
        """
        if query_embed.ndim == 2:
            query_embed = query_embed.squeeze(0)
        if concept_embed.ndim == 2:
            concept_embed = concept_embed.squeeze(0)
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
        
        # Check if the model is fitted (example check for scikit-learn models)
        model_fitted = hasattr(self.model, 'coef_')
        
        # Handle model not fitted scenario
        if not model_fitted:
            # Force full exploration and shuffle concepts for randomness
            effective_epsilon = 1.0
            concepts = list(concepts)  # Create a copy to shuffle
            random.shuffle(concepts)
        else:
            effective_epsilon = self.epsilon
        
        for concept in concepts:
            concept_embed = self.embedding_model.encode(concept)
            features = self.get_features(query_embed, concept_embed)
            
            # Epsilon-greedy exploration with effective epsilon
            if random.random() < effective_epsilon:
                selected.append(concept)
            else:
                # Only use model if fitted
                prob = self.model.predict_proba([features])[0][1]
                if prob > 0.5:
                    selected.append(concept)
            
            if len(selected) >= k:
                break
                
        return selected[:k]
        
    def update(self, concept, query_embed, reward):
        """
        Updates the model based on the reward received for a specific concept-query pair.
        
        Args:
            concept (str): The concept for which feedback is provided.
            query_embed (np.ndarray): Embedding of the query.
            reward (int): Reward label (0 or 1) indicating relevance.
        """
        concept_embed = self.embedding_model.encode(concept)
        features = self.get_features(query_embed, concept_embed)
        self.model.partial_fit([features], [reward], classes=[0, 1])

rra = RLRetrievalAgent(embedding_model=embedding_model)