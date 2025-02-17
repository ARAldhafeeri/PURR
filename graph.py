import networkx as nx
import numpy as np
from embedder import embedding_model

class Graph:
    def __init__(self, embedding_model):    
      self.embedding_model = embedding_model
      self.knowledge_graph = self._build_knowledge_graph()

    def _build_knowledge_graph(self):
        """ Build the knowledge graph with connection, used to rebuild the graph with every new concept/response"""
        G = nx.DiGraph()
        nodes = [
            ('Claws', {'type': 'Adaptation', 'traits': ['Hunting', 'Defense']}),
            ('Olfactory', {'type': 'Sensory', 'traits': ['Tracking', 'Detection']}),
            ('Hollow Bones', {'type': 'Structure', 'traits': ['Flight', 'Lightweight']}),
            ('Feline Whiskers', {'type': 'Sensory Organ', 'traits': ['Navigation', 'Tactile Sensing']}),
            ('Dental Hygiene', {'type': 'Health Practice', 'traits': ['Disease Prevention', 'Longevity']})
        ]
        G.add_nodes_from(nodes)
    
        for node in G.nodes:
            node_embed = self.embedding_model.encode([node[0]] if isinstance(node, tuple) else node)
            if node_embed is None:
                continue
            G.nodes[node]['embedding'] = node_embed
    
        relationships = self._extract_relationships(nodes)
        G.add_edges_from(relationships)
    
        return G
    
    def _extract_relationships(self, nodes):
        """Define relationship on the fact data set, TODO: intialize nodes edges"""
        relationships = [
            ('Claws', 'Feline', {'weight': 0.95}),
            ('Olfactory', 'Canine', {'weight': 0.92}),
            ('Hollow Bones', 'Avian', {'weight': 0.89}),
            ('Feline Whiskers', 'Low-light Adaptation', {'weight': 0.97}),
            ('Dental Hygiene', 'Periodontal Disease', {'weight': 0.95})
        ]
        return relationships
    
    def graph_reasoning(self, query_embed):
        """ return connected concepts from the graph"""
        similarities = []
        for node in self.knowledge_graph.nodes:
            node_embed = self.knowledge_graph.nodes[node].get('embedding')
            if node_embed is None:
                continue
            sim = np.dot(query_embed, node_embed)
            similarities.append((node, sim))
        return [n[0] for n in sorted(similarities, key=lambda x: x[1], reverse=True)[:3]]

    def update_graph(self, concepts):
        """ Update knowledge graph with new concepts from each prediction """
        for concept in concepts:
            if concept not in self.knowledge_graph:
                # Get the embedding and extract the 1D vector
                concept_embed = self.embedding_model.encode([concept])[0]
                
                # Only attempt to find a best match if there are existing nodes in the graph.
                if self.knowledge_graph.nodes:
                    # Compute the dot product between the concept's embedding and each node's embedding
                    best_match = max(
                        self.knowledge_graph.nodes,
                        key=lambda n: np.dot(
                            concept_embed, 
                            self.knowledge_graph.nodes[n].get('embedding', np.zeros_like(concept_embed))
                        )
                    )
                    # Add the new concept node with its embedding
                    self.knowledge_graph.add_node(concept, embedding=concept_embed)
                    # Connect the new node to the best match with a given weight
                    self.knowledge_graph.add_edge(concept, best_match, weight=0.9)
                else:
                    # If the graph is empty, simply add the concept node
                    self.knowledge_graph.add_node(concept, embedding=concept_embed)
graph = Graph(embedding_model)