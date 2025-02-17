import torch
from transformers import pipeline
from rl_reterival_agent import rra
from graph import graph
from store import store
class Reasoner:
    """
      Agentic AI can enable LLMs to reason with pre-trained models
      Also in training via reverse engineering the Algorithm
    """
    def __init__(self, embedding_model):
        self.model_name = "meta-llama/Llama-3.2-1B-Instruct"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.llm  = pipeline(
                "text-generation",
                model=self.model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
            )
        self.rl_agent = rra
        self.store = store
        self.graph = graph
        self.embedding_model = embedding_model

     

    def _reward(self, query, final_response, concepts ):
        """Reward if the score is high"""
        query_embed = self.embedding_model.encode([query])[0]
        reward = self._evaluate_response(final_response, query)
        for concept in concepts:
            self.rl_agent.update(concept, query_embed, int(reward >= 4))
            

    def _evaluate_response(self, response, query):
        """Evaluate the response, we can use more capabale A.I for better performance: during training"""
        prompt = f"""Rate the biological accuracy of this response to '{query}' (1-5):
        Response: {response}
        Score:"""
        # IMPORTANT NOTE : here we can implement distillation 
        output = self.llm(prompt, max_new_tokens=10)[0]["generated_text"]
        try:
            return int(output.strip().split()[0][0])
        except:
            return 3
            

    def _llm_generate_thought(self, concept, concept_type, traits, user_query):
        """Uses the llm as agentic AI to anlayze the concept to the query, and knwoledge graph to produce a thought"""
        prompt_messages = [
            {"role": "system", "content": "You are a biological expert conducting a critical reasoning and analysis of biological concepts. Your analysis and reasoning should be structured and scientifically rigorous. Present your reasoning step-by-step."},
            {"role": "user", "content": f"""Analyze the facts given the question question: '{user_query}' from first principles.

Concept Description: {concept}
{'Type: ' + concept_type if concept_type else 'Type: N/A'}
{'Traits: ' + ', '.join(traits) if traits else 'Traits: N/A'}
"""}
        ]

        output = self.llm(
            prompt_messages,
            max_new_tokens=200,
        )
        output = output[0]["generated_text"][-1]
        return output['content']

    def _llm_generate_final_response(self, analysis_chain):
        """"Generate final answer based on the concepts analyzes"""
        prompt_messages = [
            {"role": "system", "content": "You are a scientific summarizer. Synthesize a rigorous, concise, and non-speculative biological conclusion based on the following analyses. Present the synthesis in a reasoned and structured manner using clear and accessible language. Directly output the synthesized conclusion."},
            {"role": "user", "content": f"""Synthesize a biological conclusion from these analyses:\n{analysis_chain}"""}
        ]

        output =  self.llm(
            prompt_messages,
            max_new_tokens=1000,
        )
        return output[0]["generated_text"][-1]["content"]

    def chat(self, query):
        """Create response of reasoning and final answer """
        
        concepts = self.store.neural_retrieval(query)
        analysis_chain = "<biological_analysis>\n"
        for concept in concepts:
            details = self.graph.knowledge_graph.nodes.get(concept, {})
            analysis = self._llm_generate_thought(
                concept,
                details.get('type'),
                details.get('traits', []),
                query
            )
            # print("ann", analysis)
            analysis_chain += f"<concept name='{concept}'>\n{analysis}\n</concept>\n"
        analysis_chain += "</biological_analysis>"

        final_response = self._llm_generate_final_response(analysis_chain)
        self.graph.update_graph(concepts)
        # reward 
        self._reward(query, final_response, concepts)
        return analysis_chain, final_response