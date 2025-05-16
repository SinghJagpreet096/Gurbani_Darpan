from transformers import pipeline
from typing import List, Dict
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

class QueryEnhancer:
    def __init__(self, model_name: str = "llama3.2"):
        # Initialize models
        self.rewriter = ChatOllama(
            model = model_name,
            temperature=0.2,

            )
        self.expander = ChatOllama(
            model = model_name,
        ) # Replace with Punjabi model if available

    def rewrite_query(self, query: str) -> str:
        """Improves retrieval-focused phrasing"""
        prompt = ChatPromptTemplate.from_template("""Rewrite this Gurbani-related search query 2-3 variants to optimize for retrieving the most relevant shabads.
                                                   Return ONLY the rewritten queries with no additional text:
        Original: {query}
        Rewritten:""")
        # prompt = f"""
        # Rewrite this Gurbani-related query for better document retrieval:
        # Original: {query}
        # Rewritten:"""
        
        rewritter_chain = prompt | self.rewriter
        rewritten = rewritter_chain.invoke({"query": query})
        return rewritten.content.strip()
        
    def expand_query(self, query: str) -> List[str]:
        """Generates semantic variations"""
        prompt = f"""
        Generate 3 Punjabi/English variations of this Gurbani query:
        {query}
        1."""
        
        expansions = self.expander(prompt, max_length=100, num_return_sequences=3)
        return [q.split('\n')[0].strip('123. ') for q in expansions]

if __name__ == "__main__":
    # Example usage

    enhancer = QueryEnhancer(model_name="llama3.2") 
    query = "What does Guru say about ego?"
    rewritten = enhancer.rewrite_query(query) 
    print(rewritten) # "Gurbani verses discussing ego and arrogance"
    # expansions = enhancer.expand_query(query)  # ["haumai references in SGGS", ...]