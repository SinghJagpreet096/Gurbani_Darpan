from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama # Local LLM
from query_transformer import QueryEnhancer # Function to rewrite and expand queries
from create_context import Context # Function to create context
from generate_embeddings import Embedding # Function to get verses from the database


class Model:
    def __init__(self, model_name: str):
        self.embeddings = Embedding()
        self.context = Context()
        self.llm = ChatOllama(
            model = model_name,
        )
        self.enhancer = QueryEnhancer(model_name=model_name) # Initialize query enhancer
        self.prompt = ChatPromptTemplate.from_template("""You are a Gurbani expert. Answer using ONLY these verses:
                                                       {context}

                                                       Question: {question}

                                                       Cite and explain the verse(s) in detail first in English and then in Punjabi.
                                                       Answer:""")


    def response(self, query:str) -> str:
        dataset_path = "backend/db/merged_shabad.parquet"
        embeddings_path = "backend/db/shabad_embeddings.parquet"
        ## rewrite query
        rewritten_query = self.enhancer.rewrite_query(query) # "Gurbani verses discussing ego and arrogance"
        print(f"rewritten query: {rewritten_query}")
        # self.embeddings.generate_embeddings(dataset_path, embeddings_path,issample=False) # Generate embeddings
        verses, metadata = self.context.provide_context(rewritten_query, top_k=1) 
        print("verse:", verses)
        print(f"{'-'*50}\n metadata:{metadata}") # Get verses from previous function
        chain = self.prompt | self.llm | StrOutputParser()
        return chain.invoke({"question": query, "context": verses})
        
if __name__ == "__main__":
    model_name = "llama3.2"
    # embedding = Embedding()
    # # embedding.is_embedding_exist()
    # dataset_path = "backend/db/merged_shabad.parquet"
    # embeddings_path = "backend/db/shabad_embeddings.parquet"
    # embedding.generate_embeddings(dataset_path, embeddings_path, issample=True)
    m = Model(model_name)
    o = m.response("Solution to loneliness and depression")
    
    print(o)

