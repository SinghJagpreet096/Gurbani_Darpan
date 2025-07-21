from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama # Local LLM
from query_transformer import QueryEnhancer # Function to rewrite and expand queries
from create_context import Context, contextChroma # Function to create context
from generateEmbeddings import Embedding # Function to get verses from the database
from config import Config # Configuration settings for the application


class Model:
    def __init__(self, model_name: str):
        self.embeddings = Embedding()
        # self.context = Context()
        self.context = contextChroma() # Initialize context with the model name
        self.llm = ChatOllama(
            model = model_name,
        )
        self.enhancer = QueryEnhancer(model_name=model_name) # Initialize query enhancer
        self.prompt = ChatPromptTemplate.from_template("""You are an expert in Gurbani and Sikh theology. Use ONLY the provided shabads from Sri Guru Granth Sahib Ji to answer the question.

                                                            Context (Shabad Extracts):
                                                            {context}

                                                            Question:
                                                            {question}

                                                            Answer:
                                                            Cite shabad, Writer and ang from the above context. 
                                                       Your answer should be respectful, concise, and no longer than 15 lines.
                                                        Include relevant lines from the shabads with a brief interpretation in plain English.
                                                        Do not add external commentary or interpretation beyond the given text.
                                                       """)


    def response(self, query:str) -> str:
        dataset_path = Config().database_path # "backend/db/merged_shabad.parquet"
        embeddings_path = Config().embeddings_path # "backend/db/shabad_embeddings.parquet"
        ## rewrite query
        rewritten_query = self.enhancer.rewrite_query(query) # "Gurbani verses discussing ego and arrogance"
        print(f"rewritten query: {rewritten_query}")
        # self.embeddings.generate_embeddings(dataset_path, embeddings_path,issample=False) # Generate embeddings
        shabads = self.context.provide_context(rewritten_query, top_k=2) 
        print("shabad:", shabads)
        chain = self.prompt | self.llm | StrOutputParser()
        return chain.invoke({"question": query, "context": shabads})

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

