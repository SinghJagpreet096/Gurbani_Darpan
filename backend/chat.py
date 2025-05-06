from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama # Local LLM

from embedding import Embedding # Function to get verses from the database


class Model:
    def __init__(self, model_name: str):
        self.embeddings = Embedding()
        self.llm = ChatOllama(
            model = model_name,
            # temperature=0.5,
            # max_tokens=1024,
            # top_p=1,
            # frequency_penalty=0,
            # presence_penalty=0,
        )
        self.prompt = ChatPromptTemplate.from_template("""You are a Gurbani expert. Answer using ONLY these verses:
                                                       {context}

                                                       Question: {question}

                                                       Answer in 2-3 sentences and ALWAYS cite Ang numbers:""")


    def response(self, query:str) -> str:
        self.embeddings.generate_embeddings() # Generate embeddings
        verses = self.embeddings.top_k_verse(query, top_k=3) 
        print(verses) # Get verses from previous function
        chain = self.prompt | self.llm | StrOutputParser()
        return chain.invoke({"question": query, "context": verses})
        
if __name__ == "__main__":
    model_name = "llama3.2"
    e = Embedding()
    e.generate_embeddings()
    m = Model(model_name)
    o = m.response("What does Gurbani say about ego?")
    
    print(o)

