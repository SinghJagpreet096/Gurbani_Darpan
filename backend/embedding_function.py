from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer

class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        TOKENIZERS_PARALLELISM = "false"
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

    def __call__(self, input: Documents) -> Embeddings:
        

        embeddings = self.model.encode(input)

        # embed the documents somehow
        return embeddings

def main():
    embedding = MyEmbeddingFunction()
    d = ["This is an example sentence", "Each sentence is converted"]
    print(d)
    embeddings = embedding(d)
    print(embeddings)   
if __name__ == "__main__":
    main()
