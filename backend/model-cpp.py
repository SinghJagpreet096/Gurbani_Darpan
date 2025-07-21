from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Initialize callback manager
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Configuration
n_gpu_layers = -1  # Try with 1 first if this fails
n_batch = 512  # Reduce to 256 or 128 if you have memory issues
model_path = "/Users/jagpreetsingh/Downloads/llama-3.2-1b-instruct-q2_k.gguf"  # Use absolute path
class model_CPP:
    def __init__(self):
        # self.model_name = model_name
        try:
            self.llm = LlamaCpp(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_batch=n_batch,
                f16_kv=True,
                callback_manager=callback_manager,
                verbose=False,
                n_ctx=2048,  # Context window size
                max_tokens=2000,  # Max tokens to generate
                temperature=0.7,
            )
        except Exception as e:
            print(f"Error initializing LlamaCpp: {e}")
            raise
        
    def res(self, query: str) -> str:
        """
        Generate a response for the given query using the LlamaCpp model.
        """
        
        # Test query
        question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"
        print(self.llm.invoke(query))

if __name__ == "__main__":
    model = model_CPP()
    model.res("What is the capital of France?")  # Example query to test the model
    print("Model initialized successfully.")
    
    # model.res("What is the capital of France?")  # Example query to test the model
    # model.res("What is the capital of France?")  # Example query
    
