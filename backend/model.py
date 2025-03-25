from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage


class Model:
    def __init__(self, model_name: str):
        self.llm = ChatOllama(
            model=model_name,
            temperature=0,
            # other params...
        )

    def generate(self, text: str) -> str:
        o = self.llm.invoke(text)
        return o.content
# model_name = "llama3.2"
# llm = ChatOllama(
#     model=model_name,
#     # temperature=0,
#     # other params...
# )


# o = llm.invoke("ੴ ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ ਨਿਰਭਉ ਨਿਰਵੈਰੁਅਕਾਲ ਮੂਰਤਿ ਅਜੂਨੀ ਸੈਭੰ ਗੁਰ ਪ੍ਰਸਾਦਿ ॥")
# print(o.content)
if __name__ == "__main__":
    m = Model("llama3.2")
    o = m.generate("ੴ ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ ਨਿਰਭਉ ਨਿਰਵੈਰੁਅਕਾਲ ਮੂਰਤਿ ਅਜੂਨੀ ਸੈਭੰ ਗੁਰ ਪ੍ਰਸਾਦਿ ॥")
    print(o)