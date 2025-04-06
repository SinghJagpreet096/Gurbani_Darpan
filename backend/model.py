from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage
from config import Config


class Model:
    def __init__(self, model_name: str):
        self.llm = ChatOllama(
            model=model_name,
            temperature=0,
            # other params...
        )

    def generate(self, text: str) -> str:
        messages = [                    # Change below!
    {"role": "user", "content": text},
]
        o = self.llm.invoke(messages)
        return o.content

if __name__ == "__main__":
    model_name = Config().model
    m = Model(model_name)
    o = m.generate("ੴ ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ ਨਿਰਭਉ ਨਿਰਵੈਰੁਅਕਾਲ ਮੂਰਤਿ ਅਜੂਨੀ ਸੈਭੰ ਗੁਰ ਪ੍ਰਸਾਦਿ ॥")
    print(o)