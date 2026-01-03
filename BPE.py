import tiktoken
import importlib
import sys

sys.path.append('./')

class SimpleTokenizer2():
    def __init__(self, text=""):
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.text = text
    def encode(self,text):
        return self.tokenizer.encode(text)
    def decode(self,ids):
        return self.tokenizer.decode(ids)

