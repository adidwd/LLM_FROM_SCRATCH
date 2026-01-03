from tokenization import SimpleTokenizer
from BPE import SimpleTokenizer2
import re
import os
from input_data import Vocabulary
v_class = Vocabulary() 
vocab = v_class.load_vocabulary()
#print(vocab)
text="""
Egg
"""
tokenizer=SimpleTokenizer(vocab)
ids=tokenizer.encode(text)
print("Encoded IDs:", ids)
decoded_text = tokenizer.decode(ids)
print("Decoded Text:", decoded_text)

tokenizer2 = SimpleTokenizer2()
ids2 = tokenizer2.encode(text)
print("Encoded IDs with SimpleTokenizer2:", ids2)
decoded_text2 = tokenizer2.decode(ids2)
print("Decoded Text with SimpleTokenizer2:", decoded_text2)


with open("verdict.txt","r",encoding="utf-8") as f:
            raw_text=f.read()

enc_text=tokenizer2.encode(raw_text)
print(f"Total tokens in verdict.txt using SimpleTokenizer2: {len(enc_text)}")

