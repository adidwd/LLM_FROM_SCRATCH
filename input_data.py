import re
import os
import sys

sys.path.append('./')

class Vocabulary():

    def __init__(self):
        pass
        

    def load_vocabulary(self):  
        with open("verdict.txt","r",encoding="utf-8") as f:
            raw_text=f.read()

        print(f"total word_count= {len(raw_text)}")
        print(f" first 100 characters == {raw_text[:90]}")

#splitting text in individual words

        preprocessed=re.split(r'([,.;_?!"()]|--|\s)',raw_text)

        result=[item for item in preprocessed if item.strip()]
        print (result[:30])

        allwords=sorted(set(result))
        allwords.extend(['<|endoftext|>', '<|unk|>'])

        vocab = {token:integer for integer,token in enumerate(allwords)}
        print(f"Vocabulary size: {len(vocab)}")
        print(f"last 10 words in vocabulary: {list(vocab.keys())[-5:]}")


        return vocab

