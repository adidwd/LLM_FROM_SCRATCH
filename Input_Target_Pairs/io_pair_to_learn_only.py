
import sys
sys.path.append('./')
import input_data
from BPE import SimpleTokenizer2
try:
    confing_size = int(input("Enter the context_size (number of input words): "))
except ValueError:
    print("Invalid input. Using default context_size of 50.")
    confing_size = 50  # Default value
with open("verdict.txt","r",encoding="utf-8") as f:
            raw_text=f.read()
tokenizer2 = SimpleTokenizer2()
ids2 = tokenizer2.encode(raw_text[:100])  # Limiting to first 100 characters for brevity
print("Encoded IDs with SimpleTokenizer2:", ids2)
decoded_text2 = tokenizer2.decode(ids2)
print("Decoded Text with SimpleTokenizer2:", decoded_text2)

for i in range(1,len(ids2)-1):
    print(f"i==={i}") 
    input_ids=ids2[:i]
    target_id=ids2[i]
    print(f"input id:=> {input_ids} and target_kokens=> {target_id}")
    input_text=tokenizer2.decode(input_ids)
    target_text=tokenizer2.decode([target_id])
    print(f"Input Text: {input_text} => Target Text: {target_text}")
