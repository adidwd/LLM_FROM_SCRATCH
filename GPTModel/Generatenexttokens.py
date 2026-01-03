import torch
import os
from torch import nn
from GPTModel import GPTModel
from config.cfg import cfg
cfg=cfg
class Generatetext(nn.Module):
    def __init__(self,model,cfg):
        super().__init__()
        self.max_new_tokens=cfg["max_new_tokens"]
        self.context_length=cfg["context_length"]
        self.model=model
    
    def generate(self,idx, temperature=0.8):

        for _ in range(self.max_new_tokens):
            idx_cond=idx[:,-self.context_length:]

            
            with torch.no_grad():
                logits=self.model(idx_cond) # dim==#batch, n_tokens,vocab_size
            
            #We are getting the last logit for prediction for a new word, internally it predicts at each logit though

            logits=logits[:,-1,:]

            #print(f"logits ======== {logits}")

            #Apply siftmax to get probability
            probas=torch.softmax(logits/temperature, dim=-1)

            print(f"probas size ======== {probas.shape}")
            print(f"probas ======== {probas}")

            #Get the index with max probability
            #idx_next=torch.argmax(probas,dim=-1, keepdim=True)
            idx_next = torch.multinomial(probas, num_samples=1)

            #print(f"idx_next ======== {idx_next}")

            #Append sampled index to running sequence

            idx=torch.cat((idx,idx_next),dim=1)
            #print(f"IDX shape ==== {idx.shape}")
        
        return idx
    

#torch.manual_seed(123)
#model=GPTModel(cfg)
#batch_size=2
#batch = torch.randint(0, cfg["vocab_size"], (batch_size, cfg["context_length"])) 
#batch = torch.randint(0, cfg["vocab_size"], (batch_size, 10)) 
#out=model(batch)
#model.eval()
#idx=batch
#Generatetext=Generatetext(model,cfg)
#out=Generatetext.generate(idx)

#print(f"Input batch == {batch} ========")
#print(f"Input shape == {batch.shape} ========")
#print(f"Output shape ==== {out.shape}")


#print(f"===============Generate text Output  ===={out}")




