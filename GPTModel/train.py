import torch
import os
from torch import nn
from GPTModel import GPTModel
from config.cfg import cfg
import torch
from torch import nn
import os
import sys
import tiktoken
sys.path.append('./')
from Transformerblock import TransformerBlock
from Layernormalization import LayerNorm
from Input_Target_Pairs import create_dataloader_v1
from config.cfg import cfg
from Multihead_attention import Multiheadattention
from Layernormalization import LayerNorm
from Feedforward import FeedForward
from Calculate_loss import calc_loss
from BPE import SimpleTokenizer2
from Generatenexttokens import Generatetext
cfg=cfg

tokenizer = SimpleTokenizer2()


def evaluate_model(model,train_dataloader,val_dataloader,device,eval_iter):
    model.eval()

    loss_calculator = calc_loss()
    with torch.no_grad():
        train_loss=loss_calculator.calculate_loss_loader(train_dataloader,model,device, num_batches=eval_iter)
        val_loss=loss_calculator.calculate_loss_loader(val_dataloader,model,device, num_batches=eval_iter)
    
    model.train()

    return train_loss, val_loss




def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    
    # 1. Encode the text (returns a list)
    # 2. Convert the list to a torch.Tensor
    # 3. Move the Tensor to the device
    # 4. Add a batch dimension (unsqueeze) because the model expects (Batch, Seq_Len)
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).to(device).unsqueeze(0) 
    generator = Generatetext(model, cfg)

    with torch.no_grad():
        
        # Pass the tensor, not the list
        token_ids = generator.generate(idx=encoded_tensor, temperature=0.75)
    
    # squeeze(0) removes the batch dimension, and .tolist() converts it back for the tokenizer
    decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())

    print(decoded_text.replace("\n",""))
    model.train()






def main():
    
    
    


    with open("verdict.txt","r",encoding="utf-8") as f:
        txt=f.read()


    print(f"length of txt=={len(txt)}")

    


    train_ratio=0.9
    split_idx=int(train_ratio*len(txt))
    train_data=txt[:split_idx]
    val_data=txt[split_idx:]

    #****** train data being convered to input and output*************
    train_dataloader=create_dataloader_v1().dataloader(txt=train_data,batch_size=16,max_length=256,stride=4,shuffle=False,num_workers=4, pin_memory=True)
    print(f"===============data loader _v1 done for train data================")


    #****** validation data being convered to input and output*************
    val_dataloader=create_dataloader_v1().dataloader(txt=val_data,batch_size=16,max_length=256,stride=4,shuffle=False,num_workers=4, pin_memory=True)
    print(f"=================data loader _v1 done for validation data==========")

    torch.manual_seed(123)
    model=GPTModel(cfg)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"device ======================== {device}")

    model.to(device)
    loss_calculator = calc_loss()
    optimizer=torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.1)
    num_epochs=10
    train_loss,val_loss,tokens_seen=train_model(model,train_dataloader,val_dataloader,optimizer,device,num_epochs=num_epochs,eval_freq=50,eval_iter=8,start_context="Every effort moves you", tokenizer=tokenizer,loss_calculator=loss_calculator)
    #with torch.no_grad():
    #    train_loss=loss_calculator.calculate_loss_loader(train_dataloader,model,device)
    #    val_loss=loss_calculator.calculate_loss_loader(val_dataloader,model,device)

    #print(f"training loss ======================= {train_loss}")
    #print(f"validation loss ======================= {val_loss}")


def train_model(model,train_dataloader,val_dataloader,optimizer,device,num_epochs,eval_freq,eval_iter,start_context,tokenizer, loss_calculator):
    train_loss,val_loss,track_tokens_seen=[],[],[]
    tokens_seen, global_step=0,-1
    

    #model.to(device)
    #loss_calculator = calc_loss()

    for epoch in range(num_epochs):
        model.train()

        for input_batch,target_batch in train_dataloader:
            optimizer.zero_grad() ## resetting loss grad at every new batch
            loss= loss_calculator.calculate_loss(input_batch,target_batch,model,device)
            loss.backward() ## this calculates loss gradients
            optimizer.step() ## this updates the model weights after each batch
            tokens_seen+=input_batch.numel() # numel gives num of tokens in input batch
            global_step+=1

            if global_step% eval_freq==0:
                train_loss_value,val_loss_value=evaluate_model(model,train_dataloader,val_dataloader,device,eval_iter)
                train_loss.append(train_loss_value)
                val_loss.append(val_loss_value)
                track_tokens_seen.append(tokens_seen)

                # Use the _value variables here so you only see the CURRENT loss
                print(f"Epoch {epoch+1} (Step {global_step:4d}): Train Loss {train_loss_value:.4f} | Val Loss {val_loss_value:.4f}")

        #Trying out the output from training after each epoch
        generate_and_print_sample(model,tokenizer,device,start_context)

    return train_loss, val_loss, track_tokens_seen
        

if __name__ == "__main__":
    main()



