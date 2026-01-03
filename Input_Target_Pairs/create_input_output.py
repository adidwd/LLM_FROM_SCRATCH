import torch
from torch.utils.data import Dataset, DataLoader
import sys
import tiktoken
sys.path.append('./')
import input_data
from BPE import SimpleTokenizer2

class GPTDatasetV1(Dataset):
    def __init__(self,txt,tokenizer,max_length,stride=4):
        self.input_ids=[]
        self.target_ids=[]

        #tokenize the whole dataset
        
        token_ids = tokenizer.encode(txt,allowed_special={"<|endoftext|>"}) 
        #max_length=CONTEXT_LENGTH

        for i in range(0,len(token_ids)-max_length,stride):
            input_chunk=token_ids[i:i+max_length]
            output_chunk=token_ids[i+stride:i+max_length+stride]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(output_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    

class create_dataloader_v1():

    def __init__(self):
        self.tokenizer=tiktoken.get_encoding("gpt2")


        
        
    def dataloader(self,txt,batch_size=8, max_length=4, stride=4, shuffle=False,drop_last=True,num_workers=0):
        
        dataset=GPTDatasetV1(txt, self.tokenizer, max_length, stride)
        print(f"==============Dataset length: {len(dataset)}=================")
        print(f"======batches should be == {(len(txt) - max_length) // stride + 1}")

        dataloader=DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers
        )

        return dataloader
    
with open("verdict.txt","r",encoding="utf-8") as f:
    txt=f.read()


print(f"length of txt=={len(txt)}")


#******Following 4 lines are commented since it is for whole data, we need to do this for train and test data seperately****
#dataloader=create_dataloader_v1().dataloader(txt=txt,batch_size=8,max_length=256,stride=4,shuffle=False)
#print(f"=================data loader _v1 done======================")
#data_iter=iter(dataloader)

#input,target=next(data_iter)

#****** Just checking the data for each input***************
#print(f"=================data_iter done======================")
#first_batch=next(data_iter)

#print(f"=================first_batch== {first_batch}")
#second_batch=next(data_iter)
#print(second_batch)


train_ratio=0.9
split_idx=int(train_ratio*len(txt))
train_data=txt[:split_idx]
val_data=txt[split_idx:]

#****** train data being convered to input and output*************
train_dataloader=create_dataloader_v1().dataloader(txt=train_data,batch_size=8,max_length=256,stride=4,shuffle=False)
print(f"===============data loader _v1 done for train data================")
data_iter=iter(train_dataloader)
train_input,train_target=next(data_iter)

#****** validation data being convered to input and output*************
val_dataloader=create_dataloader_v1().dataloader(txt=val_data,batch_size=8,max_length=256,stride=4,shuffle=False)
print(f"=================data loader _v1 done for validation data==========")
data_iter=iter(val_dataloader)

val_input,val_target=next(data_iter)

#****** Just checking the data for each input, output***************

for x,y in train_dataloader:
    print(f"printing the shape of train input and output data =={x.shape}  and  {y.shape}")



for x,y in val_dataloader:
    print(f"printing the shape of validation input and output data =={x.shape}  and  {y.shape}")



#********* Creating embeddings for input and target data, this is not required as it is done later **********************

class Embeddings():
    def __init__(self,vocab_size,output_dim,context_length):
        self.vocab_size=vocab_size
        self.output_dim=output_dim
        self.context_length=context_length
    
    def embeddings(self):
        token_embeddings_layer=torch.nn.Embedding(self.vocab_size,self.output_dim)
        positional_embeddings_layer=torch.nn.Embedding(self.context_length,self.output_dim)

        return token_embeddings_layer,positional_embeddings_layer

#context_length=256
#output_dim=256
#vocab_size=50257
#token_embeddings_layer,positional_embeddings_layer=Embeddings(vocab_size,output_dim,context_length).embeddings()
#token_embeddings=token_embeddings_layer(input)
#pos_embeddings=positional_embeddings_layer(torch.arange(context_length))
#print(token_embeddings.shape)
#print(pos_embeddings.shape)
#input_embedding=token_embeddings+pos_embeddings

#print(input_embedding.shape)

