import torch
from torch import nn

class Multiheadattention(nn.Module):
    def __init__(self,d_in,d_out,context_length,dropout,num_heads,qkv_bias=False):
        super().__init__()
        assert(d_out%num_heads==0)
        self.d_out=d_out
        self.num_heads=num_heads
        self.head_dim=d_out//num_heads
        self.W_query=nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_Key=nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_Value=nn.Linear(d_in,d_out,bias=qkv_bias)
        self.out_proj=nn.Linear(d_out,d_out)
        self.dropout=nn.Dropout(dropout)
        self.register_buffer("mask",torch.triu(torch.ones(context_length,context_length),diagonal=1))

    
    def forward(self,x):
        b,num_tokens,d_in=x.shape

        keys=self.W_Key(x)
        values=self.W_Value(x)
        queries=self.W_query(x)


        #now I am going to unroll the K,Q,V to include num_heads
        keys=keys.view(b,num_tokens,self.num_heads,self.head_dim) 
        values=values.view(b,num_tokens,self.num_heads,self.head_dim)
        queries=queries.view(b,num_tokens,self.num_heads,self.head_dim) 

        #Transposing Q,K,V from (b,num_tokes,#heads,head_dim) -> (b,#heads,num_tokens,head_dim)
        #This is done so that each head learns for its own embeddings and then we will transpose once again later to original form
        keys=keys.transpose(1,2)
        values=values.transpose(1,2)
        queries=queries.transpose(1,2)

        #attn_score=Q.K_T but 1st 2 elements (batch, num_head) are not required to be transposed since it is batch and num_head
        attn_scores=queries@keys.transpose(2,3)
        # attn_score=(b,#heads,num_tokens,head_dim)*(b,#heads,head_dim,num_tokens)
        #attn_score dim=(b,#heads,num_tokens,num_tokens)

        # create mask but truncated only to num_tokens
        mask_bool=self.mask.bool()[:num_tokens,:num_tokens]
        
        #apply mask to attn_score
        attn_scores.masked_fill_(mask_bool,-torch.inf)

        #apply softmax to get probabilities
        #dimension of attention weights === same as attention scores ==(b,#heads,num_tokens,num_tokens) 
        attn_weights=torch.softmax(attn_scores/(keys.shape[-1]**0.5),dim=-1)
        # now lets calculate context vector = attention_weights @ values
        #(b,#heads,num_tokens,num_tokens)*(b,#heads,num_token, head_dim) =>(b,#heads,num_tokens,head_dim) 
        #shape of context vector = b,num_tokens,num_heads,head_dim and we will take transpose (to flip it back by tokens and then head)
        #(b,#heads,num_tokens, head_dim)->(b,num_tokens,#heads, head_dim)  is done by transpose 1,2 index
        context_vector=(attn_weights@ values).transpose(1,2)

        #now lets combine heads by doing #head*head_dim=d_out, we will flatten head_dim and #heads
        #contiguous function  is needed so that the vectors remain in same memory (otherwise it can cause problems when flattening)
        context_vector=context_vector.contiguous().view(b,num_tokens,self.d_out)
        context_vector=self.out_proj(context_vector)
        return context_vector




