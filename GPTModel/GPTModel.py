import torch
from torch import nn
import os
import sys
sys.path.append('./')
from Transformerblock import TransformerBlock
from Layernormalization import LayerNorm
from Input_Target_Pairs import create_dataloader_v1
from config.cfg import cfg
from Multihead_attention import Multiheadattention
from Layernormalization import LayerNorm
from Feedforward import FeedForward



cfg=cfg
class GPTModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.tok_emb=nn.Embedding(cfg["vocab_size"],cfg["emb_dim"]) #change it with input_output later
        self.pos_emb=nn.Embedding(cfg["context_length"],cfg["emb_dim"]) #change it with input_output later
        self.drop_emb=nn.Dropout(cfg["drop_rate"])
        self.trf_blocks=nn.Sequential(
            *[TransformerBlock(cfg) for _ in range (cfg["n_layers"])]
        ) # This is taking 12 sequence of transformer block in sequence and going to output at n_layer as output (12 in our case)
        # * is used to unpack the list which is created in that line because nn.sequence needs it as arguments and not list
        
        self.final_norm=LayerNorm(cfg["emb_dim"])
        self.out_head=nn.Linear(cfg["emb_dim"],cfg["vocab_size"],bias=False)

    def forward(self,in_idx):
        batch_size,seq_len=in_idx.shape
        tok_embeds=self.tok_emb(in_idx)
        pos_embeds=self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x=tok_embeds+pos_embeds
        x=self.drop_emb(x)
        x=self.trf_blocks(x)
        x=self.final_norm(x)
        logits=self.out_head(x)
        return logits



