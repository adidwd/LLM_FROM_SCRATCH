import os
import sys

sys.path.append('./')

cfg={
    "vocab_size":50257,
    "context_length":1024,
    "emb_dim":768,
    "n_heads":12,
    "n_layers":12, #just for trial, it need NOT be same as n_heads
    "drop_rate":0.1,
    "qkv_bias":False,
    "max_new_tokens":6

}


