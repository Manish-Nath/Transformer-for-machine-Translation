import torch
import torch.nn as nn
import math
from Input_Embedding import ResidualConnection,MultiHeadAttentionBlock,FeedForwardBlock,LayerNormalization,Positional_encoding,INPUT_EMBEDDING
from Encoder import EncoderBlock,Encoder

class Decoder_Block(nn.Module):
    def __init__(self,self_attention_block:MultiHeadAttentionBlock,cross_attention_block,feed_forward_block:FeedForwardBlock,dropout):
        self.self_attention_block=self_attention_block
        self.cross_atttention_block=cross_attention_block
        self.feed_forward_block=feed_forward_block
        self.residual_connections=nn.ModuleList([ResidualConnection(dropout)] for _ in range(3))

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        x=self.residual_connections[0](x,lambda x: self.self_attention_block(x,x,x,tgt_mask))
        x=self.residual_connections[1](x,lambda x: self.cross_attention_block(x,encoder_output,encoder_output,src_mask))
        x=self.residual_connections[2](x,self.feed_forward_block(x))
        return x
    
class Decoder(nn.Module):
    def __init__(self,layers:nn.ModuleList):
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalization()

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        for layer in self.layers:
            x=layer(x,encoder_output,src_mask,tgt_mask)
        return self.norm(x)

class Projection_Layer(nn.Module):
    def __init__(self,d_model,vocab_size):
        super().__init__()
        self.vocab_size=vocab_size
        self.d_model=d_model
        # self.dk=dk
        self.linear1=nn.Linear(d_model,vocab_size)
        # self.linear2=nn.linear(dk,vocab_size)

    def forward(self,x):  #[batch,seq_len,d_model]->[batch,seq_len,vocab_size]
        return torch.log_softmax(self.linear1(x),dim=1)
