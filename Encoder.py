import torch
import torch.nn as nn
import math
from Input_Embedding import ResidualConnection,MultiHeadAttentionBlock,FeedForwardBlock,LayerNormalization,Positional_encoding,INPUT_EMBEDDING

class EncoderBlock(nn.Module):
    def __init__(self,self_attention_block:MultiHeadAttentionBlock,feed_forward_block:FeedForwardBlock,dropout ):
        super(EncoderBlock).__init__()
        self.self_attention_block=self_attention_block
        self.feed_forward_block=feed_forward_block
        self.residual_connections=nn.ModuleList([ResidualConnection(dropout)] for _ in range(2))
        self.dropout=dropout

    def forward(self,x,src_mask): #src_mask to dont let padidng interact
        x=self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,src_mask))
        x=self.residual_connections[1](x,lambda x: self.feed_forward_block(x))
        return x
class Encoder(nn.Module):
    def __init__(self,layers: nn.ModuleList):       #we want interable encoder block
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalization

    def forward(self,x,mask):
        for layer in self.layers:
            x=layer(x,mask)

        return self.norm(x)
    

