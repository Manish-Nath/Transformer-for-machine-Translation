

import torch
import torch.nn as nn
import math
from architecture import INPUT_EMBEDDING, ResidualConnection, MultiHeadAttentionBlock, FeedForwardBlock, LayerNormalization, Positional_encoding
from Encoder import EncoderBlock, Encoder
from decoder import Decoder, Decoder_Block, Projection_Layer

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embd: INPUT_EMBEDDING, tgt_embd: INPUT_EMBEDDING, 
                 src_pos: Positional_encoding, tgt_pos: Positional_encoding, projection: Projection_Layer):
        super(Transformer, self).__init__()  # Proper nn.Module init
        self.encoder = encoder
        self.decoder = decoder
        self.src_embd = src_embd
        self.tgt_embd = tgt_embd
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection  # Corrected reference
    
    def encode(self, src, src_mask):
        src = self.src_embd(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        tgt = self.tgt_embd(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)

# Factory method to build the transformer model
def build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, d_model=512, N=6, h=8, dropout=0.1, d_ff=2048):
    # Create embedding layers
    src_embed = INPUT_EMBEDDING(d_model, src_vocab_size)
    tgt_embed = INPUT_EMBEDDING(d_model, tgt_vocab_size)

    src_pos = Positional_encoding(d_model, src_seq_len, dropout)
    tgt_pos = Positional_encoding(d_model, tgt_seq_len, dropout)

    # Encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_ff, d_model, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_ff, d_model, dropout)
        decoder_block = Decoder_Block(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Projection layer for the output
    projection_layer = Projection_Layer(d_model, tgt_vocab_size)

    # Create the transformer model
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize model parameters with Xavier uniform distribution
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)  # Use underscore for in-place initialization

    return transformer

 
        

